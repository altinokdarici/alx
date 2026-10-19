# alx

A headless coding agent harness, written in Rust. It started as a
TypeScript prototype (`agent.ts`), which has since been removed; see git
history if you need it as a reference.

## Principles

- **Headless first.** The core is a library (and later a daemon) with no UI.
  It never prints; it emits `Event`s through an `EventSink`. Every UI (CLI,
  TUI, web, Discord, ...) is a separate client that only consumes events.
- **Everything swappable.** Each layer sits behind a trait in `alx-core`:

  | Layer        | Trait          | Built-in               |
  |--------------|----------------|------------------------|
  | LLM backend  | `Provider`     | `AnthropicProvider`, `CopilotProvider` |
  | Tools        | `Tool`         | `BashTool`             |
  | Where commands run | `Executor` | `LocalExecutor` (Docker/remote later) |
  | Agent loop   | `AgentLoop`    | `SingleAgent` (multi-agent later) |
  | Persistence  | `SessionStore` | `FileSessionStore`, `InMemorySessionStore` |
  | UI           | `EventSink`    | `CliRenderer` (`alx-cli`) |

- Tools get the executor through `ToolContext` and never own one, so
  swapping the executor moves every tool with it.
- Protocol types are provider-neutral. Each `Provider` maps them to its own
  wire format.
- `CopilotProvider` (cargo feature `copilot`, on by default) drives GitHub
  Copilot through `github-copilot-sdk` and the user's installed `copilot`
  CLI. Copilot runs its own agent loop, so the provider strips it to a bare
  model: `ClientMode::Empty`, alx's system prompt, no built-in tools, and
  alx's tools registered declaration-only. A model tool call pauses Copilot
  and comes back as a `ToolUse`; alx's loop runs it and the next `complete`
  resolves the pending call. The SDK's CLI download at build time is turned
  off in `.cargo/config.toml`.
- `AnthropicProvider` turns on automatic prompt caching (top-level
  `cache_control`) and reports `Usage` (including cache read/write tokens) on
  each response; the agent sums it per turn into `TurnFinished`, and the CLI
  prints a `tokens:` line. Keep the prompt prefix (tools, system, history)
  byte-stable between requests or the cache misses. Copilot reports no usage.
- Cancellation: `AgentLoop::run_turn` takes a `CancellationToken`. Cancelling
  drops the in-flight model request or tool call and returns
  `Ok(StopReason::Cancelled)`; unrun tool calls get an error `tool_result`, the
  turn ends with an assistant `[Cancelled by user]` message, so the session stays
  valid. `Executor::run` must stop its process when its future is dropped;
  `LocalExecutor` kills the whole process group. `ExecRequest::timeout` does the
  same on a deadline (`ExecOutput::timed_out`). `BashTool` defaults to 120s, the
  model can pass `timeout_seconds` (max 1800). The CLI cancels on Ctrl-C, and a
  second Ctrl-C exits. Copilot's pending-call state after a cancel is untested.
- Tool output size: `BashTool` cuts stdout and stderr to 30000 bytes each
  (`MAX_OUTPUT_BYTES`), keeping the first quarter and last three quarters with a
  `[... N bytes truncated ...]` marker, because results are re-sent on every
  request. Other tools must bound their own output.
- `./AGENTS.md` in the working directory, if present, is appended to the system
  prompt (in `agent_from_settings`). No search up the tree, no toggle yet.
- Sessions are saved after every turn by `FileSessionStore` (one JSONL file per
  session: a header line, then one message per line) in `session_dir`
  (default `~/.local/share/alx/sessions`). `alx --continue` resumes the latest
  session started in the current directory, `--resume <id>` a specific one,
  `--list-sessions` lists them. Copilot keeps provider-side state that is not
  restored on resume; only the message history is.
- v1 scope: single agent, local execution, no MCP.

## Layout

- `crates/alx-protocol`: wire types shared by core and clients (messages, tool specs, events)
- `crates/alx-core`: agent loop, traits, built-in implementations, and
  `agent_from_env` / `load_dotenv`, the default wiring for binaries
- `crates/alx-cli`: the `alx` binary, an interactive line-based REPL;
  `render.rs` formats events compactly (one line per tool call, full output with `--verbose`; ANSI color on a TTY, off with `NO_COLOR`)
- `alx-cli` calls `alx-core` directly, in-process. A headless server and the
  transport clients use to reach it are still to be designed.

## Working on it

- `cargo test` must pass. Core logic is tested with fake `Provider`/`Executor`
  implementations, with no network needed.
- `cargo clippy --all-targets` and `cargo fmt` must both be clean.
- A pre-commit hook (`.cargo-husky/hooks/pre-commit`) runs `cargo fmt --check` and
  `cargo clippy --all-targets -D warnings`. `cargo-husky` installs it
  automatically on the first `cargo test`. CI (`.github/workflows/ci.yml`)
  runs the same checks plus `cargo test`.
- Commits: end every commit message with a blank line and the trailer
  `Co-authored-by: alx <alx@altinok.dev>`. Subjects use conventional prefixes
  (`feat(alx):`, `chore:`, `docs(alx):`, `refactor:`); see `git log` for examples.
- `./run.sh` starts the REPL with `--provider copilot`; `cargo run` starts it
  with whatever the settings select.
- Settings (`alx-core/src/settings.rs`) are layered, each overriding the one
  before: user file (`~/.config/alx/config.toml`) -> project file (`./alx.toml`,
  or `--config`) -> env vars -> CLI flags. Files are TOML; unknown keys are errors.
  - `provider` (`anthropic` default | `copilot`), `model`, `session_dir`, `[anthropic] base_url`,
    `[copilot] cli_path`, `home`.
  - Env: `ALX_PROVIDER`, `ALX_MODEL`, `ALX_SESSION_DIR`, `ALX_ANTHROPIC_BASE_URL`, `COPILOT_CLI_PATH`,
    `COPILOT_HOME`. Flags: `--provider`, `--model`, `--config`, `--session-dir`.
  - `CLAUDE_API_KEY` (env or `.env`) is the only way to give the Anthropic key;
    it is rejected in files.
  - A new setting means: a field in `Settings`, its `merge` line, its env var in
    `from_env_with`, and a flag if it is worth one.
