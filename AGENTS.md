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
  | Tools        | `Tool`         | `BashTool`, `ReadFileTool`, `WriteFileTool` |
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
- `AnthropicProvider` streams (`stream: true`, SSE) and reassembles the events into
  one `CompletionResponse`. `Provider::complete_streaming` also reports text pieces,
  which the loop emits as `Event::AssistantTextDelta` before the whole
  `AssistantText` (default impl: no deltas, so Copilot is unchanged). The CLI
  splits streamed text into Markdown blocks (`BlockSplitter`) and prints each
  block once complete, and skips the repeated `AssistantText`. Dropping the
  request future closes the connection.
- `AnthropicProvider` turns on automatic prompt caching (top-level
  `cache_control`) and reports `Usage` (including cache read/write tokens) on
  each response; the agent sums it per turn into `TurnFinished`, and the CLI
  prints a `tokens:` line. Keep the prompt prefix (tools, system, history)
  byte-stable between requests or the cache misses. Copilot reports usage from its `assistant.usage` events (its `input_tokens` includes cached tokens, so the provider subtracts them).
- Cancellation: `AgentLoop::run_turn` takes a `CancellationToken`. Cancelling
  drops the in-flight model request or tool call and returns
  `Ok(StopReason::Cancelled)`; unrun tool calls get an error `tool_result`, the
  turn ends with an assistant `[Cancelled by user]` message, so the session stays
  valid. `Executor::run` must stop its process when its future is dropped;
  `LocalExecutor` kills the whole process group. `ExecRequest::timeout` does the
  same on a deadline (`ExecOutput::timed_out`). `BashTool` defaults to 120s, the
  model can pass `timeout_seconds` (max 1800). The CLI cancels on Ctrl-C, and a
  second Ctrl-C exits. Copilot's pending-call state after a cancel is untested.
- Compaction: Copilot keeps the history and compacts it itself (infinite
  sessions, enabled explicitly in `new_session`; background at 80%, blocking at
  95%). The provider reports it: `CompletionResponse::{context, compactions}`
  become `Event::ContextUsage` / `Event::Compacted`, and the CLI prints a
  `context: used / limit (%)` line each turn. `/compact` calls
  `AgentLoop::compact` -> `Provider::compact` (Copilot: `history.compact` RPC;
  other providers error). alx's own message list is never compacted, so a
  resumed or replayed session sends the full transcript to a fresh Copilot
  session. Anthropic has no compaction yet (its server-side one is beta).
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
  `--list-sessions` lists them. The header records the session's provider, so it
  resumes on the same one. Copilot keeps provider-side state that is not
  restored on resume; only the message history is.
- ACP (Agent Client Protocol): `alx-acp` serves an `AgentFactory` + `SessionStore`
  over ACP (`alx_acp::serve(agents, store, transport)`), using the
  `agent-client-protocol` SDK. It calls the core directly (function calls);
  ACP is only its outside. `alx_acp::transport` has the ready-made transports:
  `native(agents, store)` (in-process channel, returns the client end; used by
  the CLI and `crates/alx-acp/tests/in_process.rs`) and `serve_stdio` for
  `alx --acp` (editors). HTTP/WebSocket are still to come; any SDK `ConnectTo`
  works with `serve`. The prompt reply names the model in `_meta.model`.
  One server is not tied to a provider: every ACP session gets its own agent from
  the `AgentFactory` (`SettingsAgentFactory` builds them from `Settings`; the
  default is `settings.provider`, and `settings.model` applies to it only). The
  session's provider is a `provider` select config option (in the new / load /
  resume replies, model in `_meta.model`); `session/set_config_option` switches it
  between turns and the choice is saved with the session. The CLI's
  `/provider [name]` shows or switches it, and `--provider` with `--continue` /
  `--resume` switches the resumed session. Likewise a `model` select option
  (category `model`) lists `Provider::list_models` (Copilot: the SDK's model
  list; Anthropic: `GET /v1/models`, needs the key), with `default` meaning the
  provider's default. Setting it rebuilds the agent with
  `AgentFactory::build(provider, Some(model))`, refuses ids the provider doesn't
  list (any id if it lists none), and saves `Session.model`; switching provider
  resets it. The CLI has `/model [id]`, and `--model` switches a resumed session.
  The CLI's `/clear` (aliases `/reset`, `/new`) starts a new ACP session on the
  same provider and model; the old one stays stored and resumable.
  Typing `/` at the prompt lists the slash commands as a gray hint
  (`alx-cli/src/commands.rs`, a rustyline helper); Tab completes, → accepts.
  `Event`s map to `session/update` (text deltas, tool calls, context usage);
  `session/cancel` cancels the turn's token; `session/load` replays history;
  `session/list` (paged, optional `cwd` filter; message count in
  `_meta.messageCount`) and `session/resume` (no replay) are served from the
  `SessionStore`.
  `Compacted` becomes a `compaction_update`; turn `Usage` goes in
  `PromptResponse.usage` (unstable SDK feature). A `/compact` prompt (advertised
  via `available_commands_update`) runs `AgentLoop::compact`. Each turn runs
  through `AgentLoop::run_turn_with(.., TurnOptions)`: the ACP server passes a
  `ClientExecutor` (client terminal / fs when the client advertises them, else
  local) and a `ClientApprover` (`session/request_permission` before every
  non-`read_only` tool; "always" answers remembered per tool name). The CLI
  answers every permission request with `allow_always`, so tools run unasked
  there.
- v1 scope: single agent, local execution, no MCP.

## Layout

- `crates/alx-protocol`: wire types shared by core and clients (messages, tool specs, events)
- `crates/alx-acp`: ACP server over `alx-core` (see above)
- `crates/alx-core`: agent loop, traits, built-in implementations, and
  `agent_from_settings` / `SettingsAgentFactory` / `load_dotenv`, the default wiring for binaries
- `crates/alx-cli`: the `alx` binary, an interactive REPL with line editing and history (`rustyline`: arrows, Home/End, Up/Down);
  `markdown.rs` renders assistant text (Markdown via `pulldown-cmark`) as ANSI-styled terminal text, word-wrapped to the terminal width (tables shrink and wrap cells), plain when color is off;
  `render.rs` formats events compactly (one line per tool call, full output with `--verbose`; ANSI color on a TTY, off with `NO_COLOR`)
- `alx-cli` is an ACP client: it reaches the agent through `alx-acp`'s native
  transport, and `client.rs` turns `session/update`s back into `Event`s for the
  renderer. `--list-sessions`, `--continue` (newest `session/list` entry for
  the cwd) and `--resume` (`session/resume`) all go over ACP; the CLI only
  builds the agent and store to hand to the transport, so listing needs a
  working provider config.

## Working on it

- The toolchain is pinned in `rust-toolchain.toml` (Rust 1.99.1, with rustfmt and
  clippy); rustup installs it on first use, and CI uses the same one. Bump it there
  and in the workspace `rust-version`.
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
    it is rejected in files. It is checked on the first request, not at startup,
    so listing sessions or using Copilot needs no key.
  - A new setting means: a field in `Settings`, its `merge` line, its env var in
    `from_env_with`, and a flag if it is worth one.
