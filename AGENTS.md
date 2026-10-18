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
  | Persistence  | `SessionStore` | `InMemorySessionStore` |
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
- v1 scope: single agent, local execution, no MCP.

## Layout

- `crates/alx-protocol`: wire types shared by core and clients (messages, tool specs, events)
- `crates/alx-core`: agent loop, traits, built-in implementations, and
  `agent_from_env` / `load_dotenv`, the default wiring for binaries
- `crates/alx-cli`: the `alx` binary, an interactive line-based REPL
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
- `./run.sh` starts the REPL, always with the Copilot provider. `cargo run`
  starts it with whatever the environment or `.env` selects:
  - `ALX_PROVIDER`: `anthropic` (default, needs `CLAUDE_API_KEY`) or
    `copilot` (needs the `copilot` CLI, logged in; `COPILOT_CLI_PATH` and
    `COPILOT_HOME` override where it is and where its state lives).
  - `ALX_MODEL` overrides the model for either provider.
