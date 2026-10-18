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
  | LLM backend  | `Provider`     | `AnthropicProvider`    |
  | Tools        | `Tool`         | `BashTool`             |
  | Where commands run | `Executor` | `LocalExecutor` (Docker/remote later) |
  | Agent loop   | `AgentLoop`    | `SingleAgent` (multi-agent later) |
  | Persistence  | `SessionStore` | `InMemorySessionStore` |
  | UI           | `EventSink`    | `CliRenderer` (`alx-cli`), `JsonLines` (`alxd`) |

- Tools get the executor through `ToolContext` and never own one, so
  swapping the executor moves every tool with it.
- Protocol types are provider-neutral. Each `Provider` maps them to its own
  wire format.
- v1 scope: single agent, local execution, no MCP.

## Layout

- `crates/alx-protocol`: wire types shared by core and clients (messages, tool specs, events)
- `crates/alx-core`: agent loop, traits, built-in implementations, and
  `agent_from_env` / `load_dotenv`, the shared wiring used by every binary
- `crates/alx-cli`: the `alx` binary, an interactive line-based REPL
- `crates/alxd`: the headless agent. JSON lines over stdio for now: requests
  `{"text": "..."}` on stdin, `alx-protocol` events on stdout. The long-term
  transport is still open.
- Both binaries call `alx-core` directly, in-process.

## Working on it

- `cargo test` must pass. Core logic is tested with fake `Provider`/`Executor`
  implementations, with no network needed.
- `cargo clippy --all-targets` and `cargo fmt` must both be clean.
- `./run.sh` or `cargo run --bin alx` starts the REPL; `cargo run --bin alxd`
  runs headless. Both read `CLAUDE_API_KEY` (from the environment or
  `.env`); `ALX_MODEL` overrides the model.
