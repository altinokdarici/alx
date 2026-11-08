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
  | Tools        | `Tool`         | `BashTool`, `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `GrepTool`, `GlobTool`, `ListDirTool`, `WebFetchTool`, `TodoWriteTool` |
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
  `AgentLoop::compact` -> `Provider::compact` (Copilot: `history.compact` RPC,
  for a provider that `compacts_itself`). Other providers get alx's own
  compaction (`compaction.rs`, pi's numbers): when the context passes
  `context_window()` minus 16,384 tokens (checked before each request; Anthropic
  says 200k), or on `/compact [instructions]`, the messages before a cut that
  keeps about 20,000 recent tokens (a user message, else an assistant one, never
  tool results) are summarized by the model. `Session::compaction` stores the
  cut and summary (in the session file header); `messages` keeps everything,
  and `Session::request_messages()` is the summary plus the kept messages. A
  later compaction folds the previous summary in. Under 2,000 tokens to remove
  does nothing. The agent also emits `ContextUsage` from `Usage` for these
  providers.
- Steering: `TurnOptions::steering` is a queue (`Steering`) of user messages
  for the running turn. The loop adds queued messages to the next tool-results
  message, and when the model would end the turn (`EndTurn`) it continues with
  them instead. Over ACP, a `session/prompt` that arrives while a turn runs is
  queued that way and answered (`end_turn`) when the turn ends; if the turn
  ended without taking it, it runs as its own turn. The CLI steers on a TTY
  (Unix): during a turn `steer.rs` polls fd 0 (raw `libc::read`, so rustyline
  loses no buffered input) and each line typed is sent as a prompt. Piped
  stdin doesn't steer. Typed echo may interleave with streamed output.
- Tool calls in one model response run concurrently: approvals are asked in
  order first, then the approved calls run together (`join_all`) and the
  results go back in call order. If any call's tool returns
  `Tool::sequential()`, the whole batch runs one at a time. `write_file` and
  `edit_file` lock the file (`tools::file_lock`, keyed by the lexically
  normalized path) so calls on the same file run one after another. Cancelling
  cancels every running call. Clients must match `ToolCallFinished` to its
  `ToolCallStarted` by id, since starts and finishes interleave.
- Retries: a provider returns `TransientError` (overload, 408/409/429/5xx,
  send failures; not quota/billing, not `x-should-retry: false`) only before
  any of the response is reported. `SingleAgent` retries it up to
  `AgentConfig::max_retries` (4) times, waiting `retry-after` or 2s, 4s, 8s...
  (cap 60s; a longer server wait fails), and emits `Event::Retrying`. Over ACP
  that is a thought chunk with `_meta["alx/retrying"]`; the CLI prints it.
- Truncated responses: tool calls in a response that stopped on `max_tokens`
  are not run; each gets an error result telling the model to re-issue it with
  complete arguments, and the loop continues. Anthropic tolerates the cut-off
  JSON input in that case.
- Images: `ContentBlock::Image{media_type, data}` (base64). A tool returns
  them in `ToolOutput.images`; the agent puts them right after its
  `ToolResult`, and an image there belongs to that result. `read_file` returns
  PNG/JPEG/GIF/WebP (by extension, checked by magic bytes, max 3.75 MB) as an
  image via `Executor::read_bytes`. Anthropic folds such images into the
  `tool_result` content; Copilot sends them as `binaryResultsForLlm`, and user
  images as `Attachment::Blob`. ACP prompt images (`prompt_capabilities.image`)
  become user image blocks. Copilot's replay transcript drops images.
- `web_fetch` (`tools/web.rs`) GETs an http(s) URL with reqwest (30s
  timeout, body read up to 5 MB), converts HTML to text with a small built-in
  converter (no extra crate), and pages output in 50000-byte windows by
  `offset`. Runs on the host, not through the executor.
- `todo_write` (`tools/todo.rs`) takes the whole task list each call and keeps
  no state; the list lives in the conversation. ACP sends each call as a
  `plan` update; the CLI prints the list under the call.
- Tool output size: `BashTool` cuts stdout and stderr to 30000 bytes each
  (`MAX_OUTPUT_BYTES`), keeping the first quarter and last three quarters with a
  `[... N bytes truncated ...]` marker, because results are re-sent on every
  request. Other tools must bound their own output.
- `edit_file` (`tools/edit.rs`) replaces exact strings instead of rewriting a
  file. Several `edits` in one call are all matched against the original text
  (not the result of earlier edits), must be unique (unless `replace_all`) and
  must not overlap; any failure leaves the file untouched. BOM and mostly-CRLF
  line endings are kept (matched as LF). A miss that only differs in trailing
  whitespace or typographic quotes/dashes gets a "near match at line N" hint;
  no fuzzy replacing. Lessons taken from pi's `edit` tool.
- `grep`, `glob`, `list_dir` (`tools/search.rs`, all `read_only`) run small
  bash scripts through the executor (args as `$1..`, never spliced into code).
  They prefer `rg` (`--no-config`, respects .gitignore), falling back to
  `grep -E` / `git ls-files` / `find`. `grep` searches from inside a directory
  so `glob` filters are relative to it, then re-prefixes paths; it caps
  matches (`limit`, default 100), line length and bytes. `glob` matches a
  pattern without `/` against file names at any depth, else the relative path
  (`globset`). The fallbacks are tested by running the tests with no `rg` on
  `PATH`.
- System prompt (`context.rs`, wired in `agent_from_settings`): the base
  prompt, then an `<environment>` block (working directory, OS, today's UTC
  date; no time, so the prompt cache holds all day), then context files in
  `<project_context>`: `~/.config/alx/AGENTS.md`, then one file per directory
  from `/` down to the cwd (`AGENTS.md`, else `CLAUDE.md`), most specific last.
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
  (`alx-cli/src/commands.rs`, a rustyline helper): its own commands plus the
  agent's `available_commands_update`. Tab completes, → accepts. After `/model `
  or `/provider `, values match any part of their id or name (`opu`), and Enter
  takes a unique partial match (`commands::resolve`).
  A bare `/model` or `/provider` on a terminal opens an inline picker
  (`alx-cli/src/picker.rs`, crossterm raw mode): ↑/↓ move, typing filters,
  Enter picks, Esc cancels; off a terminal it prints the listing instead.
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
