//! Terminal rendering of core events, kept compact: a tool call is one line
//! (`▸ bash ls -la  ✓ 12 lines`), the "thinking" hint is erased when the
//! answer arrives, and full tool output is only shown with `--verbose`.
//!
//! `Renderer::handle` turns an event into text and holds the little state this
//! needs, so it can be tested without a terminal. `CliRenderer` only prints.

use alx_core::EventSink;
use alx_core::protocol::Event;
use std::io::{IsTerminal, Write};
use std::sync::Mutex;

/// With `--verbose`, tool output longer than this many lines is cut.
const MAX_OUTPUT_LINES: usize = 20;
/// Failed tool calls show at most this many lines of output.
const MAX_ERROR_LINES: usize = 5;
/// Clears the current terminal line.
const CLEAR_LINE: &str = "\r\x1b[2K";

/// ANSI styling that degrades to plain text when color is off.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub color: bool,
}

impl Style {
    /// Color only on a real terminal, and never when `NO_COLOR` is set.
    pub fn detect() -> Self {
        let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Self { color }
    }

    pub(crate) fn paint(&self, code: &str, s: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_owned()
        }
    }

    pub fn bold(&self, s: &str) -> String {
        self.paint("1", s)
    }
    pub fn dim(&self, s: &str) -> String {
        self.paint("2", s)
    }
    pub fn red(&self, s: &str) -> String {
        self.paint("31", s)
    }
    pub fn green(&self, s: &str) -> String {
        self.paint("32", s)
    }
    pub fn yellow(&self, s: &str) -> String {
        self.paint("33", s)
    }
    pub fn cyan_bold(&self, s: &str) -> String {
        self.paint("1;36", s)
    }
}

/// The startup banner.
pub fn banner(style: &Style, provider: &str, model: Option<&str>) -> String {
    format!(
        "{} {}\n{}\n",
        style.cyan_bold("alx"),
        style.dim(&format!(
            "· provider: {provider} · model: {}",
            model.unwrap_or("provider default")
        )),
        style.dim("Type 'exit' or press Ctrl-D to quit.")
    )
}

/// The input prompt.
pub fn prompt(style: &Style) -> String {
    format!("{} ", style.green("❯"))
}

/// A one-line error notice.
pub fn error_line(style: &Style, msg: &str) -> String {
    format!("{} {msg}", style.red("✗ error:"))
}

fn indent(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|l| format!("{prefix}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Cuts `s` to `max` characters, ending in `…` when something was dropped.
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let cut: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{cut}…")
}

/// The terminal width: the real size when stdout is a terminal, else `COLUMNS`, else 100.
fn width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| usize::from(w.0))
        .or_else(|| std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()))
        .filter(|&c| c >= 40)
        .unwrap_or(100)
}

/// The one-line description of a tool call, e.g. `ls -la` (+2 lines).
fn call_summary(name: &str, input: &serde_json::Value) -> String {
    let detail = match input.get("input").and_then(|v| v.as_str()) {
        Some(script) if name == "bash" => script.to_owned(),
        _ => input.to_string(),
    };
    let mut lines = detail.lines().filter(|l| !l.trim().is_empty());
    let first = lines.next().unwrap_or("").trim();
    let more = lines.count();
    let room = width().saturating_sub(name.len() + 16);
    let mut out = clip(first, room);
    if more > 0 {
        out.push_str(&format!(" (+{more} lines)"));
    }
    out
}

fn plural(n: usize) -> String {
    format!("{n} line{}", if n == 1 { "" } else { "s" })
}

#[derive(Default)]
struct State {
    /// A transient "thinking" hint is on the current line.
    thinking: bool,
    /// A tool call line was printed without its newline, awaiting the result.
    open_call: bool,
    last_model: Option<String>,
    /// Streamed text waiting for its block to be complete.
    splitter: crate::markdown::BlockSplitter,
    /// Deltas were seen since the last model request, so the whole
    /// `AssistantText` that follows them is already on screen.
    streamed: bool,
    /// The `●` marker of the current answer has been printed.
    answer_started: bool,
    /// The latest context window fill level, shown when the turn finishes.
    context: Option<alx_core::protocol::ContextUsage>,
}

/// Turns events into terminal text.
pub struct Renderer {
    style: Style,
    verbose: bool,
    state: Mutex<State>,
}

impl Renderer {
    pub fn new(style: Style, verbose: bool) -> Self {
        Self {
            style,
            verbose,
            state: Mutex::default(),
        }
    }

    /// The text to print for `event` (it carries its own newlines), or `None`.
    pub fn handle(&self, event: &Event) -> Option<String> {
        let style = &self.style;
        let mut st = self.state.lock().unwrap();
        // Whatever is half-printed on the current line has to be dealt with first.
        let before = if st.thinking {
            CLEAR_LINE
        } else if st.open_call && !matches!(event, Event::ToolCallFinished { .. }) {
            "\n"
        } else {
            ""
        };
        st.thinking = false;

        // Anything else ends the streamed text: show what is left of it.
        let mut flushed = String::new();
        if !matches!(
            event,
            Event::AssistantTextDelta { .. } | Event::AssistantText { .. }
        ) {
            if matches!(event, Event::TurnStarted) {
                // Left over from a turn that failed: don't show it late.
                st.splitter = Default::default();
            }
            let rest = st.splitter.finish();
            flushed = self.blocks(&mut st, &rest);
            st.streamed = false;
            st.answer_started = false;
        }

        let body = match event {
            Event::ModelRequest if style.color => {
                st.thinking = true;
                return Some(format!("{before}{flushed}{}", style.dim("… thinking")));
            }
            Event::ModelRequest | Event::TurnStarted => None,
            Event::AssistantTextDelta { text } => {
                st.streamed = true;
                let done = st.splitter.push(text);
                let shown = self.blocks(&mut st, &done);
                (!shown.is_empty()).then_some(shown)
            }
            // Already shown, block by block, as it streamed in.
            Event::AssistantText { .. } if st.streamed => None,
            Event::AssistantText { text } => {
                let text = crate::markdown::render(text.trim(), style, width().saturating_sub(2));
                let text = indent(&text, "  ");
                let text = text.strip_prefix("  ").unwrap_or(&text);
                Some(format!("\n{} {text}\n", style.cyan_bold("●")))
            }
            Event::ToolCallStarted { name, input, .. } => {
                st.open_call = true;
                Some(format!(
                    "{} {}",
                    style.yellow(&format!("▸ {name}")),
                    style.bold(&call_summary(name, input))
                ))
            }
            Event::ToolCallFinished {
                output, is_error, ..
            } => {
                st.open_call = false;
                Some(self.finished(output, *is_error))
            }
            Event::ContextUsage(c) => {
                st.context = Some(*c);
                None
            }
            Event::Compacted(c) => Some(format!("{}\n", style.dim(&compaction_line(c)))),
            Event::TurnFinished {
                model,
                usage,
                stop_reason,
            } => {
                let mut lines = Vec::new();
                if *stop_reason == alx_core::protocol::StopReason::Cancelled {
                    lines.push("cancelled".to_owned());
                }
                if let Some(model) = model
                    && st.last_model.as_deref() != Some(model)
                {
                    st.last_model = Some(model.clone());
                    lines.push(format!("model: {model}"));
                }
                if let Some(usage) = usage {
                    lines.push(usage_line(usage));
                }
                if let Some(c) = st.context.take() {
                    lines.push(context_line(&c));
                }
                if lines.is_empty() {
                    None
                } else {
                    Some(format!("{}\n", style.dim(&lines.join("\n"))))
                }
            }
        };
        match body {
            Some(body) => Some(format!("{before}{flushed}{body}")),
            None if !before.is_empty() || !flushed.is_empty() => Some(format!("{before}{flushed}")),
            None => None,
        }
    }

    /// Renders streamed Markdown blocks: the first behind the `●` marker,
    /// the rest under it, separated by a blank line.
    fn blocks(&self, st: &mut State, blocks: &[String]) -> String {
        let style = &self.style;
        let mut out = String::new();
        for block in blocks {
            let text = crate::markdown::render(block.trim(), style, width().saturating_sub(2));
            let text = indent(&text, "  ");
            if st.answer_started {
                out.push_str(&format!("  \n{text}\n"));
            } else {
                let text = text.strip_prefix("  ").unwrap_or(&text);
                out.push_str(&format!("\n{} {text}\n", style.cyan_bold("●")));
                st.answer_started = true;
            }
        }
        out
    }

    fn finished(&self, output: &str, is_error: bool) -> String {
        let style = &self.style;
        let lines: Vec<&str> = output.lines().collect();
        let status = if is_error {
            style.red("✗")
        } else {
            style.green("✓")
        };
        let note = match lines.len() {
            0 => String::new(),
            n => style.dim(&format!(" {}", plural(n))),
        };
        let mut out = format!("  {status}{note}\n");
        // Errors always show some output; successes only when verbose.
        let max = match (self.verbose, is_error) {
            (true, _) => MAX_OUTPUT_LINES,
            (false, true) => MAX_ERROR_LINES,
            (false, false) => 0,
        };
        if max > 0 {
            let bar = style.dim("  │ ");
            for l in lines.iter().take(max) {
                let l = if is_error {
                    style.red(l)
                } else {
                    (*l).to_owned()
                };
                out.push_str(&format!("{bar}{l}\n"));
            }
            if lines.len() > max {
                out.push_str(&format!(
                    "{}\n",
                    style.dim(&format!("  … {} more lines", lines.len() - max))
                ));
            }
        }
        out
    }
}

/// Prints core events to stdout.
pub struct CliRenderer {
    renderer: Renderer,
}

impl CliRenderer {
    pub fn new(style: Style, verbose: bool) -> Self {
        Self {
            renderer: Renderer::new(style, verbose),
        }
    }
}

impl EventSink for CliRenderer {
    fn emit(&self, event: Event) {
        if let Some(text) = self.renderer.handle(&event) {
            let mut out = std::io::stdout().lock();
            let _ = write!(out, "{text}");
            let _ = out.flush();
        }
    }
}

/// `tokens: 12.3k in (11.0k cached, 0.4k new cache) / 0.5k out`
fn usage_line(u: &alx_core::protocol::Usage) -> String {
    let total_in = u.input_tokens + u.cache_read_tokens + u.cache_write_tokens;
    format!(
        "tokens: {} in ({} cache read, {} cache write) / {} out",
        compact(total_in),
        compact(u.cache_read_tokens),
        compact(u.cache_write_tokens),
        compact(u.output_tokens)
    )
}

/// `context: 62.0k / 200.0k (31%)`
fn context_line(c: &alx_core::protocol::ContextUsage) -> String {
    let percent = (c.used_tokens * 100)
        .checked_div(c.limit_tokens)
        .unwrap_or(0);
    format!(
        "context: {} / {} ({percent}%)",
        compact(c.used_tokens),
        compact(c.limit_tokens)
    )
}

/// `compacted: 12 messages removed, 150.0k -> 8.0k tokens`
fn compaction_line(c: &alx_core::protocol::Compaction) -> String {
    let tokens = match (c.tokens_before, c.tokens_after) {
        (Some(b), Some(a)) => format!(", {} -> {} tokens", compact(b), compact(a)),
        (None, Some(a)) => format!(", now {} tokens", compact(a)),
        _ => String::new(),
    };
    format!("compacted: {} messages removed{tokens}", c.messages_removed)
}

fn compact(n: u64) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PLAIN: Style = Style { color: false };

    fn call(r: &Renderer, script: &str) -> String {
        r.handle(&Event::ToolCallStarted {
            id: "1".into(),
            name: "bash".into(),
            input: json!({"input": script}),
        })
        .unwrap()
    }

    fn finish(r: &Renderer, output: &str, is_error: bool) -> String {
        r.handle(&Event::ToolCallFinished {
            id: "1".into(),
            name: "bash".into(),
            output: output.into(),
            is_error,
        })
        .unwrap()
    }

    #[test]
    fn tool_call_is_one_line_with_a_summary() {
        let r = Renderer::new(PLAIN, false);
        let out = call(&r, "ls -la") + &finish(&r, "a\nb\nc\n", false);
        assert_eq!(out, "▸ bash ls -la  ✓ 3 lines\n");
    }

    #[test]
    fn multi_line_scripts_show_only_the_first_line() {
        let r = Renderer::new(PLAIN, false);
        assert_eq!(call(&r, "cd x\nmake\nmake test"), "▸ bash cd x (+2 lines)");
    }

    #[test]
    fn verbose_shows_output_and_truncates() {
        let r = Renderer::new(PLAIN, true);
        call(&r, "seq 30");
        let output = (0..30)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let s = finish(&r, &output, false);
        assert!(s.starts_with("  ✓ 30 lines\n  │ 0\n"));
        assert!(s.ends_with("  … 10 more lines\n"));
        assert_eq!(s.lines().count(), 1 + 20 + 1);
    }

    #[test]
    fn errors_show_a_few_lines_even_when_not_verbose() {
        let r = Renderer::new(PLAIN, false);
        call(&r, "false");
        let output = (0..9)
            .map(|i| format!("e{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let s = finish(&r, &output, true);
        assert!(s.starts_with("  ✗ 9 lines\n  │ e0\n"));
        assert!(s.ends_with("  … 4 more lines\n"));
    }

    #[test]
    fn empty_output_has_no_line_count() {
        let r = Renderer::new(PLAIN, false);
        call(&r, "true");
        assert_eq!(finish(&r, "", false), "  ✓\n");
    }

    #[test]
    fn interrupted_call_line_is_terminated() {
        let r = Renderer::new(PLAIN, false);
        call(&r, "sleep 1");
        let s = r
            .handle(&Event::AssistantText { text: "hi".into() })
            .unwrap();
        assert!(s.starts_with("\n\n● hi"));
    }

    #[test]
    fn assistant_text_hangs_its_continuation_lines() {
        let r = Renderer::new(PLAIN, false);
        let s = r
            .handle(&Event::AssistantText {
                text: "a\nb".into(),
            })
            .unwrap();
        assert_eq!(s, "\n● a\n  b\n");
    }

    #[test]
    fn thinking_hint_is_erased_by_the_next_event() {
        let r = Renderer::new(Style { color: true }, false);
        assert!(r.handle(&Event::ModelRequest).unwrap().contains("thinking"));
        let s = r
            .handle(&Event::AssistantText { text: "hi".into() })
            .unwrap();
        assert!(s.starts_with(CLEAR_LINE));
        // And not shown at all without a terminal.
        assert!(
            Renderer::new(PLAIN, false)
                .handle(&Event::ModelRequest)
                .is_none()
        );
    }

    #[test]
    fn model_is_only_shown_when_it_changes() {
        let r = Renderer::new(PLAIN, false);
        let done = || Event::TurnFinished {
            stop_reason: alx_core::protocol::StopReason::EndTurn,
            model: Some("m1".into()),
            usage: None,
        };
        assert_eq!(r.handle(&done()).unwrap(), "model: m1\n");
        assert!(r.handle(&done()).is_none());
    }

    #[test]
    fn color_adds_ansi_only_when_enabled() {
        assert!(Style { color: true }.red("x").contains("\x1b[31m"));
        assert_eq!(PLAIN.red("x"), "x");
    }

    #[test]
    fn usage_is_shown_every_turn() {
        let r = Renderer::new(PLAIN, false);
        let done = Event::TurnFinished {
            stop_reason: alx_core::protocol::StopReason::EndTurn,
            model: None,
            usage: Some(alx_core::protocol::Usage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_tokens: 2000,
                cache_write_tokens: 400,
            }),
        };
        let expected = "tokens: 2.5k in (2.0k cache read, 400 cache write) / 50 out\n";
        assert_eq!(r.handle(&done).unwrap(), expected);
        assert_eq!(r.handle(&done).unwrap(), expected);
    }

    #[test]
    fn context_is_shown_once_per_turn_and_compaction_at_once() {
        use alx_core::protocol::{Compaction, ContextUsage};
        let r = Renderer::new(PLAIN, false);
        assert!(
            r.handle(&Event::ContextUsage(ContextUsage {
                used_tokens: 62_000,
                limit_tokens: 200_000
            }))
            .is_none()
        );
        let done = Event::TurnFinished {
            stop_reason: alx_core::protocol::StopReason::EndTurn,
            model: None,
            usage: None,
        };
        assert_eq!(r.handle(&done).unwrap(), "context: 62.0k / 200.0k (31%)\n");
        assert!(r.handle(&done).is_none());
        let c = Event::Compacted(Compaction {
            messages_removed: 12,
            tokens_before: Some(150_000),
            tokens_after: Some(8_000),
        });
        assert_eq!(
            r.handle(&c).unwrap(),
            "compacted: 12 messages removed, 150.0k -> 8.0k tokens\n"
        );
    }

    fn delta(r: &Renderer, t: &str) -> Option<String> {
        r.handle(&Event::AssistantTextDelta { text: t.into() })
    }

    #[test]
    fn streamed_text_shows_per_block_and_the_whole_text_is_not_repeated() {
        let r = Renderer::new(PLAIN, false);
        r.handle(&Event::ModelRequest);
        assert!(delta(&r, "Hello **wor").is_none());
        assert!(delta(&r, "ld**\n").is_none());
        assert_eq!(delta(&r, "\nSecond ").unwrap(), "\n● Hello world\n");
        assert!(delta(&r, "para").is_none());
        // The whole text arrives after the deltas: nothing more to show.
        assert!(
            r.handle(&Event::AssistantText {
                text: "Hello **world**\n\nSecond para".into()
            })
            .is_none()
        );
        // The next event flushes the unfinished block, under the same marker.
        let done = Event::TurnFinished {
            stop_reason: alx_core::protocol::StopReason::EndTurn,
            model: None,
            usage: None,
        };
        assert_eq!(r.handle(&done).unwrap(), "  \n  Second para\n");
        // A later answer is shown normally again.
        r.handle(&Event::ModelRequest);
        let s = r
            .handle(&Event::AssistantText {
                text: "again".into(),
            })
            .unwrap();
        assert_eq!(s, "\n● again\n");
    }

    #[test]
    fn streamed_text_is_flushed_before_a_tool_call_and_discarded_after_a_failed_turn() {
        let r = Renderer::new(PLAIN, false);
        delta(&r, "let me look");
        let s = call(&r, "ls");
        assert!(s.starts_with("\n● let me look\n"), "{s:?}");
        finish(&r, "", false);
        // A turn that ended in an error leaves text behind; the next turn
        // must not show it.
        delta(&r, "stale");
        r.handle(&Event::TurnStarted);
        assert_eq!(r.handle(&Event::ModelRequest), None);
        let s = r
            .handle(&Event::AssistantText {
                text: "fresh".into(),
            })
            .unwrap();
        assert_eq!(s, "\n● fresh\n");
    }
}
