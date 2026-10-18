//! Terminal rendering of core events: colors, indentation and truncation.
//!
//! `format_event` is pure (event in, text out) so it can be tested without a
//! terminal. `CliRenderer` only decides about color and which stream to print to.

use alx_core::EventSink;
use alx_core::protocol::Event;
use std::io::IsTerminal;

/// Tool output longer than this many lines is cut, with a note on what was hidden.
const MAX_OUTPUT_LINES: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

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

    fn paint(&self, code: &str, s: &str) -> String {
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

fn truncate_lines(text: &str, max: usize) -> (String, usize) {
    let total = text.lines().count();
    if total <= max {
        return (text.trim_end().to_owned(), 0);
    }
    let kept = text.lines().take(max).collect::<Vec<_>>().join("\n");
    (kept, total - max)
}

/// Format one event; `None` means nothing to show.
pub fn format_event(style: &Style, event: &Event) -> Option<(Stream, String)> {
    match event {
        Event::ModelRequest => Some((Stream::Stdout, style.dim("… thinking"))),
        Event::AssistantText { text } => Some((
            Stream::Stdout,
            format!(
                "\n{}\n{}\n",
                style.cyan_bold("● alx"),
                indent(text.trim(), "  ")
            ),
        )),
        Event::ToolCallStarted { name, input, .. } => {
            let detail = match input.get("input").and_then(|v| v.as_str()) {
                Some(script) if name == "bash" => script.to_owned(),
                _ => input.to_string(),
            };
            let mut lines = detail.lines();
            let first = lines.next().unwrap_or("");
            let mut out = format!(
                "{} {}",
                style.yellow(&format!("▸ {name}")),
                style.bold(first)
            );
            for l in lines {
                out.push_str(&format!("\n  {}", style.bold(l)));
            }
            Some((Stream::Stdout, out))
        }
        Event::ToolCallFinished {
            output, is_error, ..
        } => {
            let (shown, hidden) = truncate_lines(output, MAX_OUTPUT_LINES);
            if shown.is_empty() && hidden == 0 {
                return Some((Stream::Stdout, style.dim("  (no output)")));
            }
            let bar = style.dim("  │ ");
            let mut body = shown
                .lines()
                .map(|l| {
                    let l = if *is_error {
                        style.red(l)
                    } else {
                        l.to_owned()
                    };
                    format!("{bar}{l}")
                })
                .collect::<Vec<_>>()
                .join("\n");
            if hidden > 0 {
                body.push_str(&format!(
                    "\n{}",
                    style.dim(&format!("  … {hidden} more lines"))
                ));
            }
            let stream = if *is_error {
                Stream::Stderr
            } else {
                Stream::Stdout
            };
            Some((stream, body))
        }
        Event::TurnFinished {
            model: Some(model), ..
        } => Some((
            Stream::Stdout,
            format!("{}\n", style.dim(&format!("model: {model}"))),
        )),
        Event::TurnStarted | Event::TurnFinished { .. } => None,
    }
}

/// Prints core events to stdout/stderr.
pub struct CliRenderer {
    pub style: Style,
}

impl EventSink for CliRenderer {
    fn emit(&self, event: Event) {
        match format_event(&self.style, &event) {
            Some((Stream::Stdout, s)) => println!("{s}"),
            Some((Stream::Stderr, s)) => eprintln!("{s}"),
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PLAIN: Style = Style { color: false };

    fn text(e: Event) -> String {
        format_event(&PLAIN, &e).unwrap().1
    }

    #[test]
    fn bash_call_shows_script() {
        let s = text(Event::ToolCallStarted {
            id: "1".into(),
            name: "bash".into(),
            input: json!({"input": "ls -la"}),
        });
        assert_eq!(s, "▸ bash ls -la");
    }

    #[test]
    fn assistant_text_is_indented() {
        let s = text(Event::AssistantText {
            text: "a\nb".into(),
        });
        assert!(s.contains("● alx\n  a\n  b"));
    }

    #[test]
    fn long_output_is_truncated() {
        let output = (0..30)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let s = text(Event::ToolCallFinished {
            id: "1".into(),
            name: "bash".into(),
            output,
            is_error: false,
        });
        assert!(s.ends_with("… 10 more lines"));
        assert_eq!(s.lines().count(), 21);
    }

    #[test]
    fn errors_go_to_stderr_and_empty_output_is_noted() {
        let (stream, _) = format_event(
            &PLAIN,
            &Event::ToolCallFinished {
                id: "1".into(),
                name: "bash".into(),
                output: "boom".into(),
                is_error: true,
            },
        )
        .unwrap();
        assert_eq!(stream, Stream::Stderr);
        let s = text(Event::ToolCallFinished {
            id: "1".into(),
            name: "bash".into(),
            output: String::new(),
            is_error: false,
        });
        assert_eq!(s, "  (no output)");
    }

    #[test]
    fn color_adds_ansi_only_when_enabled() {
        assert!(Style { color: true }.red("x").contains("\x1b[31m"));
        assert_eq!(PLAIN.red("x"), "x");
    }
}
