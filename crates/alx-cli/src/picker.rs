//! An inline, arrow-key list picker (like Claude Code's `/model`): ↑/↓ move,
//! typing filters, Enter picks, Esc or Ctrl-C cancels. Drawn below the prompt
//! and erased when done.

use std::io::{self, Write};

use crossterm::cursor::{Hide, MoveToColumn, MoveUp, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{self, Clear, ClearType};
use crossterm::{execute, queue};

/// Rows of items shown at once; the list scrolls past this.
const WINDOW: usize = 10;

/// One entry: the value returned, and an optional label shown beside it.
pub type Item = (String, String);

/// The picker's state, kept free of terminal I/O so it can be tested.
#[derive(Debug)]
pub struct State {
    items: Vec<Item>,
    filter: String,
    /// Index into `visible()`.
    selected: usize,
    /// First visible row of the scrolled window.
    offset: usize,
}

pub enum Outcome {
    Continue,
    Picked(String),
    Cancelled,
}

impl State {
    pub fn new(items: Vec<Item>, current: Option<&str>) -> Self {
        let selected = current
            .and_then(|c| items.iter().position(|(v, _)| v == c))
            .unwrap_or(0);
        let mut s = Self {
            items,
            filter: String::new(),
            selected,
            offset: 0,
        };
        s.scroll();
        s
    }

    /// Items matching the filter (any part of the value or label, any case).
    pub fn visible(&self) -> Vec<&Item> {
        let f = self.filter.to_lowercase();
        self.items
            .iter()
            .filter(|(v, n)| v.to_lowercase().contains(&f) || n.to_lowercase().contains(&f))
            .collect()
    }

    fn scroll(&mut self) {
        if self.selected < self.offset {
            self.offset = self.selected;
        } else if self.selected >= self.offset + WINDOW {
            self.offset = self.selected + 1 - WINDOW;
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> Outcome {
        let len = self.visible().len();
        match key.code {
            KeyCode::Esc => return Outcome::Cancelled,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Outcome::Cancelled;
            }
            KeyCode::Enter => {
                return match self.visible().get(self.selected) {
                    Some((v, _)) => Outcome::Picked(v.clone()),
                    None => Outcome::Continue,
                };
            }
            KeyCode::Up if len > 0 => self.selected = (self.selected + len - 1) % len,
            KeyCode::Down if len > 0 => self.selected = (self.selected + 1) % len,
            KeyCode::PageUp => self.selected = self.selected.saturating_sub(WINDOW),
            KeyCode::PageDown if len > 0 => self.selected = (self.selected + WINDOW).min(len - 1),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = len.saturating_sub(1),
            KeyCode::Backspace => {
                self.filter.pop();
                self.selected = 0;
                self.offset = 0;
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.filter.push(c);
                self.selected = 0;
                self.offset = 0;
            }
            _ => {}
        }
        self.scroll();
        Outcome::Continue
    }

    /// The lines to draw: a header, the window of items, and a footer.
    pub fn lines(&self, title: &str, current: Option<&str>, color: bool) -> Vec<String> {
        let (dim, rev, off) = if color {
            ("\x1b[2m", "\x1b[7m", "\x1b[0m")
        } else {
            ("", "", "")
        };
        let mut out = vec![format!(
            "{title} {dim}(↑/↓ move, type to filter, Enter pick, Esc cancel){off}"
        )];
        if !self.filter.is_empty() {
            out.push(format!("{dim}filter:{off} {}", self.filter));
        }
        let visible = self.visible();
        if visible.is_empty() {
            out.push(format!("{dim}  no match{off}"));
        }
        let end = (self.offset + WINDOW).min(visible.len());
        if self.offset > 0 {
            out.push(format!("{dim}  ↑ {} more{off}", self.offset));
        }
        for (i, (v, n)) in visible.iter().enumerate().take(end).skip(self.offset) {
            let cursor = if i == self.selected { '❯' } else { ' ' };
            let mark = if Some(v.as_str()) == current {
                " ✓"
            } else {
                ""
            };
            let label = if n != v {
                format!("  {dim}{n}{off}")
            } else {
                String::new()
            };
            let item = format!("{v}{mark}");
            let item = if i == self.selected && color {
                format!("{rev}{item}{off}")
            } else {
                item
            };
            out.push(format!("{cursor} {item}{label}"));
        }
        if end < visible.len() {
            out.push(format!("{dim}  ↓ {} more{off}", visible.len() - end));
        }
        out
    }
}

/// Runs the picker on the terminal. `Ok(None)` means cancelled.
pub fn pick(
    title: &str,
    items: Vec<Item>,
    current: Option<&str>,
    color: bool,
) -> io::Result<Option<String>> {
    let mut state = State::new(items, current);
    let mut out = io::stdout();
    terminal::enable_raw_mode()?;
    execute!(out, Hide)?;
    let result = (|| {
        let mut drawn = 0u16;
        loop {
            let lines = state.lines(title, current, color);
            if drawn > 0 {
                queue!(out, MoveUp(drawn))?;
            }
            queue!(out, MoveToColumn(0), Clear(ClearType::FromCursorDown))?;
            for line in &lines {
                // Raw mode: \n does not return the carriage.
                write!(out, "{line}\r\n")?;
            }
            out.flush()?;
            drawn = lines.len() as u16;
            if let Event::Key(key) = event::read()?
                && key.kind != KeyEventKind::Release
            {
                match state.key(key) {
                    Outcome::Continue => {}
                    Outcome::Picked(v) => break Ok((Some(v), drawn)),
                    Outcome::Cancelled => break Ok((None, drawn)),
                }
            }
        }
    })();
    let mut cleanup = |drawn: u16| -> io::Result<()> {
        if drawn > 0 {
            queue!(out, MoveUp(drawn))?;
        }
        queue!(out, MoveToColumn(0), Clear(ClearType::FromCursorDown), Show)?;
        out.flush()
    };
    let picked = match result {
        Ok((picked, drawn)) => {
            cleanup(drawn)?;
            picked
        }
        Err(e) => {
            let _ = execute!(io::stdout(), Show);
            let _ = terminal::disable_raw_mode();
            return Err(e);
        }
    };
    terminal::disable_raw_mode()?;
    Ok(picked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(n: usize) -> Vec<Item> {
        (0..n)
            .map(|i| (format!("m{i}"), format!("Model {i}")))
            .collect()
    }

    fn press(s: &mut State, code: KeyCode) -> Outcome {
        s.key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn picked(o: Outcome) -> Option<String> {
        match o {
            Outcome::Picked(v) => Some(v),
            _ => None,
        }
    }

    #[test]
    fn starts_on_current_and_moves_with_wrap() {
        let mut s = State::new(items(3), Some("m1"));
        press(&mut s, KeyCode::Down);
        press(&mut s, KeyCode::Down);
        assert_eq!(picked(press(&mut s, KeyCode::Enter)).as_deref(), Some("m0"));
        press(&mut s, KeyCode::Up);
        assert_eq!(picked(press(&mut s, KeyCode::Enter)).as_deref(), Some("m2"));
    }

    #[test]
    fn typing_filters_and_resets_selection() {
        let mut s = State::new(
            vec![
                ("gpt-5".into(), "GPT 5".into()),
                ("claude-opus-5.5".into(), "Claude Opus".into()),
            ],
            Some("gpt-5"),
        );
        for c in "OPU".chars() {
            press(&mut s, KeyCode::Char(c));
        }
        assert_eq!(s.visible().len(), 1);
        assert_eq!(
            picked(press(&mut s, KeyCode::Enter)).as_deref(),
            Some("claude-opus-5.5")
        );
        press(&mut s, KeyCode::Char('z'));
        assert!(picked(press(&mut s, KeyCode::Enter)).is_none());
    }

    #[test]
    fn window_scrolls_to_reveal_every_item() {
        let mut s = State::new(items(25), None);
        press(&mut s, KeyCode::End);
        let lines = s.lines("t", None, false);
        assert!(lines.iter().any(|l| l.starts_with("❯ m24")));
        assert!(lines.iter().any(|l| l.contains("↑ 15 more")));
        press(&mut s, KeyCode::Down); // wraps to the top
        let lines = s.lines("t", None, false);
        assert!(lines.iter().any(|l| l.starts_with("❯ m0")));
        assert!(lines.iter().any(|l| l.contains("↓ 15 more")));
    }

    #[test]
    fn esc_cancels() {
        let mut s = State::new(items(2), None);
        assert!(matches!(press(&mut s, KeyCode::Esc), Outcome::Cancelled));
    }
}
