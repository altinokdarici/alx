//! Lines typed while a turn runs steer it.
//!
//! During a turn the terminal's last row becomes an input line: the rows above
//! it are a scroll region the turn's output prints into, echo is off, and the
//! typed text is drawn on the last row, so it never mixes with the output.
//! Input is read from fd 0 directly (not `std::io::stdin()`), so nothing is
//! left in a buffer rustyline won't see.

use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc::UnboundedSender;

#[cfg(unix)]
pub fn is_terminal() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

#[cfg(not(unix))]
pub fn is_terminal() -> bool {
    false
}

/// What one input byte does to the line being typed.
#[derive(Debug, PartialEq, Eq)]
enum Key {
    Submit,
    Backspace,
    ClearLine,
    Byte(u8),
    Ignore,
}

/// Turns raw input bytes into line edits, dropping escape sequences
/// (arrows and the like).
#[derive(Default)]
struct Keys {
    in_escape: bool,
}

impl Keys {
    fn key(&mut self, b: u8) -> Key {
        if self.in_escape {
            // A CSI/SS3 sequence ends at its final byte (a letter or `~`).
            if b.is_ascii_alphabetic() || b == b'~' {
                self.in_escape = false;
            }
            return Key::Ignore;
        }
        match b {
            0x1b => {
                self.in_escape = true;
                Key::Ignore
            }
            b'\r' | b'\n' => Key::Submit,
            0x7f | 0x08 => Key::Backspace,
            0x15 => Key::ClearLine,
            b if b < 0x20 => Key::Ignore,
            b => Key::Byte(b),
        }
    }
}

/// Removes the last char (not byte) of a UTF-8 buffer.
fn pop_char(buf: &mut Vec<u8>) {
    while let Some(b) = buf.pop() {
        if b & 0xC0 != 0x80 {
            break;
        }
    }
}

/// Reads lines until `stop` is set, sending each one. Returns what was typed
/// but not yet sent, for the next prompt to start with.
#[cfg(unix)]
pub fn read_lines(stop: &AtomicBool, tx: &UnboundedSender<String>) -> String {
    let Some(_screen) = Screen::enter() else {
        return String::new();
    };
    let mut keys = Keys::default();
    let mut line = Vec::new();
    let mut buf = [0u8; 1024];
    Screen::draw(&line);
    while !stop.load(Ordering::Relaxed) {
        let mut fd = libc::pollfd {
            fd: 0,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd.
        let ready = unsafe { libc::poll(&mut fd, 1, 100) };
        if ready <= 0 || fd.revents & libc::POLLIN == 0 {
            continue;
        }
        // SAFETY: buf is valid for buf.len() bytes.
        let n = unsafe { libc::read(0, buf.as_mut_ptr().cast(), buf.len()) };
        if n <= 0 {
            break;
        }
        for &b in &buf[..n as usize] {
            match keys.key(b) {
                Key::Submit => {
                    let text = String::from_utf8_lossy(&line).into_owned();
                    line.clear();
                    let _ = tx.send(text);
                }
                Key::Backspace => pop_char(&mut line),
                Key::ClearLine => line.clear(),
                Key::Byte(b) => line.push(b),
                Key::Ignore => {}
            }
        }
        Screen::draw(&line);
    }
    String::from_utf8_lossy(&line).into_owned()
}

#[cfg(not(unix))]
pub fn read_lines(_stop: &AtomicBool, _tx: &UnboundedSender<String>) -> String {
    String::new()
}

/// The terminal set up for steering; dropping it puts the terminal back.
#[cfg(unix)]
struct Screen {
    saved: libc::termios,
    rows: u16,
}

#[cfg(unix)]
impl Screen {
    fn rows() -> Option<u16> {
        // SAFETY: TIOCGWINSZ fills a winsize.
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        let ok = unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) } == 0;
        (ok && ws.ws_row >= 3).then_some(ws.ws_row)
    }

    fn enter() -> Option<Self> {
        let rows = Self::rows()?;
        // SAFETY: tcgetattr/tcsetattr on fd 0, which is a terminal.
        let mut saved: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(0, &mut saved) } != 0 {
            return None;
        }
        let mut raw = saved;
        // No line buffering or echo; signals (Ctrl-C) still work.
        raw.c_lflag &= !(libc::ICANON | libc::ECHO);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        if unsafe { libc::tcsetattr(0, libc::TCSANOW, &raw) } != 0 {
            return None;
        }
        // Make room for the input row (a newline then back up, which scrolls
        // if the cursor is on the last row), then limit scrolling to the rows
        // above it. Setting the region homes the cursor, hence save/restore.
        write(&format!("\n\x1b[A\x1b7\x1b[1;{}r\x1b8", rows - 1));
        Some(Self { saved, rows })
    }

    /// Draws the input row: a dim arrow, then the typed text (its tail, if
    /// it's wider than the terminal).
    fn draw(line: &[u8]) {
        let Some(rows) = Self::rows() else { return };
        let text = String::from_utf8_lossy(line);
        let cols = term_cols().saturating_sub(3).max(1);
        let chars: Vec<char> = text.chars().collect();
        let shown: String = chars[chars.len().saturating_sub(cols)..].iter().collect();
        write(&format!(
            "\x1b7\x1b[{rows};1H\x1b[2K\x1b[2m↪\x1b[0m {shown}\x1b8"
        ));
    }
}

#[cfg(unix)]
impl Drop for Screen {
    fn drop(&mut self) {
        let rows = Self::rows().unwrap_or(self.rows);
        // Full-screen scrolling again, and the input row cleared.
        write(&format!("\x1b7\x1b[r\x1b8\x1b7\x1b[{rows};1H\x1b[2K\x1b8"));
        // SAFETY: restores the attributes read in `enter`.
        unsafe { libc::tcsetattr(0, libc::TCSANOW, &self.saved) };
    }
}

#[cfg(unix)]
fn term_cols() -> usize {
    // SAFETY: TIOCGWINSZ fills a winsize.
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    if unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col > 0 {
        ws.ws_col as usize
    } else {
        80
    }
}

/// One locked write, so it can't land in the middle of another print.
#[cfg(unix)]
fn write(s: &str) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(s.as_bytes());
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_bytes(input: &[u8]) -> (Vec<String>, Vec<u8>) {
        let mut keys = Keys::default();
        let (mut sent, mut line) = (Vec::new(), Vec::new());
        for &b in input {
            match keys.key(b) {
                Key::Submit => sent.push(String::from_utf8(std::mem::take(&mut line)).unwrap()),
                Key::Backspace => pop_char(&mut line),
                Key::ClearLine => line.clear(),
                Key::Byte(b) => line.push(b),
                Key::Ignore => {}
            }
        }
        (sent, line)
    }

    #[test]
    fn edits_lines_and_skips_escape_sequences() {
        let (sent, rest) = type_bytes("héx\x7f\x7fi\x1b[Dyo\rnext".as_bytes());
        assert_eq!(sent, ["hiyo"]);
        assert_eq!(rest, b"next");
        let (sent, rest) = type_bytes(b"abc\x15d\n");
        assert_eq!(sent, ["d"]);
        assert!(rest.is_empty());
    }
}
