//! Reads lines typed while a turn runs, so they can steer it.
//!
//! Reads fd 0 directly (not `std::io::stdin()`), so no line is left in a
//! buffer that rustyline won't see.

use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc::UnboundedSender;

#[cfg(unix)]
pub fn is_terminal() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal()
}

#[cfg(not(unix))]
pub fn is_terminal() -> bool {
    false
}

/// Sends each line read from stdin until `stop` is set.
#[cfg(unix)]
pub fn read_lines(stop: &AtomicBool, tx: &UnboundedSender<String>) {
    let mut buf = [0u8; 4096];
    let mut pending = Vec::new();
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
            return;
        }
        pending.extend_from_slice(&buf[..n as usize]);
        while let Some(i) = pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = pending.drain(..=i).collect();
            let _ = tx.send(String::from_utf8_lossy(&line).into_owned());
        }
    }
}

#[cfg(not(unix))]
pub fn read_lines(_stop: &AtomicBool, _tx: &UnboundedSender<String>) {}
