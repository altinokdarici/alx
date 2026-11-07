//! Slash commands for the REPL's line editor: as soon as the line starts
//! with `/`, the matching commands show as a gray hint below it; Tab
//! completes them and → accepts a single match.

use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hint, Hinter};
use rustyline::validate::Validator;
use rustyline::{Context, Helper};
use std::borrow::Cow;

/// The REPL's slash commands and what they do.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/clear", "start a new session (also /reset, /new)"),
    ("/compact", "compact the conversation"),
    ("/model", "show or switch the model: /model [id]"),
    ("/provider", "show or switch the provider: /provider [name]"),
    ("/reset", "same as /clear"),
    ("/new", "same as /clear"),
];

/// The commands starting with `line`, if it is a command being typed.
fn matching(line: &str) -> Vec<(&'static str, &'static str)> {
    if !line.starts_with('/') || line.contains(char::is_whitespace) {
        return Vec::new();
    }
    COMMANDS
        .iter()
        .copied()
        .filter(|(c, _)| c.starts_with(line))
        .collect()
}

pub struct CommandHelper {
    /// Gray the hint (off with `NO_COLOR` / no TTY).
    pub color: bool,
}

impl Completer for CommandHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let found = matching(&line[..pos])
            .into_iter()
            .map(|(c, _)| Pair {
                display: c.to_owned(),
                replacement: format!("{c} "),
            })
            .collect();
        Ok((0, found))
    }
}

/// What shows after the cursor, and what → inserts (only a command's rest).
pub struct CommandHint {
    display: String,
    completion: Option<String>,
}

impl Hint for CommandHint {
    fn display(&self) -> &str {
        &self.display
    }
    fn completion(&self) -> Option<&str> {
        self.completion.as_deref()
    }
}

impl Hinter for CommandHelper {
    type Hint = CommandHint;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<CommandHint> {
        if pos < line.len() {
            return None;
        }
        match matching(line).as_slice() {
            [] => None,
            // One match: the rest of it inline, which → accepts.
            [(c, d)] => Some(CommandHint {
                display: format!("{}  {d}", &c[line.len()..]),
                completion: Some(c[line.len()..].to_owned()),
            }),
            many => Some(CommandHint {
                display: many
                    .iter()
                    .map(|(c, d)| format!("\n  {c:<10} {d}"))
                    .collect(),
                completion: None,
            }),
        }
    }
}

impl Highlighter for CommandHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        if self.color {
            Cow::Owned(format!("\x1b[2m{hint}\x1b[0m"))
        } else {
            Cow::Borrowed(hint)
        }
    }
}

impl Validator for CommandHelper {}
impl Helper for CommandHelper {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_only_a_command_being_typed() {
        assert_eq!(matching("/").len(), COMMANDS.len());
        assert_eq!(matching("/mo"), vec![COMMANDS[2]]);
        assert!(matching("/model x").is_empty());
        assert!(matching("hello /").is_empty());
    }
}
