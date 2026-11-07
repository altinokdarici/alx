//! Slash commands for the REPL's line editor: as soon as the line starts
//! with `/`, the matching commands show as a gray hint below it; Tab
//! completes them and → accepts a single match. The list is the REPL's own
//! commands plus whatever the agent advertises (`available_commands_update`),
//! which the REPL sends to the agent as a prompt.

use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hint, Hinter};
use rustyline::validate::Validator;
use rustyline::{Context, Helper};
use std::borrow::Cow;
use std::sync::{Arc, Mutex};

/// The slash commands the REPL handles itself, and what they do.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/clear", "start a new session (also /reset, /new)"),
    ("/model", "show or switch the model: /model [id]"),
    ("/provider", "show or switch the provider: /provider [name]"),
    ("/reset", "same as /clear"),
    ("/new", "same as /clear"),
];

/// The agent's advertised commands as `(/name, description)`.
pub type AgentCommands = Arc<Mutex<Vec<(String, String)>>>;

/// The commands starting with `line`, if it is a command being typed: the
/// REPL's own first, then the agent's.
fn matching(line: &str, agent: &[(String, String)]) -> Vec<(String, String)> {
    if !line.starts_with('/') || line.contains(char::is_whitespace) {
        return Vec::new();
    }
    let own = COMMANDS.iter().map(|&(c, d)| (c.to_owned(), d.to_owned()));
    let agent = agent
        .iter()
        .filter(|(c, _)| !COMMANDS.iter().any(|(o, _)| o == c))
        .cloned();
    own.chain(agent)
        .filter(|(c, _)| c.starts_with(line))
        .collect()
}

pub struct CommandHelper {
    /// Gray the hint (off with `NO_COLOR` / no TTY).
    pub color: bool,
    pub agent: AgentCommands,
}

impl CommandHelper {
    fn matching(&self, line: &str) -> Vec<(String, String)> {
        matching(line, &self.agent.lock().unwrap())
    }
}

impl Completer for CommandHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let found = self
            .matching(&line[..pos])
            .into_iter()
            .map(|(c, _)| Pair {
                replacement: format!("{c} "),
                display: c,
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
        match self.matching(line).as_slice() {
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
    fn matches_own_and_agent_commands_being_typed() {
        let agent = vec![
            ("/compact".to_owned(), "compact".to_owned()),
            ("/model".to_owned(), "dup".to_owned()),
        ];
        assert_eq!(matching("/", &agent).len(), COMMANDS.len() + 1);
        assert_eq!(matching("/co", &agent), vec![agent[0].clone()]);
        assert_eq!(matching("/mo", &agent)[0].1, COMMANDS[1].1);
        assert!(matching("/model x", &agent).is_empty());
        assert!(matching("hello /", &agent).is_empty());
    }
}
