//! Slash commands for the REPL's line editor. As soon as the line starts
//! with `/`, the matching commands show as a gray hint below it; Tab
//! completes them and → accepts a single match. The list is the REPL's own
//! commands plus whatever the agent advertises (`available_commands_update`),
//! which the REPL sends to the agent as a prompt.
//!
//! After `/model ` or `/provider `, the same goes for their values: any part
//! of a model's id or name finds it (`opu` → `claude-opus-…`), Tab cycles
//! through the matches, and [`resolve`] lets Enter take a unique partial one.

use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hint, Hinter};
use rustyline::validate::Validator;
use rustyline::{Context, Helper};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The slash commands the REPL handles itself, and what they do.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/clear", "start a new session (also /reset, /new)"),
    ("/model", "show or switch the model: /model [id]"),
    ("/provider", "show or switch the provider: /provider [name]"),
    ("/reset", "same as /clear"),
    ("/new", "same as /clear"),
];

/// Most values a hint lists before "… N more".
const MAX_SHOWN: usize = 12;

/// An `(id, display name)` pair: a command and its description, or a value.
pub type Item = (String, String);

/// What the hints draw from, kept current by the REPL: the agent's
/// advertised commands, and the values `/model` and `/provider` take.
#[derive(Default)]
pub struct Known {
    pub agent_commands: Vec<Item>,
    pub values: HashMap<&'static str, Vec<Item>>,
}

pub type SharedKnown = Arc<Mutex<Known>>;

/// The commands starting with `line`, if it is a command being typed: the
/// REPL's own first, then the agent's.
fn matching_commands(line: &str, agent: &[Item]) -> Vec<Item> {
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

/// The values matching `query`: an exact id alone, else every value whose
/// id or name contains it (ignoring case), ids starting with it first.
pub fn matching_values<'a>(query: &str, values: &'a [Item]) -> Vec<&'a Item> {
    if let Some(exact) = values.iter().find(|(id, _)| id == query) {
        return vec![exact];
    }
    let q = query.to_lowercase();
    let mut found: Vec<_> = values
        .iter()
        .filter(|(id, name)| id.to_lowercase().contains(&q) || name.to_lowercase().contains(&q))
        .collect();
    found.sort_by_key(|(id, _)| !id.to_lowercase().starts_with(&q));
    found
}

/// What `/cmd <query>` means: the one value it matches, else an error
/// naming the candidates. With nothing to match against, `query` as is.
pub fn resolve(query: &str, values: &[Item]) -> Result<String, String> {
    if values.is_empty() {
        return Ok(query.to_owned());
    }
    match matching_values(query, values).as_slice() {
        [] => Err(format!("nothing matches `{query}`")),
        [(id, _)] => Ok(id.clone()),
        many => Err(format!(
            "`{query}` matches {}: {}",
            many.len(),
            many.iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// `line` split as `/cmd query` when `/cmd` takes values.
fn value_query<'l>(line: &'l str, known: &Known) -> Option<(&'static str, &'l str)> {
    let (cmd, query) = line.split_once(' ')?;
    let (&cmd, _) = known.values.get_key_value(cmd)?;
    Some((cmd, query.trim_start()))
}

pub struct CommandHelper {
    /// Gray the hint (off with `NO_COLOR` / no TTY).
    pub color: bool,
    pub known: SharedKnown,
}

impl Completer for CommandHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let line = &line[..pos];
        let known = self.known.lock().unwrap();
        if let Some((cmd, query)) = value_query(line, &known) {
            let found = matching_values(query, &known.values[cmd])
                .into_iter()
                .map(|(id, _)| Pair {
                    display: id.clone(),
                    replacement: id.clone(),
                })
                .collect();
            return Ok((line.len() - query.len(), found));
        }
        let found = matching_commands(line, &known.agent_commands)
            .into_iter()
            .map(|(c, _)| Pair {
                replacement: format!("{c} "),
                display: c,
            })
            .collect();
        Ok((0, found))
    }
}

/// What shows after the cursor, and what → inserts (only a match's rest).
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

/// The hint for `typed` given its matches: the rest of a single match
/// inline (→ takes it if it extends `typed`), else one match per line.
fn hint_for(typed: &str, found: &[Item]) -> Option<CommandHint> {
    match found {
        [] => None,
        [(id, about)] => {
            let rest = id.strip_prefix(typed);
            let shown = rest.map_or_else(|| format!("  → {id}"), str::to_owned);
            let about = if about == id { "" } else { about };
            Some(CommandHint {
                display: format!("{shown}  {about}").trim_end().to_owned(),
                completion: rest.filter(|r| !r.is_empty()).map(str::to_owned),
            })
        }
        many => {
            let width = many.iter().map(|(id, _)| id.len()).max().unwrap_or(0);
            let mut display: String = many
                .iter()
                .take(MAX_SHOWN)
                .map(|(id, about)| {
                    let about = if about == id { "" } else { about };
                    format!("\n  {id:<width$}  {about}").trim_end().to_owned()
                })
                .collect();
            if many.len() > MAX_SHOWN {
                display.push_str(&format!(
                    "\n  … {} more (keep typing to narrow; Enter on the bare command lists all)",
                    many.len() - MAX_SHOWN
                ));
            }
            Some(CommandHint {
                display,
                completion: None,
            })
        }
    }
}

impl Hinter for CommandHelper {
    type Hint = CommandHint;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<CommandHint> {
        if pos < line.len() {
            return None;
        }
        let known = self.known.lock().unwrap();
        if let Some((cmd, query)) = value_query(line, &known) {
            let found: Vec<Item> = matching_values(query, &known.values[cmd])
                .into_iter()
                .cloned()
                .collect();
            return hint_for(query, &found);
        }
        hint_for(line, &matching_commands(line, &known.agent_commands))
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

    fn items(ids: &[(&str, &str)]) -> Vec<Item> {
        ids.iter()
            .map(|&(a, b)| (a.to_owned(), b.to_owned()))
            .collect()
    }

    #[test]
    fn matches_own_and_agent_commands_being_typed() {
        let agent = items(&[("/compact", "compact"), ("/model", "dup")]);
        assert_eq!(matching_commands("/", &agent).len(), COMMANDS.len() + 1);
        assert_eq!(matching_commands("/co", &agent), vec![agent[0].clone()]);
        assert_eq!(matching_commands("/mo", &agent)[0].1, COMMANDS[1].1);
        assert!(matching_commands("/model x", &agent).is_empty());
        assert!(matching_commands("hello /", &agent).is_empty());
    }

    #[test]
    fn values_match_any_part_of_id_or_name() {
        let models = items(&[
            ("claude-opus-5.5", "Claude Opus 5.5"),
            ("claude-sonnet-5", "Claude Sonnet 5"),
            ("gpt-6-sol", "GPT-6 Sol"),
            ("gpt-6", "GPT-6"),
        ]);
        assert_eq!(resolve("opu", &models).unwrap(), "claude-opus-5.5");
        assert_eq!(resolve("SONNET", &models).unwrap(), "claude-sonnet-5");
        // An exact id wins over the longer ids containing it.
        assert_eq!(resolve("gpt-6", &models).unwrap(), "gpt-6");
        assert!(
            resolve("claude", &models)
                .unwrap_err()
                .contains("matches 2")
        );
        assert!(resolve("zzz", &models).is_err());
        // Nothing listed: taken as typed.
        assert_eq!(resolve("x", &[]).unwrap(), "x");
    }
}
