//! Finding code: `grep` (search file contents), `glob` (find files by name)
//! and `list_dir`. Each runs a short bash script through the context's
//! executor, so it searches wherever commands run. `grep` and `glob` use
//! ripgrep when it is installed (respecting `.gitignore`) and fall back to
//! `grep` / `git ls-files` / `find`.

use crate::executor::{ExecOutput, ExecRequest};
use crate::tool::{Tool, ToolContext, ToolOutput};
use alx_protocol::ToolSpec;
use async_trait::async_trait;
use globset::GlobBuilder;
use serde_json::{Value, json};
use std::time::Duration;

/// Longest result in bytes; later lines are dropped with a note.
pub const MAX_OUTPUT_BYTES: usize = 30_000;
/// Longest single line kept in `grep` output.
const MAX_LINE_CHARS: usize = 300;
/// Most paths `glob` returns unless the model passes `limit`.
const MAX_ENTRIES: usize = 1000;
const TIMEOUT: Duration = Duration::from_secs(60);

/// Runs `script` with `bash -c`, passing `args` as `$1`, `$2`, ... so they
/// are never parsed as shell code.
async fn run_script(ctx: &ToolContext, script: &str, args: &[&str]) -> std::io::Result<ExecOutput> {
    let mut all = vec!["-c".to_owned(), script.to_owned(), "alx".to_owned()];
    all.extend(args.iter().map(|a| (*a).to_owned()));
    ctx.executor
        .run(ExecRequest {
            program: "bash".into(),
            args: all,
            timeout: Some(TIMEOUT),
            ..Default::default()
        })
        .await
}

fn str_arg<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// Joins `lines` up to [`MAX_OUTPUT_BYTES`] and `max_lines`, then says how
/// many were left out and what `narrow` should be tried.
fn bounded<'a>(lines: impl Iterator<Item = &'a str>, max_lines: usize, narrow: &str) -> String {
    let mut out = String::new();
    let mut shown = 0;
    let mut dropped = 0;
    for line in lines {
        if shown >= max_lines || out.len() + line.len() + 1 > MAX_OUTPUT_BYTES {
            dropped += 1;
            continue;
        }
        out.push_str(line);
        out.push('\n');
        shown += 1;
    }
    if dropped > 0 {
        out.push_str(&format!("[... {dropped} more not shown; {narrow} ...]\n"));
    }
    out
}

fn clip_line(line: &str) -> std::borrow::Cow<'_, str> {
    match line.char_indices().nth(MAX_LINE_CHARS) {
        Some((cut, _)) => format!("{} [... line cut ...]", &line[..cut]).into(),
        None => line.into(),
    }
}

fn failed(tool: &str, out: &ExecOutput) -> ToolOutput {
    if out.timed_out {
        return ToolOutput::error(format!("{tool} timed out; narrow the path or pattern"));
    }
    let err = out.stderr.trim();
    ToolOutput::error(format!(
        "{tool} failed{}",
        if err.is_empty() {
            String::new()
        } else {
            format!(": {err}")
        }
    ))
}

/// Searches file contents with a regular expression.
#[derive(Debug, Default, Clone)]
pub struct GrepTool;

impl GrepTool {
    /// Matches returned when the model doesn't pass `limit`.
    pub const DEFAULT_LIMIT: usize = 100;
}

/// Prints `@dir` first when `$2` is a directory: it then searches from
/// inside it, so `-g` globs are relative to it, and paths come out relative.
/// `--no-config` keeps a user's ripgrep config from changing results.
const GREP_SCRIPT: &str = r#"
pat=$1 dir=$2 glob=$3 icase=$4 literal=$5 ctx=$6
if [ -d "$dir" ]; then cd -- "$dir" || exit 2; echo @dir; set --; else set -- "$dir"; fi
target=("$@")
if command -v rg >/dev/null 2>&1; then
  set -- rg --no-config -n -H --no-heading --color=never --hidden -g '!.git'
  [ -n "$glob" ] && set -- "$@" -g "$glob"
else
  set -- grep -rnHI --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=target
  [ -z "$literal" ] && set -- "$@" -E
  [ ${#target[@]} -eq 0 ] && target=(.)
  if [[ $glob == */* && ${target[0]} == . ]]; then
    # --include only matches file names; expand a path glob in bash instead.
    shopt -s globstar 2>/dev/null; shopt -s nullglob dotglob
    target=($glob)
    [ ${#target[@]} -eq 0 ] && exit 1
  elif [ -n "$glob" ]; then
    set -- "$@" --include="$glob"
  fi
fi
[ -n "$icase" ] && set -- "$@" -i
[ -n "$literal" ] && set -- "$@" -F
[ "$ctx" -gt 0 ] && set -- "$@" -C "$ctx"
exec "$@" -e "$pat" -- "${target[@]}"
"#;

/// Whether a grep output line is a match (`path:12:text`) rather than
/// context (`path-12-text`) or a `--` separator.
fn is_match_line(line: &str) -> bool {
    let first = |sep: char| {
        line.char_indices().find_map(|(i, c)| {
            (c == sep).then(|| {
                let rest = &line[i + 1..];
                let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
                (digits > 0 && rest[digits..].starts_with(sep)).then_some(i)
            })?
        })
    };
    match (first(':'), first('-')) {
        (Some(m), Some(c)) => m < c,
        (m, _) => m.is_some(),
    }
}

/// Formats raw grep output: paths made relative to the working directory
/// again, long lines cut, at most `limit` matches, at most
/// [`MAX_OUTPUT_BYTES`].
fn format_grep(stdout: &str, dir: &str, limit: usize, context: bool) -> String {
    let (prefix, body) = match stdout.strip_prefix("@dir\n") {
        Some(rest) if dir.trim_end_matches('/') != "." => {
            (format!("{}/", dir.trim_end_matches('/')), rest)
        }
        Some(rest) => (String::new(), rest),
        None => (String::new(), stdout),
    };
    let mut lines = Vec::new();
    let mut matches = 0;
    let mut more = 0;
    // Without context every line is a match; with it, tell them apart
    // (a heuristic: a file name like `a-1-b` can look like context).
    let is_match = |line: &str| {
        if context {
            is_match_line(line)
        } else {
            line != "--"
        }
    };
    for line in body.lines() {
        if is_match(line) {
            matches += 1;
        }
        if matches > limit {
            more += usize::from(is_match(line));
            continue;
        }
        let line = line.strip_prefix("./").unwrap_or(line);
        lines.push(if line == "--" {
            line.to_owned()
        } else {
            format!("{prefix}{}", clip_line(line))
        });
    }
    let mut out = bounded(
        lines.iter().map(String::as_str),
        usize::MAX,
        "narrow the pattern, path or glob",
    );
    if more > 0 {
        out.push_str(&format!(
            "[... {more} more matches; limit {limit} reached. Pass a bigger `limit` or narrow the search ...]\n"
        ));
    }
    out
}

#[async_trait]
impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep".into(),
            description: "Search file contents for a regular expression (or a literal string), \
                          recursively under `path` (default: the working directory). Uses \
                          ripgrep when installed (skips .gitignored files), else grep -E. \
                          Returns `file:line:text` lines (context lines as `file-line-text`), \
                          long lines cut, at most `limit` matches (default 100) and 30000 bytes."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regular expression to search for" },
                    "path": { "type": "string", "description": "File or directory to search (default: .)" },
                    "glob": { "type": "string", "description": "Only search files matching this glob, relative to `path`, e.g. *.rs or src/**/*.rs. Without '/' it matches file names at any depth." },
                    "ignore_case": { "type": "boolean", "description": "Case-insensitive search" },
                    "literal": { "type": "boolean", "description": "Treat `pattern` as a plain string, not a regex" },
                    "context": { "type": "integer", "description": "Lines to show before and after each match (default 0)" },
                    "limit": { "type": "integer", "description": "Most matches to return (default 100)" }
                },
                "required": ["pattern"]
            }),
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let Some(pattern) = input
            .get("pattern")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
        else {
            return ToolOutput::error("Invalid tool call: grep requires a string `pattern`");
        };
        let path = str_arg(&input, "path").unwrap_or(".");
        let glob = strip_dot_slash(str_arg(&input, "glob").unwrap_or(""));
        let flag = |key: &str| {
            if input.get(key).and_then(Value::as_bool) == Some(true) {
                "1"
            } else {
                ""
            }
        };
        let context = input
            .get("context")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(20)
            .to_string();
        let limit = limit_arg(&input, Self::DEFAULT_LIMIT);
        let args = [
            pattern,
            path,
            &glob,
            flag("ignore_case"),
            flag("literal"),
            &context,
        ];
        let out = match run_script(ctx, GREP_SCRIPT, &args).await {
            Ok(o) => o,
            Err(e) => return ToolOutput::error(format!("grep could not run: {e}")),
        };
        match out.exit_code {
            Some(0) if !out.timed_out => {
                ToolOutput::ok(format_grep(&out.stdout, path, limit, context != "0"))
            }
            Some(1) if !out.timed_out => ToolOutput::ok("No matches."),
            _ => failed("grep", &out),
        }
    }
}

/// `limit` from the input, at least 1.
fn limit_arg(input: &Value, default: usize) -> usize {
    input
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(default, |l| l as usize)
        .max(1)
}

/// Drops a leading `./` (also after `!`): globs are matched against paths
/// without one.
fn strip_dot_slash(glob: &str) -> String {
    let (neg, rest) = match glob.strip_prefix('!') {
        Some(r) => ("!", r),
        None => ("", glob),
    };
    let mut rest = rest;
    while let Some(r) = rest.strip_prefix("./") {
        rest = r;
    }
    format!("{neg}{rest}")
}

/// Finds files by glob pattern.
#[derive(Debug, Default, Clone)]
pub struct GlobTool;

/// Lists files under `$1`, relative to it, one per line.
const LIST_FILES_SCRIPT: &str = r#"
cd -- "$1" || exit 2
if command -v rg >/dev/null 2>&1; then
  exec rg --files --hidden -g '!.git'
elif git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  exec git ls-files -co --exclude-standard
else
  find . -type f -not -path '*/.git/*' -not -path '*/node_modules/*' -not -path '*/target/*' | sed 's|^\./||'
fi
"#;

/// The paths in `files` matching `pattern`. A pattern without `/` matches
/// file names at any depth (`*.rs`); one with `/` matches the whole relative
/// path (`src/**/*.rs`).
fn match_glob<'a>(pattern: &str, files: &'a str) -> Result<Vec<&'a str>, globset::Error> {
    let by_name = !pattern.contains('/');
    let glob = GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()?
        .compile_matcher();
    let mut out: Vec<&str> = files
        .lines()
        .filter(|f| {
            let f = f.strip_prefix("./").unwrap_or(f);
            if by_name {
                glob.is_match(f.rsplit('/').next().unwrap_or(f))
            } else {
                glob.is_match(f)
            }
        })
        .collect();
    out.sort_unstable();
    Ok(out)
}

#[async_trait]
impl Tool for GlobTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "glob".into(),
            description: "Find files by glob pattern under `path` (default: the working \
                          directory), skipping .gitignored files when ripgrep or git is \
                          available. A pattern without `/` matches file names at any depth \
                          (`*.rs`); with `/` it matches the path relative to `path` \
                          (`src/**/*.rs`). Returns sorted paths, at most `limit` (default 1000)."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern, e.g. *.rs or src/**/mod.rs" },
                    "path": { "type": "string", "description": "Directory to search (default: .)" },
                    "limit": { "type": "integer", "description": "Most paths to return (default 1000)" }
                },
                "required": ["pattern"]
            }),
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let Some(pattern) = str_arg(&input, "pattern").map(strip_dot_slash) else {
            return ToolOutput::error("Invalid tool call: glob requires a string `pattern`");
        };
        let dir = str_arg(&input, "path").unwrap_or(".");
        let out = match run_script(ctx, LIST_FILES_SCRIPT, &[dir]).await {
            Ok(o) => o,
            Err(e) => return ToolOutput::error(format!("glob could not run: {e}")),
        };
        // rg exits 1 when there are no files at all.
        if !matches!(out.exit_code, Some(0 | 1)) || out.timed_out {
            return failed("glob", &out);
        }
        let found = match match_glob(&pattern, &out.stdout) {
            Ok(f) => f,
            Err(e) => return ToolOutput::error(format!("Invalid glob pattern: {e}")),
        };
        if found.is_empty() {
            return ToolOutput::ok("No files found.");
        }
        let prefix = dir.trim_end_matches('/');
        let paths: Vec<String> = found
            .iter()
            .map(|f| {
                if prefix == "." {
                    (*f).to_owned()
                } else {
                    format!("{prefix}/{f}")
                }
            })
            .collect();
        ToolOutput::ok(bounded(
            paths.iter().map(String::as_str),
            limit_arg(&input, MAX_ENTRIES),
            "pass a bigger `limit` or use a narrower pattern or path",
        ))
    }
}

/// Lists one directory.
#[derive(Debug, Default, Clone)]
pub struct ListDirTool;

impl ListDirTool {
    /// Entries returned when the model doesn't pass `limit`.
    pub const DEFAULT_LIMIT: usize = 500;
}

#[async_trait]
impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".into(),
            description: "List the entries of one directory (default: the working directory), \
                          including hidden ones, sorted, directories marked with a trailing `/`. \
                          At most `limit` entries (default 500)."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory to list (default: .)" },
                    "limit": { "type": "integer", "description": "Most entries to return (default 500)" }
                }
            }),
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let dir = str_arg(&input, "path").unwrap_or(".");
        let out = match run_script(ctx, r#"exec ls -1Ap -- "$1""#, &[dir]).await {
            Ok(o) => o,
            Err(e) => return ToolOutput::error(format!("list_dir could not run: {e}")),
        };
        if out.exit_code != Some(0) {
            return failed("list_dir", &out);
        }
        if out.stdout.trim().is_empty() {
            return ToolOutput::ok("(empty directory)");
        }
        let mut entries: Vec<&str> = out.stdout.lines().collect();
        entries.sort_by_cached_key(|e| (e.to_lowercase(), *e));
        ToolOutput::ok(bounded(
            entries.into_iter(),
            limit_arg(&input, ListDirTool::DEFAULT_LIMIT),
            "pass a bigger `limit` or use glob",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::LocalExecutor;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn ctx() -> ToolContext {
        ToolContext {
            executor: Arc::new(LocalExecutor),
        }
    }

    fn tree(name: &str) -> (PathBuf, String) {
        let dir = std::env::temp_dir().join(format!("alx-search-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src/inner")).unwrap();
        std::fs::write(dir.join("src/a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        std::fs::write(dir.join("src/inner/b.rs"), "// Alpha here\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "alpha in text\n").unwrap();
        let s = dir.to_string_lossy().into_owned();
        (dir, s)
    }

    #[test]
    fn glob_matches_names_or_paths() {
        let files = "src/a.rs\nsrc/inner/b.rs\nnotes.txt\n";
        assert_eq!(
            match_glob("*.rs", files).unwrap(),
            ["src/a.rs", "src/inner/b.rs"]
        );
        assert_eq!(match_glob("src/*.rs", files).unwrap(), ["src/a.rs"]);
        assert_eq!(match_glob("**/b.rs", files).unwrap(), ["src/inner/b.rs"]);
        assert!(match_glob("[", files).is_err());
    }

    #[test]
    fn grep_output_is_relative_limited_and_clipped() {
        assert!(!is_match_line("src/a.rs-3-x:1:"));
        assert!(!is_match_line("--"));
        let raw = "@dir\n./a.rs:1:m1\n./a.rs-2-ctx\n--\n./b.rs:9:m2\n./b.rs:10:m3\n";
        assert_eq!(
            format_grep(raw, "src/", 2, true),
            "src/a.rs:1:m1\nsrc/a.rs-2-ctx\n--\nsrc/b.rs:9:m2\n\
             [... 1 more matches; limit 2 reached. Pass a bigger `limit` or narrow the search ...]\n"
        );
        assert_eq!(format_grep("@dir\na.rs:1:x\n", ".", 9, false), "a.rs:1:x\n");
        assert_eq!(format_grep("f.rs:1:x\n", "f.rs", 9, false), "f.rs:1:x\n");
        assert_eq!(strip_dot_slash("!./x/*.rs"), "!x/*.rs");
    }

    #[test]
    fn bounded_says_what_was_left_out() {
        let lines = ["a", "b", "c"];
        assert_eq!(
            bounded(lines.into_iter(), 2, "narrow"),
            "a\nb\n[... 1 more not shown; narrow ...]\n"
        );
        assert_eq!(
            clip_line(&"x".repeat(400)).chars().count(),
            MAX_LINE_CHARS + 19
        );
    }

    #[tokio::test]
    async fn grep_finds_lines_and_filters() {
        let (dir, d) = tree("grep");
        let out = GrepTool
            .call(json!({"pattern": "alpha", "path": d}), &ctx())
            .await;
        assert!(!out.is_error, "{}", out.content);
        assert!(
            out.content.contains("a.rs:1:fn alpha() {}"),
            "{}",
            out.content
        );
        assert!(out.content.contains("notes.txt:1:alpha in text"));
        assert!(!out.content.contains("b.rs"));

        let out = GrepTool
            .call(
                json!({"pattern": "alpha", "path": d, "glob": "*.rs", "ignore_case": true}),
                &ctx(),
            )
            .await;
        assert!(
            out.content.contains("b.rs:1:// Alpha here"),
            "{}",
            out.content
        );
        assert!(!out.content.contains("notes.txt"));

        let out = GrepTool
            .call(json!({"pattern": "fn alpha()", "path": d, "literal": true, "glob": "src/*.rs", "context": 1}), &ctx())
            .await;
        assert_eq!(
            out,
            ToolOutput::ok(format!(
                "{d}/src/a.rs:1:fn alpha() {{}}\n{d}/src/a.rs-2-fn beta() {{}}\n"
            ))
        );
        let file = format!("{d}/notes.txt");
        let out = GrepTool
            .call(json!({"pattern": "text", "path": file}), &ctx())
            .await;
        assert_eq!(out, ToolOutput::ok(format!("{file}:1:alpha in text\n")));

        let out = GrepTool
            .call(json!({"pattern": "zzz", "path": d}), &ctx())
            .await;
        assert_eq!(out, ToolOutput::ok("No matches."));
        let out = GrepTool
            .call(json!({"pattern": "x", "path": format!("{d}/nope")}), &ctx())
            .await;
        assert!(out.is_error);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn glob_and_list_dir() {
        let (dir, d) = tree("glob");
        let out = GlobTool
            .call(json!({"pattern": "*.rs", "path": d}), &ctx())
            .await;
        assert_eq!(
            out,
            ToolOutput::ok(format!("{d}/src/a.rs\n{d}/src/inner/b.rs\n"))
        );
        let out = GlobTool
            .call(json!({"pattern": "*.md", "path": d}), &ctx())
            .await;
        assert_eq!(out, ToolOutput::ok("No files found."));

        let out = ListDirTool.call(json!({"path": d}), &ctx()).await;
        assert_eq!(out, ToolOutput::ok("notes.txt\nsrc/\n"));
        let out = ListDirTool
            .call(json!({"path": format!("{d}/nope")}), &ctx())
            .await;
        assert!(out.is_error);
        let _ = std::fs::remove_dir_all(dir);
    }
}
