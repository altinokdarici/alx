//! Targeted edits: replace exact strings in a file instead of rewriting it.

use crate::tool::{Tool, ToolContext, ToolOutput};
use alx_protocol::ToolSpec;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::path::Path;

/// Replaces exact strings in a text file. Several edits in one call are
/// matched against the original text and applied all-or-nothing. Line
/// endings (CRLF) and a byte-order mark are kept.
#[derive(Debug, Default, Clone)]
pub struct EditFileTool;

#[derive(Debug, PartialEq, Eq)]
struct Edit {
    old: String,
    new: String,
    replace_all: bool,
}

fn parse_edit(v: &Value) -> Result<Edit, String> {
    let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_owned);
    let (Some(old), Some(new)) = (s("old_string"), s("new_string")) else {
        return Err("each edit needs string `old_string` and `new_string`".into());
    };
    Ok(Edit {
        old,
        new,
        replace_all: v.get("replace_all").and_then(Value::as_bool) == Some(true),
    })
}

fn parse_edits(input: &Value) -> Result<Vec<Edit>, String> {
    match input.get("edits") {
        Some(Value::Array(list)) if !list.is_empty() => list.iter().map(parse_edit).collect(),
        Some(_) => Err("`edits` must be a non-empty array".into()),
        None => Ok(vec![parse_edit(input)?]),
    }
}

/// Text compared loosely, to point the model at a near miss: no trailing
/// whitespace, typographic quotes, dashes and spaces made ASCII.
fn loose(text: &str) -> String {
    text.lines()
        .map(|l| {
            l.trim_end()
                .chars()
                .map(|c| match c {
                    '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
                    '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
                    '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
                    '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => {
                        ' '
                    }
                    c => c,
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn line_of(text: &str, byte: usize) -> usize {
    text[..byte].matches('\n').count() + 1
}

/// Applies `edits` to LF-only `text`. Every edit is matched against the
/// original text, not the result of earlier edits, and matches may not
/// overlap. Returns the new text, the number of replacements and the first
/// changed line, or a message for the model naming the failing edit.
fn apply(text: &str, edits: &[Edit]) -> Result<(String, usize, usize), String> {
    let mut ranges: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, edit index)
    for (i, e) in edits.iter().enumerate() {
        let which = if edits.len() > 1 {
            format!("edit {}: ", i + 1)
        } else {
            String::new()
        };
        if e.old.is_empty() {
            return Err(format!(
                "{which}`old_string` is empty; use write_file to create a file"
            ));
        }
        if e.old == e.new {
            return Err(format!("{which}`old_string` and `new_string` are the same"));
        }
        let found: Vec<usize> = text.match_indices(&e.old).map(|(at, _)| at).collect();
        match found.len() {
            0 => {
                let hint = loose(text)
                    .find(&loose(&e.old))
                    .filter(|_| !loose(&e.old).is_empty())
                    .map(|at| {
                        format!(
                            " A near match differs only in whitespace or quote/dash characters \
                             around line {}; copy the text exactly from the file.",
                            loose(text)[..at].matches('\n').count() + 1
                        )
                    })
                    .unwrap_or_else(|| {
                        " It must match the file exactly, including whitespace and indentation; \
                         read the file again and retry."
                            .into()
                    });
                return Err(format!("{which}`old_string` was not found.{hint}"));
            }
            1 => {}
            n if !e.replace_all => {
                return Err(format!(
                    "{which}`old_string` appears {n} times. Add surrounding lines to make it \
                     unique, or set `replace_all` to change every occurrence."
                ));
            }
            _ => {}
        }
        ranges.extend(found.into_iter().map(|at| (at, at + e.old.len(), i)));
    }
    ranges.sort_unstable();
    for w in ranges.windows(2) {
        if w[0].1 > w[1].0 {
            let (a, b) = (w[0].2.min(w[1].2) + 1, w[0].2.max(w[1].2) + 1);
            return Err(format!(
                "edits {a} and {b} overlap. Edits are all matched against the original \
                 file; merge them into one edit."
            ));
        }
    }
    let mut out = String::with_capacity(text.len());
    let mut pos = 0;
    for &(start, end, i) in &ranges {
        out.push_str(&text[pos..start]);
        out.push_str(&edits[i].new);
        pos = end;
    }
    out.push_str(&text[pos..]);
    Ok((out, ranges.len(), line_of(text, ranges[0].0)))
}

/// A file's text as the model sees it (no BOM, LF line breaks), and how to
/// put those back when writing.
struct Encoding {
    bom: bool,
    crlf: bool,
}

impl Encoding {
    fn decode(raw: &str) -> (Self, String) {
        let (bom, text) = match raw.strip_prefix('\u{FEFF}') {
            Some(t) => (true, t),
            None => (false, raw),
        };
        // Mostly-CRLF files are edited as LF and written back as CRLF.
        let crlf = text.matches("\r\n").count() * 2 > text.matches('\n').count();
        let text = if crlf {
            text.replace("\r\n", "\n")
        } else {
            text.to_owned()
        };
        (Self { bom, crlf }, text)
    }

    fn encode(&self, text: &str) -> String {
        let body = if self.crlf {
            text.replace('\n', "\r\n")
        } else {
            text.to_owned()
        };
        if self.bom {
            format!("\u{FEFF}{body}")
        } else {
            body
        }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn spec(&self) -> ToolSpec {
        let edit_props = json!({
            "old_string": { "type": "string", "description": "Exact text to replace, including whitespace and indentation. Must be unique in the file unless `replace_all` is set." },
            "new_string": { "type": "string", "description": "Text to put in its place" },
            "replace_all": { "type": "boolean", "description": "Replace every occurrence (default false)" }
        });
        ToolSpec {
            name: "edit_file".into(),
            description: "Edit a text file by replacing exact strings, instead of rewriting it \
                          whole. Give one `old_string`/`new_string` pair, or several in `edits`. \
                          Every edit is matched against the original file (not the result of \
                          earlier edits) and edits must not overlap; merge nearby changes into \
                          one. If any edit fails, the file is left unchanged. Read the file \
                          first so `old_string` matches exactly."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path of the file to edit" },
                    "old_string": edit_props["old_string"],
                    "new_string": edit_props["new_string"],
                    "replace_all": edit_props["replace_all"],
                    "edits": {
                        "type": "array",
                        "description": "Several non-overlapping edits, each matched against the original file (instead of old_string/new_string)",
                        "items": {
                            "type": "object",
                            "properties": edit_props,
                            "required": ["old_string", "new_string"]
                        }
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let Some(path) = input
            .get("path")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
        else {
            return ToolOutput::error("Invalid tool call: edit_file requires a string `path`");
        };
        let edits = match parse_edits(&input) {
            Ok(e) => e,
            Err(e) => return ToolOutput::error(format!("Invalid tool call: edit_file: {e}")),
        };
        let text = match ctx.executor.read_file(Path::new(path)).await {
            Ok(t) => t,
            Err(e) => return ToolOutput::error(format!("Could not read {path}: {e}")),
        };
        let (enc, text) = Encoding::decode(&text);
        let edits: Vec<Edit> = edits
            .into_iter()
            .map(|e| Edit {
                old: e.old.replace("\r\n", "\n"),
                new: e.new.replace("\r\n", "\n"),
                ..e
            })
            .collect();
        let (new_text, count, line) = match apply(&text, &edits) {
            Ok(r) => r,
            Err(e) => return ToolOutput::error(format!("{path} was not changed: {e}")),
        };
        match ctx
            .executor
            .write_file(Path::new(path), &enc.encode(&new_text))
            .await
        {
            Ok(()) => ToolOutput::ok(format!(
                "Edited {path}: {count} replacement{}, first at line {line}",
                if count == 1 { "" } else { "s" }
            )),
            Err(e) => ToolOutput::error(format!("Could not write {path}: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::LocalExecutor;
    use std::sync::Arc;

    fn edit(old: &str, new: &str, all: bool) -> Edit {
        Edit {
            old: old.into(),
            new: new.into(),
            replace_all: all,
        }
    }

    #[test]
    fn replaces_against_the_original() {
        assert_eq!(
            apply("a b a", &[edit("b", "c", false)]),
            Ok(("a c a".into(), 1, 1))
        );
        assert_eq!(
            apply("a\nb a", &[edit("a", "x", true)]),
            Ok(("x\nb x".into(), 2, 1))
        );
        // The second edit sees the original text, not "two".
        assert_eq!(
            apply(
                "one\nthree",
                &[edit("three", "3", false), edit("one", "two", false)]
            ),
            Ok(("two\n3".into(), 2, 1))
        );
        assert!(apply("one", &[edit("one", "two", false), edit("two", "x", false)]).is_err());
    }

    #[test]
    fn rejects_missing_ambiguous_overlapping_and_empty() {
        assert!(
            apply("abc", &[edit("z", "y", false)])
                .unwrap_err()
                .contains("not found")
        );
        assert!(
            apply("a a", &[edit("a", "b", false)])
                .unwrap_err()
                .contains("appears 2 times")
        );
        assert!(
            apply("a", &[edit("", "b", false)])
                .unwrap_err()
                .contains("empty")
        );
        assert!(
            apply("a b", &[edit("a", "x", false), edit("q", "y", false)])
                .unwrap_err()
                .starts_with("edit 2: ")
        );
        assert!(
            apply("abcd", &[edit("abc", "x", false), edit("cd", "y", false)])
                .unwrap_err()
                .contains("edits 1 and 2 overlap")
        );
    }

    #[test]
    fn points_at_near_misses() {
        let err = apply(
            "x\nlet s = \u{201C}hi\u{201D};   \n",
            &[edit("let s = \"hi\";", "y", false)],
        )
        .unwrap_err();
        assert!(
            err.contains("near match") && err.contains("line 2"),
            "{err}"
        );
    }

    #[test]
    fn keeps_crlf_and_bom() {
        let (enc, text) = Encoding::decode("\u{FEFF}a\r\nb\r\n");
        assert_eq!(text, "a\nb\n");
        assert_eq!(enc.encode("a\nc\n"), "\u{FEFF}a\r\nc\r\n");
        let (enc, text) = Encoding::decode("a\nb\r\nc\n");
        assert_eq!(
            (enc.bom, enc.crlf, text.as_str()),
            (false, false, "a\nb\r\nc\n")
        );
    }

    #[tokio::test]
    async fn edits_the_file_and_leaves_it_alone_on_failure() {
        let ctx = ToolContext {
            executor: Arc::new(LocalExecutor),
        };
        let dir = std::env::temp_dir().join(format!("alx-edit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.txt");
        let p = path.to_string_lossy().into_owned();
        std::fs::write(&path, "fn a() {}\nfn b() {}\n").unwrap();

        let out = EditFileTool
            .call(
                json!({"path": p, "old_string": "fn b() {}", "new_string": "fn c() {}"}),
                &ctx,
            )
            .await;
        assert_eq!(
            out,
            ToolOutput::ok(format!("Edited {p}: 1 replacement, first at line 2"))
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "fn a() {}\nfn c() {}\n"
        );

        let out = EditFileTool
            .call(
                json!({"path": p, "edits": [
                    {"old_string": "fn a", "new_string": "fn x"},
                    {"old_string": "nope", "new_string": "y"}
                ]}),
                &ctx,
            )
            .await;
        assert!(
            out.is_error && out.content.contains("edit 2"),
            "{}",
            out.content
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "fn a() {}\nfn c() {}\n"
        );

        assert!(EditFileTool.call(json!({"path": p}), &ctx).await.is_error);
        let _ = std::fs::remove_dir_all(dir);
    }
}
