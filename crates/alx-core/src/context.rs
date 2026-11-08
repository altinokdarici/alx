//! What the system prompt says about where the agent runs: the context
//! files (`AGENTS.md` / `CLAUDE.md`) and a short environment block.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// The names a context file can have, in order of preference: a directory
/// contributes its first one that exists.
const CONTEXT_FILES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];

/// A context file found for a working directory.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextFile {
    pub path: PathBuf,
    pub contents: String,
}

/// The context files that apply to `cwd`: the global one in `global_dir`
/// (`~/.config/alx`), then one per directory from the filesystem root down to
/// `cwd`, so the most specific comes last. Empty files are skipped.
pub fn context_files(cwd: &Path, global_dir: Option<&Path>) -> Vec<ContextFile> {
    let mut dirs: Vec<&Path> = cwd.ancestors().collect();
    dirs.reverse();
    global_dir
        .into_iter()
        .chain(dirs)
        .filter_map(|dir| {
            CONTEXT_FILES.iter().find_map(|name| {
                let path = dir.join(name);
                let contents = std::fs::read_to_string(&path).ok()?;
                (!contents.trim().is_empty()).then_some(ContextFile { path, contents })
            })
        })
        .collect()
}

/// `system_prompt` followed by the environment block and the context files.
/// The date has no time of day, so the prompt (and the provider's prompt
/// cache) stays the same all day.
pub fn build_system_prompt(
    system_prompt: &str,
    cwd: &Path,
    today: &str,
    files: &[ContextFile],
) -> String {
    let mut out = format!(
        "{system_prompt}\n\n<environment>\nWorking directory: {}\nPlatform: {}\nToday's date: {today}\n</environment>",
        cwd.display(),
        std::env::consts::OS,
    );
    if !files.is_empty() {
        out.push_str(
            "\n\n<project_context>\nInstructions from context files. Later (more specific) files take precedence over earlier ones.\n",
        );
        for f in files {
            out.push_str(&format!(
                "\n<project_instructions path=\"{}\">\n{}\n</project_instructions>\n",
                f.path.display(),
                f.contents.trim()
            ));
        }
        out.push_str("</project_context>");
    }
    out
}

/// Today's UTC date as `YYYY-MM-DD`.
pub fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Days since 1970-01-01 to a (year, month, day), per Howard Hinnant's
/// `civil_from_days`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_from_days() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(20_765), (2026, 11, 8));
    }

    #[test]
    fn finds_context_files_from_the_root_down() {
        let tmp = std::env::temp_dir().join(format!("alx-ctx-{}", std::process::id()));
        let global = tmp.join("global");
        let sub = tmp.join("repo").join("sub");
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(global.join("AGENTS.md"), "global").unwrap();
        std::fs::write(tmp.join("repo").join("AGENTS.md"), "repo").unwrap();
        std::fs::write(tmp.join("repo").join("CLAUDE.md"), "ignored").unwrap();
        std::fs::write(sub.join("CLAUDE.md"), "sub").unwrap();

        let found: Vec<String> = context_files(&sub, Some(&global))
            .into_iter()
            .map(|f| f.contents)
            .collect();
        std::fs::remove_dir_all(&tmp).unwrap();
        assert_eq!(found, ["global", "repo", "sub"]);
    }

    #[test]
    fn prompt_has_environment_and_files() {
        let files = [ContextFile {
            path: "/r/AGENTS.md".into(),
            contents: "\nbe terse\n".into(),
        }];
        let p = build_system_prompt("base", Path::new("/r"), "2026-11-08", &files);
        assert!(p.starts_with("base\n\n<environment>\nWorking directory: /r\n"));
        assert!(p.contains("Today's date: 2026-11-08\n"));
        assert!(p.contains(
            "<project_instructions path=\"/r/AGENTS.md\">\nbe terse\n</project_instructions>"
        ));
        assert!(build_system_prompt("base", Path::new("/r"), "d", &[]).ends_with("</environment>"));
    }
}
