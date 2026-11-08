//! Built-in tools.

pub mod bash;
pub mod edit;
pub mod files;
pub mod search;
pub mod todo;
pub mod web;

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Returns the lock for a file path, so concurrent tool calls that modify the
/// same file run one after another. Paths are normalized lexically (the
/// executor may be remote, so the file system can't be asked).
pub(crate) fn file_lock(path: &str) -> Arc<tokio::sync::Mutex<()>> {
    static LOCKS: Mutex<Option<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>> = Mutex::new(None);
    let mut key = PathBuf::new();
    for c in Path::new(path).components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir
                if matches!(key.components().next_back(), Some(Component::Normal(_))) =>
            {
                key.pop();
            }
            c => key.push(c),
        }
    }
    LOCKS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .entry(key)
        .or_default()
        .clone()
}
