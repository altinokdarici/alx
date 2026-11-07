//! `session/list` and the `SessionInfo` it returns, from the session store.

use agent_client_protocol::Error;
use agent_client_protocol::schema::v1::{
    ListSessionsRequest, ListSessionsResponse, Meta, SessionInfo,
};
use alx_core::SessionSummary;
use std::path::Path;

/// Sessions per `session/list` page.
pub const PAGE_SIZE: usize = 50;

/// `_meta` key of a `SessionInfo` (and a `session/resume` reply) with the
/// number of messages in the session.
pub const MESSAGE_COUNT_META: &str = "messageCount";

/// A Unix timestamp as ISO 8601 UTC, `YYYY-MM-DDTHH:MM:SSZ`.
pub fn iso8601(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60
    )
}

pub fn message_count_meta(count: usize) -> Meta {
    let mut meta = Meta::new();
    meta.insert(MESSAGE_COUNT_META.into(), count.into());
    meta
}

fn info(s: SessionSummary) -> SessionInfo {
    let preview = s.preview.trim();
    SessionInfo::new(s.id, s.cwd.unwrap_or_default())
        .title((!preview.is_empty()).then(|| preview.to_owned()))
        .updated_at(iso8601(s.updated_at))
        .meta(message_count_meta(s.message_count))
}

/// One page of `all` (most recent first), keeping only sessions started in
/// `req.cwd` if it is set. The cursor is the index of the next session, so a
/// session saved between two pages can shift the list by one.
pub fn page(
    all: Vec<SessionSummary>,
    req: &ListSessionsRequest,
) -> Result<ListSessionsResponse, Error> {
    let start = match &req.cursor {
        Some(c) => c
            .parse::<usize>()
            .map_err(|_| Error::invalid_params().data(format!("bad cursor `{c}`")))?,
        None => 0,
    };
    let wanted = |s: &SessionSummary| match &req.cwd {
        Some(dir) => s.cwd.as_deref().map(Path::new) == Some(dir.as_path()),
        None => true,
    };
    let matching: Vec<_> = all.into_iter().filter(wanted).collect();
    let total = matching.len();
    let sessions: Vec<_> = matching
        .into_iter()
        .skip(start)
        .take(PAGE_SIZE)
        .map(info)
        .collect();
    let next = start + sessions.len();
    Ok(ListSessionsResponse::new(sessions).next_cursor((next < total).then(|| next.to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(id: usize, cwd: &str) -> SessionSummary {
        SessionSummary {
            id: format!("s{id}"),
            created_at: 0,
            updated_at: 0,
            cwd: Some(cwd.into()),
            message_count: id,
            preview: format!("hello {id}"),
        }
    }

    #[test]
    fn timestamps_are_iso8601_utc() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601(1_709_210_096), "2024-02-29T12:34:56Z");
    }

    #[test]
    fn pages_follow_the_cursor_and_filter_by_cwd() {
        let all: Vec<_> = (0..PAGE_SIZE + 3)
            .map(|i| summary(i, if i == 1 { "/other" } else { "/work" }))
            .collect();
        let req = ListSessionsRequest::new().cwd(std::path::PathBuf::from("/work"));
        let first = page(all.clone(), &req).unwrap();
        assert_eq!(first.sessions.len(), PAGE_SIZE);
        assert!(
            first
                .sessions
                .iter()
                .all(|s| s.session_id.0.as_ref() != "s1")
        );
        let cursor = first.next_cursor.expect("more to come");
        let second = page(all, &req.cursor(cursor)).unwrap();
        let ids: Vec<_> = second
            .sessions
            .iter()
            .map(|s| s.session_id.0.to_string())
            .collect();
        assert_eq!(
            ids,
            [format!("s{}", PAGE_SIZE + 1), format!("s{}", PAGE_SIZE + 2)]
        );
        assert_eq!(second.next_cursor, None);
        let meta = second.sessions[0].meta.as_ref().unwrap();
        assert_eq!(meta[MESSAGE_COUNT_META], PAGE_SIZE + 1);
    }

    #[test]
    fn a_bad_cursor_is_an_error() {
        assert!(page(vec![], &ListSessionsRequest::new().cursor("x".to_string())).is_err());
    }
}
