//! `web_fetch`: read a web page as text.

use crate::executor::FetchResponse;
use crate::tool::{Tool, ToolContext, ToolOutput};
use alx_protocol::ToolSpec;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Fetches a URL and returns its body as text, HTML converted to plain text.
/// The request goes through the executor, so it runs where commands run.
#[derive(Debug, Clone, Default)]
pub struct WebFetchTool;

impl WebFetchTool {
    const MAX_OUTPUT_BYTES: usize = 50_000;
    /// Bodies are read up to this size; past it the page is cut.
    const MAX_BODY_BYTES: usize = 5_000_000;
}

#[async_trait]
impl Tool for WebFetchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_fetch".into(),
            description: "Fetch an http(s) URL and return its content as text (HTML is \
                          converted to plain text). Returns up to 50000 bytes starting at \
                          byte `offset` (default 0), with a note on where to continue."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The http or https URL" },
                    "offset": { "type": "integer", "description": "Byte offset to start at" }
                },
                "required": ["url"]
            }),
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    async fn call(&self, input: Value, ctx: &ToolContext) -> ToolOutput {
        let Some(url) = input.get("url").and_then(Value::as_str) else {
            return ToolOutput::error("web_fetch needs a `url` string");
        };
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return ToolOutput::error("web_fetch only fetches http:// and https:// URLs");
        }
        let offset = input.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        match ctx.executor.fetch(url, Self::MAX_BODY_BYTES).await {
            Ok(resp) if (200..300).contains(&resp.status) => {
                ToolOutput::ok(page(&body_text(&resp), offset, Self::MAX_OUTPUT_BYTES))
            }
            Ok(resp) => ToolOutput::error(format!(
                "HTTP {}\n{}",
                resp.status,
                page(&body_text(&resp), 0, 2_000)
            )),
            Err(e) => ToolOutput::error(format!("Could not fetch {url}: {e:#}")),
        }
    }
}

fn body_text(resp: &FetchResponse) -> String {
    let raw = String::from_utf8_lossy(&resp.body);
    if resp
        .content_type
        .as_deref()
        .is_some_and(|ct| ct.contains("html"))
    {
        html_to_text(&raw)
    } else {
        raw.into_owned()
    }
}

/// `text[offset..]` cut at `max` bytes (on char boundaries), with a note on
/// where to continue.
fn page(text: &str, offset: usize, max: usize) -> String {
    let start = floor_char(text, offset.min(text.len()));
    let mut end = floor_char(text, (start + max).min(text.len()));
    // Always make progress, even if one char is wider than `max`.
    while end == start && end < text.len() {
        end += 1;
        end = (end..=text.len())
            .find(|&i| text.is_char_boundary(i))
            .unwrap();
    }
    let mut out = text[start..end].to_owned();
    if end < text.len() {
        out.push_str(&format!(
            "\n[... {} more bytes; read on with offset {end} ...]\n",
            text.len() - end
        ));
    }
    out
}

fn floor_char(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// A rough HTML to text conversion: drops scripts, styles and tags, breaks
/// lines at block elements, decodes common entities, collapses whitespace.
pub fn html_to_text(html: &str) -> String {
    const BLOCKS: &[&str] = &[
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "section",
        "article",
        "header",
        "footer",
        "table",
        "ul",
        "ol",
        "blockquote",
        "hr",
        "title",
    ];
    const SKIP: &[&str] = &["script", "style", "noscript", "svg", "head"];
    let mut out = String::new();
    let mut rest = html;
    let mut skipping: Option<String> = None;
    while let Some(lt) = rest.find('<') {
        if skipping.is_none() {
            out.push_str(&rest[..lt]);
        }
        rest = &rest[lt..];
        if rest.starts_with("<!--") {
            rest = rest.find("-->").map_or("", |e| &rest[e + 3..]);
            continue;
        }
        let Some(gt) = rest.find('>') else {
            rest = "";
            break;
        };
        let tag = &rest[1..gt];
        rest = &rest[gt + 1..];
        let closing = tag.starts_with('/');
        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if let Some(skip) = &skipping {
            if closing && *skip == name {
                skipping = None;
            }
            continue;
        }
        // `<head>` holds `<title>`; keep the title, skip the rest of head.
        if !closing && SKIP.contains(&name.as_str()) && !tag.ends_with('/') {
            if name == "head"
                && let Some(t) = head_title(rest)
            {
                out.push_str(&t);
                out.push('\n');
            }
            skipping = Some(name);
            continue;
        }
        if BLOCKS.contains(&name.as_str()) {
            out.push('\n');
        }
    }
    if skipping.is_none() {
        out.push_str(rest);
    }
    collapse(&decode_entities(&out))
}

fn head_title(head: &str) -> Option<String> {
    let lower = head.to_ascii_lowercase();
    let end = lower.find("</head")?;
    let start = lower[..end].find("<title")?;
    let open = start + lower[start..].find('>')? + 1;
    let close = open + lower[open..end].find("</title")?;
    Some(head[open..close].trim().to_owned())
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let decoded = rest.find(';').filter(|&e| e <= 10).and_then(|e| {
            let c = match &rest[1..e] {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" | "#39" => Some('\''),
                "nbsp" => Some(' '),
                ent if ent.starts_with("#x") || ent.starts_with("#X") => {
                    u32::from_str_radix(&ent[2..], 16)
                        .ok()
                        .and_then(char::from_u32)
                }
                ent if ent.starts_with('#') => ent[1..].parse().ok().and_then(char::from_u32),
                _ => None,
            }?;
            Some((c, e + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Collapses runs of spaces within lines and keeps at most one blank line.
fn collapse(s: &str) -> String {
    let mut out = String::new();
    let mut blank = true;
    for line in s.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            if !blank {
                out.push('\n');
                blank = true;
            }
        } else {
            out.push_str(&line);
            out.push('\n');
            blank = false;
        }
    }
    let len = out.trim_end().len();
    out.truncate(len);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_becomes_readable_text() {
        let html = "<html><head><title>T &amp; U</title><style>x{}</style></head>\
                    <body><script>evil()</script><!-- c --><h1>Hi</h1>\
                    <p>a&nbsp;<b>bold</b>   word &#x41;&#66;</p><ul><li>one</li><li>two</li></ul>\
                    </body></html>";
        assert_eq!(
            html_to_text(html),
            "T & U\n\nHi\n\na bold word AB\n\none\n\ntwo\n"
        );
    }

    #[test]
    fn pages_with_a_continue_offset() {
        assert_eq!(
            page("abcdef", 0, 4),
            "abcd\n[... 2 more bytes; read on with offset 4 ...]\n"
        );
        assert_eq!(page("abcdef", 4, 4), "ef");
        assert_eq!(
            page("éa", 0, 1),
            "é\n[... 1 more bytes; read on with offset 2 ...]\n"
        );
    }

    #[tokio::test]
    async fn rejects_non_http_urls() {
        let ctx = ToolContext {
            executor: std::sync::Arc::new(crate::LocalExecutor),
        };
        let out = WebFetchTool
            .call(json!({"url": "file:///etc/passwd"}), &ctx)
            .await;
        assert!(out.is_error);
    }

    struct FakeWeb;

    #[async_trait]
    impl crate::Executor for FakeWeb {
        async fn run(&self, _: crate::ExecRequest) -> std::io::Result<crate::ExecOutput> {
            unreachable!()
        }
        async fn fetch(&self, url: &str, _: usize) -> std::io::Result<FetchResponse> {
            Ok(FetchResponse {
                status: 200,
                content_type: Some("text/html".into()),
                body: format!("<p>from {url}</p>").into_bytes(),
            })
        }
    }

    #[tokio::test]
    async fn fetches_through_the_executor() {
        let ctx = ToolContext {
            executor: std::sync::Arc::new(FakeWeb),
        };
        let out = WebFetchTool
            .call(json!({"url": "https://x.test/"}), &ctx)
            .await;
        assert!(!out.is_error);
        assert_eq!(out.content.trim(), "from https://x.test/");
    }
}
