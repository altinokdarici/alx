//! Renders Markdown (what models write) as terminal text: bold, italic,
//! headings, lists, quotes, code, links and tables. With color off it falls
//! back to readable plain text (no escape codes, `#` kept on headings,
//! backticks kept around inline code).

use crate::render::Style;
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

/// Renders `md` for a terminal of about `width` columns. The result has no
/// trailing newline.
pub fn render(md: &str, style: &Style, width: usize) -> String {
    let options = Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_FOOTNOTES;
    let mut r = Renderer::new(style, width);
    for event in Parser::new_ext(md, options) {
        r.event(event);
    }
    r.out.trim_end().to_owned()
}

/// A block that prefixes every line it contains: a list item or a quote.
struct Container {
    /// Prefix for continuation lines.
    cont: String,
    /// Prefix for the first line only (a list marker), until used.
    first: Option<String>,
    /// Paint the prefix dim (quote bars).
    dim: bool,
}

#[derive(Default)]
struct Table {
    rows: Vec<Vec<String>>,
    header_rows: usize,
    row: Vec<String>,
    cell: String,
}

struct Renderer<'a> {
    style: &'a Style,
    width: usize,
    out: String,
    at_line_start: bool,
    last_line_blank: bool,
    /// Don't separate the next block from the previous one (first thing in a
    /// list item: the marker is already on its own line).
    suppress_blank: bool,
    containers: Vec<Container>,
    /// Active SGR codes (bold, italic, ...), combined when text is written.
    styles: Vec<&'static str>,
    /// One entry per open list: the next number, or `None` for bullets.
    lists: Vec<Option<u64>>,
    /// Open links: destination and the text shown for it.
    links: Vec<(String, String)>,
    in_code_block: bool,
    table: Option<Table>,
}

impl<'a> Renderer<'a> {
    fn new(style: &'a Style, width: usize) -> Self {
        Self {
            style,
            width,
            out: String::new(),
            at_line_start: true,
            last_line_blank: false,
            suppress_blank: false,
            containers: Vec::new(),
            styles: Vec::new(),
            lists: Vec::new(),
            links: Vec::new(),
            in_code_block: false,
            table: None,
        }
    }

    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(t) if self.in_code_block => self.code_lines(&t),
            Event::Text(t) => self.text(&t),
            Event::Code(c) => {
                if self.style.color {
                    self.segment(&c, Some("33"));
                } else {
                    self.segment(&format!("`{c}`"), None);
                }
            }
            Event::Html(h) | Event::InlineHtml(h) => self.text(&h),
            Event::SoftBreak | Event::HardBreak => self.text("\n"),
            Event::Rule => {
                self.block_start();
                let rule = "─".repeat(self.width.min(40));
                self.segment(&rule, Some("2"));
                self.newline();
            }
            Event::TaskListMarker(done) => self.text(if done { "[x] " } else { "[ ] " }),
            Event::FootnoteReference(name) => self.text(&format!("[{name}]")),
            _ => {}
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => self.block_start(),
            Tag::Heading { level, .. } => {
                self.block_start();
                if self.style.color {
                    self.styles.push(match level {
                        HeadingLevel::H1 => "1;4;36",
                        HeadingLevel::H2 => "1;36",
                        _ => "1",
                    });
                } else {
                    self.segment(&format!("{} ", "#".repeat(level as usize)), None);
                }
            }
            Tag::BlockQuote(_) => {
                self.block_start();
                self.containers.push(Container {
                    cont: if self.style.color { "│ " } else { "> " }.into(),
                    first: None,
                    dim: true,
                });
            }
            Tag::CodeBlock(kind) => {
                self.block_start();
                if let CodeBlockKind::Fenced(lang) = kind
                    && !lang.is_empty()
                {
                    self.segment(&lang, Some("2"));
                    self.newline();
                }
                self.in_code_block = true;
            }
            Tag::List(start) => {
                if self.lists.is_empty() {
                    self.block_start();
                } else if !self.at_line_start {
                    self.newline();
                }
                self.lists.push(start);
                self.suppress_blank = true;
            }
            Tag::Item => {
                let marker = match self.lists.last_mut() {
                    Some(Some(n)) => {
                        let m = format!("{n}. ");
                        *n += 1;
                        m
                    }
                    _ => "• ".to_owned(),
                };
                self.containers.push(Container {
                    cont: " ".repeat(marker.chars().count()),
                    first: Some(marker),
                    dim: false,
                });
            }
            Tag::Emphasis => self.styles.push("3"),
            Tag::Strong => self.styles.push("1"),
            Tag::Strikethrough => self.styles.push("9"),
            Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => {
                self.styles.push("4");
                self.links.push((dest_url.to_string(), String::new()));
            }
            Tag::Table(_) => {
                self.block_start();
                self.table = Some(Table::default());
            }
            Tag::TableHead | Tag::TableRow => {}
            Tag::TableCell => {
                if let Some(t) = &mut self.table {
                    t.cell.clear();
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.newline_if_open(),
            TagEnd::Heading(_) => {
                if self.style.color {
                    self.styles.pop();
                }
                self.newline_if_open();
            }
            TagEnd::BlockQuote(_) => {
                self.containers.pop();
            }
            TagEnd::CodeBlock => {
                self.in_code_block = false;
                self.newline_if_open();
            }
            TagEnd::List(_) => {
                self.lists.pop();
                self.newline_if_open();
            }
            TagEnd::Item => {
                self.newline_if_open();
                self.containers.pop();
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.styles.pop();
            }
            TagEnd::Link | TagEnd::Image => {
                self.styles.pop();
                if let Some((url, text)) = self.links.pop()
                    && url != text
                    && !url.is_empty()
                {
                    self.segment(&format!(" ({url})"), Some("2"));
                }
            }
            TagEnd::TableHead => {
                if let Some(t) = &mut self.table {
                    t.rows.push(std::mem::take(&mut t.row));
                    t.header_rows = t.rows.len();
                }
            }
            TagEnd::TableRow => {
                if let Some(t) = &mut self.table {
                    t.rows.push(std::mem::take(&mut t.row));
                }
            }
            TagEnd::TableCell => {
                if let Some(t) = &mut self.table {
                    let cell = std::mem::take(&mut t.cell);
                    t.row.push(cell);
                }
            }
            TagEnd::Table => {
                if let Some(t) = self.table.take() {
                    self.table_out(&t);
                }
            }
            _ => {}
        }
    }

    // --- output primitives ---

    fn newline(&mut self) {
        self.out.push('\n');
        self.at_line_start = true;
    }

    fn newline_if_open(&mut self) {
        if !self.at_line_start {
            self.newline();
        }
    }

    /// The prefix for a new line, consuming list markers.
    fn prefix(&mut self) -> String {
        let style = self.style;
        let mut p = String::new();
        for c in &mut self.containers {
            let part = c.first.take().unwrap_or_else(|| c.cont.clone());
            p.push_str(&if c.dim { style.paint("2", &part) } else { part });
        }
        p
    }

    /// The prefix for an empty line: quote bars without trailing space.
    fn blank_prefix(&self) -> String {
        self.containers
            .iter()
            .map(|c| {
                if c.dim {
                    self.style.paint("2", c.cont.trim_end())
                } else {
                    c.cont.clone()
                }
            })
            .collect::<String>()
            .trim_end()
            .to_owned()
    }

    /// Starts a block: finishes the current line and leaves one empty line.
    fn block_start(&mut self) {
        if self.out.is_empty() {
            return;
        }
        self.newline_if_open();
        if self.suppress_blank {
            return;
        }
        if !self.last_line_blank {
            let blank = self.blank_prefix();
            self.out.push_str(&blank);
            self.out.push('\n');
            self.last_line_blank = true;
        }
    }

    fn text(&mut self, s: &str) {
        for (i, seg) in s.split('\n').enumerate() {
            if i > 0 {
                self.newline();
            }
            if !seg.is_empty() {
                self.segment(seg, None);
            }
        }
    }

    /// Writes one line's worth of text in the active styles (plus `extra`).
    fn segment(&mut self, s: &str, extra: Option<&str>) {
        for (_, text) in &mut self.links {
            text.push_str(s);
        }
        let mut codes: Vec<&str> = self.styles.clone();
        codes.extend(extra);
        let styled = if self.style.color && !codes.is_empty() {
            self.style.paint(&codes.join(";"), s)
        } else {
            s.to_owned()
        };
        if let Some(t) = &mut self.table {
            t.cell.push_str(&styled);
            return;
        }
        if self.at_line_start {
            let prefix = self.prefix();
            self.out.push_str(&prefix);
            self.at_line_start = false;
        }
        self.out.push_str(&styled);
        self.last_line_blank = false;
        self.suppress_blank = false;
    }

    fn code_lines(&mut self, text: &str) {
        for line in text.lines() {
            if self.at_line_start {
                let prefix = self.prefix();
                self.out.push_str(&prefix);
            }
            self.out.push_str("  ");
            let line = if self.style.color {
                self.style.paint("36", line)
            } else {
                line.to_owned()
            };
            self.out.push_str(&line);
            self.newline();
        }
        self.last_line_blank = false;
        self.suppress_blank = false;
    }

    fn table_out(&mut self, t: &Table) {
        let cols = t.rows.iter().map(Vec::len).max().unwrap_or(0);
        if cols == 0 {
            return;
        }
        let mut widths = vec![1; cols];
        for row in &t.rows {
            for (w, cell) in widths.iter_mut().zip(row) {
                *w = (*w).max(visible_width(cell));
            }
        }
        // Shrink the widest columns until the table fits the line.
        let prefix_w: usize = self.containers.iter().map(|c| c.cont.chars().count()).sum();
        let avail = self.width.saturating_sub(prefix_w);
        let seps = 3 * (cols - 1);
        let min_col = 6;
        while widths.iter().sum::<usize>() + seps > avail {
            let (i, &w) = widths.iter().enumerate().max_by_key(|(_, w)| **w).unwrap();
            if w <= min_col {
                break;
            }
            widths[i] = w - 1;
        }
        let sep = self.style.paint("2", " │ ");
        for (i, row) in t.rows.iter().enumerate() {
            let wrapped: Vec<Vec<String>> = (0..cols)
                .map(|c| wrap_ansi(row.get(c).map_or("", String::as_str), widths[c]))
                .collect();
            let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
            for l in 0..height {
                let cells: Vec<String> = (0..cols)
                    .map(|c| {
                        let cell = wrapped[c].get(l).map_or("", String::as_str);
                        let pad = " ".repeat(widths[c].saturating_sub(visible_width(cell)));
                        if i < t.header_rows {
                            format!("{}{pad}", self.style.paint("1", cell))
                        } else {
                            format!("{cell}{pad}")
                        }
                    })
                    .collect();
                self.table_line(cells.join(&sep).trim_end());
            }
            if i + 1 == t.header_rows {
                let rule: Vec<String> = widths.iter().map(|w| "─".repeat(*w)).collect();
                self.table_line(&self.style.paint("2", &rule.join("─┼─")));
            }
        }
    }

    fn table_line(&mut self, line: &str) {
        let prefix = self.prefix();
        self.out.push_str(&prefix);
        self.out.push_str(line);
        self.newline();
        self.last_line_blank = false;
        self.suppress_blank = false;
    }
}

/// Word-wraps `s` to `width` visible columns, keeping ANSI styling intact
/// across the breaks. Words longer than `width` are split.
fn wrap_ansi(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut line_w = 0;
    let mut active = String::new(); // SGR sequences in effect
    for word in s.split(' ') {
        // Split the word into pieces that each fit on a line.
        let mut piece = String::new();
        let mut piece_w = 0;
        let mut pieces = Vec::new();
        let mut chars = word.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                let mut seq = String::from(c);
                for n in chars.by_ref() {
                    seq.push(n);
                    if n == 'm' {
                        break;
                    }
                }
                piece.push_str(&seq);
                continue;
            }
            if piece_w == width {
                pieces.push(std::mem::take(&mut piece));
                piece_w = 0;
            }
            piece.push(c);
            piece_w += 1;
        }
        pieces.push(piece);
        for p in pieces {
            let w = visible_width(&p);
            if line_w > 0 && line_w + 1 + w > width {
                if !active.is_empty() {
                    line.push_str("\x1b[0m");
                }
                lines.push(std::mem::take(&mut line));
                line.push_str(&active);
                line_w = 0;
            }
            if line_w > 0 {
                line.push(' ');
                line_w += 1;
            }
            line.push_str(&p);
            line_w += w;
            // Track which styles are on at the end of this piece.
            let mut rest = p.as_str();
            while let Some(at) = rest.find('\x1b') {
                let end = rest[at..].find('m').map_or(rest.len(), |e| at + e + 1);
                if &rest[at..end] == "\x1b[0m" {
                    active.clear();
                } else {
                    active.push_str(&rest[at..end]);
                }
                rest = &rest[end..];
            }
        }
    }
    lines.push(line);
    lines
}

/// Width of `s` on screen: its characters, without ANSI escape sequences.
fn visible_width(s: &str) -> usize {
    let mut n = 0;
    let mut in_escape = false;
    for c in s.chars() {
        match (in_escape, c) {
            (false, '\x1b') => in_escape = true,
            (true, 'm') => in_escape = false,
            (true, _) => {}
            (false, _) => n += 1,
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: Style = Style { color: false };
    const COLOR: Style = Style { color: true };

    fn plain(md: &str) -> String {
        render(md, &PLAIN, 80)
    }

    #[test]
    fn plain_text_passes_through() {
        assert_eq!(plain("hello"), "hello");
        assert_eq!(plain("a\nb"), "a\nb");
    }

    #[test]
    fn emphasis_markers_are_removed() {
        assert_eq!(
            plain("**bold** and *it* and ~~gone~~"),
            "bold and it and gone"
        );
    }

    #[test]
    fn emphasis_is_styled_with_color() {
        let s = render("**bold**", &COLOR, 80);
        assert_eq!(s, "\x1b[1mbold\x1b[0m");
        let s = render("**a *b***", &COLOR, 80);
        assert!(s.contains("\x1b[1;3mb\x1b[0m"), "{s:?}");
    }

    #[test]
    fn inline_code_is_colored_or_backticked() {
        assert_eq!(plain("run `ls -la` now"), "run `ls -la` now");
        assert_eq!(
            render("`x`", &COLOR, 80),
            "\x1b[33mx\x1b[0m",
            "no backticks when colored"
        );
    }

    #[test]
    fn headings_and_paragraphs_are_separated_by_blank_lines() {
        assert_eq!(plain("# Title\ntext\n\nmore"), "# Title\n\ntext\n\nmore");
        let s = render("## Sub", &COLOR, 80);
        assert_eq!(s, "\x1b[1;36mSub\x1b[0m");
    }

    #[test]
    fn bullet_and_numbered_lists() {
        assert_eq!(plain("- a\n- b"), "• a\n• b");
        assert_eq!(plain("1. a\n2. b"), "1. a\n2. b");
        assert_eq!(plain("3. a\n4. b"), "3. a\n4. b");
    }

    #[test]
    fn nested_lists_indent() {
        assert_eq!(plain("- a\n  - b\n  - c\n- d"), "• a\n  • b\n  • c\n• d");
    }

    #[test]
    fn loose_lists_keep_blank_lines_between_items() {
        assert_eq!(plain("- a\n\n- b"), "• a\n\n• b");
    }

    #[test]
    fn list_item_continuation_is_indented() {
        assert_eq!(plain("- a\n\n  more\n- b"), "• a\n\n  more\n\n• b");
    }

    #[test]
    fn task_list() {
        assert_eq!(plain("- [x] done\n- [ ] todo"), "• [x] done\n• [ ] todo");
    }

    #[test]
    fn block_quote_gets_a_bar() {
        assert_eq!(plain("> hi\n>\n> there"), "> hi\n>\n> there");
        let s = render("> hi", &COLOR, 80);
        assert!(s.contains('│') && s.contains("hi"));
    }

    #[test]
    fn code_blocks_are_indented_and_keep_their_lines() {
        assert_eq!(
            plain("text\n\n```rust\nfn main() {}\n\nlet x = 1;\n```\n\nafter"),
            "text\n\nrust\n  fn main() {}\n  \n  let x = 1;\n\nafter"
        );
    }

    #[test]
    fn links_show_their_url() {
        assert_eq!(plain("[docs](https://x.dev)"), "docs (https://x.dev)");
        assert_eq!(plain("<https://x.dev>"), "https://x.dev");
    }

    #[test]
    fn rules() {
        assert_eq!(
            plain("a\n\n---\n\nb"),
            format!("a\n\n{}\n\nb", "─".repeat(40))
        );
    }

    #[test]
    fn tables_are_aligned() {
        let md = "| Name | Age |\n|---|---|\n| Bob | 3 |\n| Alexandra | 41 |";
        assert_eq!(
            plain(md),
            "Name      │ Age\n──────────┼────\nBob       │ 3\nAlexandra │ 41"
        );
    }

    #[test]
    fn styled_cells_align_by_visible_width() {
        let s = render("| a |\n|---|\n| **b** |", &COLOR, 80);
        let widths: Vec<usize> = s.lines().map(visible_width).collect();
        assert!(widths.iter().all(|w| *w == 1), "{widths:?}\n{s:?}");
    }

    #[test]
    fn html_and_empty_input_do_not_panic() {
        assert_eq!(plain(""), "");
        assert_eq!(plain("<b>x</b>"), "<b>x</b>");
    }

    #[test]
    fn wide_tables_shrink_and_wrap_to_the_width() {
        let md = "| Name | Description |\n|---|---|\n| a | one two three four five six seven eight nine ten |";
        let s = render(md, &PLAIN, 30);
        assert!(s.lines().all(|l| visible_width(l) <= 30), "{s}");
        assert!(s.contains("seven"), "{s}");
        assert!(s.lines().count() > 4, "cell should wrap:\n{s}");
    }

    #[test]
    fn wrapped_styles_continue_on_the_next_line() {
        let lines = wrap_ansi("\x1b[1maaa bbb ccc\x1b[0m", 7);
        assert_eq!(lines, ["\x1b[1maaa bbb\x1b[0m", "\x1b[1mccc\x1b[0m"]);
    }
}
