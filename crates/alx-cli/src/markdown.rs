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

/// Cuts streamed Markdown into blocks that can each be rendered on their own
/// (paragraphs, lists, quotes, tables, fenced code), as soon as they are
/// complete. Only whole lines are looked at, so a half-written `**` or table
/// row is never rendered early.
///
/// Blocks end at a blank line, except that a fenced code block runs to its
/// closing fence, and a list carries on over blank lines when the next line
/// is indented or another item (a "loose" list).
#[derive(Default)]
pub struct BlockSplitter {
    /// Text not yet seen a newline for.
    partial: String,
    block: String,
    fence: Option<String>,
    /// Blank lines after a list, held until we know whether it continues.
    held: String,
    awaiting: bool,
}

fn is_list_item(line: &str) -> bool {
    let t = line.trim_start();
    if ["- ", "* ", "+ "].iter().any(|m| t.starts_with(m)) {
        return true;
    }
    let digits = t.chars().take_while(char::is_ascii_digit).count();
    digits > 0 && t[digits..].starts_with(['.', ')']) && t[digits + 1..].starts_with(' ')
}

fn fence_marker(line: &str) -> Option<String> {
    let t = line.trim_start();
    let c = t.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let n = t.chars().take_while(|&x| x == c).count();
    (n >= 3).then(|| c.to_string().repeat(n))
}

impl BlockSplitter {
    /// Adds streamed text; returns the blocks it completed.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.partial.push_str(text);
        let mut out = Vec::new();
        while let Some(nl) = self.partial.find('\n') {
            let line: String = self.partial.drain(..=nl).collect();
            self.line(&line, &mut out);
        }
        out
    }

    /// The end of the text: returns whatever is left, however incomplete.
    pub fn finish(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.partial.is_empty() {
            let line = std::mem::take(&mut self.partial) + "\n";
            self.line(&line, &mut out);
        }
        self.emit(&mut out);
        out
    }

    fn emit(&mut self, out: &mut Vec<String>) {
        let block = std::mem::take(&mut self.block);
        self.held.clear();
        self.awaiting = false;
        self.fence = None;
        if !block.trim().is_empty() {
            out.push(block);
        }
    }

    fn line(&mut self, line: &str, out: &mut Vec<String>) {
        let text = line.trim_end_matches(['\n', '\r']);
        if let Some(marker) = &self.fence {
            self.block.push_str(line);
            if text.trim_start().starts_with(marker.as_str()) {
                self.emit(out);
            }
            return;
        }
        if self.awaiting {
            if text.trim().is_empty() {
                self.held.push_str(line);
                return;
            }
            if text.starts_with([' ', '\t']) || is_list_item(text) {
                let held = std::mem::take(&mut self.held);
                self.block.push_str(&held);
                self.awaiting = false;
            } else {
                self.emit(out);
            }
        }
        if text.trim().is_empty() {
            let is_list = self.block.lines().next().is_some_and(is_list_item);
            if is_list {
                self.awaiting = true;
                self.held.push_str(line);
            } else {
                self.emit(out);
            }
            return;
        }
        self.block.push_str(line);
        if let Some(marker) = fence_marker(text) {
            self.fence = Some(marker);
        }
    }
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
    /// Visible width of the current output line, prefix included.
    col: usize,
    /// The current line has text after its prefix (list marker, quote bar).
    line_has_text: bool,
    /// A space is owed before the next word (it is dropped at a wrap).
    pending_space: bool,
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
            col: 0,
            line_has_text: false,
            pending_space: false,
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
        self.col = 0;
        self.line_has_text = false;
        self.pending_space = false;
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
        // Word-wrap: break before a word that would run past the width.
        let paint = |w: &str| {
            if self.style.color && !codes.is_empty() {
                self.style.paint(&codes.join(";"), w)
            } else {
                w.to_owned()
            }
        };
        let mut pieces: Vec<(String, usize, bool)> = Vec::new(); // (text, width, space owed)
        let mut pending = self.pending_space;
        for (i, word) in s.split(' ').enumerate() {
            if i > 0 {
                pending = true;
            }
            if word.is_empty() {
                continue;
            }
            pieces.push((paint(word), word.chars().count(), pending));
            pending = false;
        }
        let trailing_space = s.ends_with(' ');
        for (text, w, owed) in pieces {
            if self.at_line_start {
                let prefix = self.prefix();
                self.col = visible_width(&prefix);
                self.out.push_str(&prefix);
                self.at_line_start = false;
            }
            let gap = usize::from((owed || self.pending_space) && self.line_has_text);
            self.pending_space = false;
            if self.line_has_text && self.col + gap + w > self.width {
                self.newline();
                let prefix = self.prefix();
                self.col = visible_width(&prefix);
                self.out.push_str(&prefix);
                self.at_line_start = false;
            } else if gap == 1 {
                self.out.push(' ');
                self.col += 1;
            }
            self.out.push_str(&text);
            self.col += w;
            self.line_has_text = true;
        }
        self.pending_space = trailing_space;
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
        // Shrink the widest columns until the table fits the line. Borders
        // take 3 columns between cells and 4 around the outside.
        let prefix_w: usize = self.containers.iter().map(|c| c.cont.chars().count()).sum();
        let avail = self.width.saturating_sub(prefix_w);
        let borders = 3 * (cols - 1) + 4;
        let min_col = 6;
        while widths.iter().sum::<usize>() + borders > avail {
            let (i, &w) = widths.iter().enumerate().max_by_key(|(_, w)| **w).unwrap();
            if w <= min_col {
                break;
            }
            widths[i] = w - 1;
        }
        let dim = |x: &str| self.style.paint("2", x);
        let rule = |l: &str, m: &str, r: &str| {
            let segs: Vec<String> = widths.iter().map(|w| "─".repeat(w + 2)).collect();
            dim(&format!("{l}{}{r}", segs.join(m)))
        };
        self.table_line(&rule("┌", "┬", "┐"));
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
                let bar = dim("│");
                self.table_line(&format!("{bar} {} {bar}", cells.join(&format!(" {bar} "))));
            }
            if i + 1 == t.rows.len() {
                self.table_line(&rule("└", "┴", "┘"));
            } else {
                self.table_line(&rule("├", "┼", "┤"));
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
            "┌───────────┬─────┐\n│ Name      │ Age │\n├───────────┼─────┤\n│ Bob       │ 3   │\n├───────────┼─────┤\n│ Alexandra │ 41  │\n└───────────┴─────┘"
        );
    }

    #[test]
    fn styled_cells_align_by_visible_width() {
        let s = render("| a |\n|---|\n| **b** |", &COLOR, 80);
        let widths: Vec<usize> = s.lines().map(visible_width).collect();
        assert!(widths.iter().all(|w| *w == 5), "{widths:?}\n{s:?}");
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

    #[test]
    fn paragraphs_wrap_at_word_boundaries() {
        let s = render("one two three four five six seven", &PLAIN, 15);
        assert_eq!(s, "one two three\nfour five six\nseven");
    }

    #[test]
    fn wrapped_list_items_hang_under_their_text() {
        let s = render("- one two three four five six", &PLAIN, 14);
        assert_eq!(s, "• one two\n  three four\n  five six");
        let s = render("> one two three four", &PLAIN, 12);
        assert_eq!(s, "> one two\n> three four");
    }

    #[test]
    fn wrapping_keeps_spaces_around_styled_words() {
        let s = render("a **b c** d", &PLAIN, 80);
        assert_eq!(s, "a b c d");
        let s = render("aaa **bbb** ccc", &PLAIN, 8);
        assert_eq!(s, "aaa bbb\nccc");
    }

    #[test]
    fn long_words_are_not_split() {
        let s = render("x https://example.com/a/very/long/url y", &PLAIN, 10);
        assert_eq!(s, "x\nhttps://example.com/a/very/long/url\ny");
    }

    fn split(chunks: &[&str]) -> Vec<String> {
        let mut sp = BlockSplitter::default();
        let mut out = Vec::new();
        for c in chunks {
            out.extend(sp.push(c));
        }
        out.extend(sp.finish());
        out
    }

    #[test]
    fn splitter_cuts_at_blank_lines_and_waits_for_whole_lines() {
        let mut sp = BlockSplitter::default();
        assert!(sp.push("one **bo").is_empty());
        assert!(sp.push("ld**\ntwo\n").is_empty());
        assert_eq!(sp.push("\nthree"), ["one **bold**\ntwo\n"]);
        assert_eq!(sp.finish(), ["three\n"]);
    }

    #[test]
    fn splitter_keeps_fenced_code_together() {
        let blocks = split(&["intro\n\n```rust\nlet a = 1;\n\nlet b = 2;\n```\nafter\n"]);
        assert_eq!(
            blocks,
            [
                "intro\n",
                "```rust\nlet a = 1;\n\nlet b = 2;\n```\n",
                "after\n"
            ]
        );
        // An unclosed fence is flushed at the end.
        assert_eq!(split(&["```\ncode\n"]), ["```\ncode\n"]);
    }

    #[test]
    fn splitter_keeps_loose_lists_together_but_not_what_follows() {
        let blocks = split(&["1. a\n\n2. b\n\n   more\n\nDone.\n"]);
        assert_eq!(blocks, ["1. a\n\n2. b\n\n   more\n", "Done.\n"]);
    }

    #[test]
    fn splitter_keeps_tables_whole_and_is_chunk_independent() {
        let md = "| a | b |\n|---|---|\n| 1 | 2 |\n\ntext\n\n- x\n- y\n";
        let whole = split(&[md]);
        assert_eq!(whole.len(), 3);
        let bytes: Vec<String> = md.chars().map(String::from).collect();
        let refs: Vec<&str> = bytes.iter().map(String::as_str).collect();
        assert_eq!(split(&refs), whole);
    }
}
