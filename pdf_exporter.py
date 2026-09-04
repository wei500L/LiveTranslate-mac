"""PDF meeting-minute export via QTextDocument + QPdfWriter.

Uses the Qt stack the app already ships (no new dependency): a
``QTextDocument`` carries the Markdown rendering, ``QPdfWriter`` lays it out
on A4 with margins, and a header/footer pass adds page numbers. The output
is real selectable text — CJK and Cyrillic render through the same font
fallback Qt uses on screen, so no tofu boxes.

Exports never embed credentials: only the session's own metadata (title,
date, duration, engines, languages) and content the user chose to include.
"""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QMarginsF, QRectF, QSizeF, Qt
from PyQt6.QtGui import (
    QFont,
    QFontDatabase,
    QPainter,
    QPageLayout,
    QPageSize,
    QPdfWriter,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)

# Millimetre-based A4 layout, mirrored from typical word-processor defaults.
_PAGE_MARGIN_MM = 18.0
_FONT_BODY_PT = 10.5
_FONT_TITLE_PT = 18.0
_FONT_TIMESTAMP_PT = 8.0
_UNSAFE_FILENAME = re.compile(r"[^\w\-.()\[\] ]+", re.UNICODE)


def safe_filename(title: str, when: str, ext: str = "pdf") -> str:
    """Filesystem-safe default name from the meeting title and date."""
    base = _UNSAFE_FILENAME.sub("", (title or "").strip()).strip()
    base = base.replace(" ", "_") or "meeting"
    date = _UNSAFE_FILENAME.sub("", (when or "").strip()) or "meeting"
    return f"{base}_{date}.{ext}"


def _default_font() -> QFont:
    """A font with honest CJK coverage on both platforms."""
    families = QFontDatabase.families()
    # Windows-first names resolve there, the PingFang/Hiragino ones on macOS;
    # Noto covers Linux. Cross-platform order: any hit wins on its platform.
    preferred = (
        "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei"
    )
    for name in preferred:
        if name in families:
            return QFont(name)
    for name in families:
        if "CJK" in name or "Noto" in name or "Hei" in name or "Song" in name:
            return QFont(name)
    return QFont()


def export_pdf(
    path,
    *,
    title: str,
    meta_rows: list[tuple[str, str]],
    summary_markdown: str | None,
    entries: list[dict] | None = None,
    show_original: bool = True,
    progress=None,
) -> bool:
    """Render the meeting PDF. Returns success.

    ``entries`` with ``include_entries=True`` appends the full bilingual
    record; each entry renders timestamp (small, grey) + translation (body)
    + original (secondary). Long records render incrementally into the
    document before the single print pass, so the UI thread is only held
    by one print job, not by widget-per-entry construction.
    """
    try:
        target = Path(path)
        if target.exists() and not target.is_file():
            return False
        parent = target.parent if str(target.parent) else Path(".")
        if not parent.is_dir():
            return False
        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        # QPdfWriter works in device pixels (1200 dpi default); margins in mm.
        writer.setPageMargins(
            QMarginsF(_PAGE_MARGIN_MM, _PAGE_MARGIN_MM, _PAGE_MARGIN_MM, _PAGE_MARGIN_MM),
            QPageLayout.Unit.Millimeter,
        )
        writer.setTitle(title)
        writer.setCreator("LiveTranslate")

        doc = QTextDocument()
        doc.setDefaultFont(_default_font())
        body_font = doc.defaultFont()
        body_font.setPointSizeF(_FONT_BODY_PT)
        doc.setDefaultFont(body_font)

        cursor = QTextCursor(doc)
        _write_header(cursor, title, meta_rows)
        if summary_markdown:
            cursor.insertBlock()
            _write_summary(cursor, summary_markdown)
        if entries:
            _write_entries(cursor, entries, show_original, progress)

        _print_with_page_numbers(writer, doc, title)
        return True
    except Exception:
        import logging

        logging.getLogger("LiveTranslate.PDF").error(
            "PDF export failed", exc_info=True
        )
        return False


# --- document assembly -----------------------------------------------------------

def _write_header(cursor: QTextCursor, title: str, meta_rows: list[tuple[str, str]]):
    title_fmt = QTextCharFormat()
    font = _default_font()
    font.setPointSizeF(_FONT_TITLE_PT)
    font.setBold(True)
    title_fmt.setFont(font)
    cursor.insertText((title or "").strip() + "\n", title_fmt)

    meta_fmt = QTextCharFormat()
    mfont = _default_font()
    mfont.setPointSizeF(8.5)
    meta_fmt.setFont(mfont)
    meta_fmt.setForeground(Qt.GlobalColor.darkGray)
    line = "   ·   ".join(f"{k}: {v}" for k, v in meta_rows if v)
    if line:
        cursor.insertText(line + "\n", meta_fmt)

    # A hairline separator under the header (thin grey text, prints cleanly).
    separator = QTextCharFormat()
    sfont = _default_font()
    sfont.setPointSizeF(2.0)
    separator.setFont(sfont)
    separator.setForeground(Qt.GlobalColor.lightGray)
    block = QTextBlockFormat()
    block.setTopMargin(2)
    block.setBottomMargin(8)
    cursor.insertBlock(block)
    cursor.insertText("_" * 120, separator)


def _write_summary(cursor: QTextCursor, markdown_text: str):
    """Render the AI minutes section from Markdown.

    Headings, bullets and bold get real formatting; everything else is
    plain paragraphs. A full Markdown engine is unnecessary for the
    structured output the summary prompts produce.
    """
    lines = (markdown_text or "").splitlines()
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            cursor.insertBlock()
            continue
        if line.startswith("### "):
            _insert_heading(cursor, line[4:], 3)
        elif line.startswith("## "):
            _insert_heading(cursor, line[3:], 2)
        elif line.startswith("# "):
            _insert_heading(cursor, line[2:], 1)
        elif re.match(r"^[-*] +", line):
            cursor.insertText("  •  ")
            _insert_rich_text(cursor, re.sub(r"^[-*] +", "", line))
            cursor.insertBlock()
        elif line.strip() == "---":
            cursor.insertBlock()
        else:
            _insert_rich_text(cursor, line)
            cursor.insertBlock()


def _insert_heading(cursor: QTextCursor, text: str, level: int):
    fmt = QTextCharFormat()
    font = _default_font()
    font.setBold(True)
    font.setPointSizeF({1: 14.0, 2: 12.5, 3: 11.0}.get(level, 11.0))
    fmt.setFont(font)
    block = QTextBlockFormat()
    block.setTopMargin(8 if level > 1 else 12)
    block.setBottomMargin(4)
    cursor.insertBlock(block)
    cursor.insertText(text.strip() + "\n", fmt)


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")


def _insert_rich_text(cursor: QTextCursor, text: str):
    """Inline **bold** and `code` spans, left as plain text otherwise."""
    pos = 0
    for match in _BOLD_RE.finditer(text):
        if match.start() > pos:
            _insert_code_spans(cursor, text[pos:match.start()])
        fmt = QTextCharFormat()
        font = _default_font()
        font.setBold(True)
        fmt.setFont(font)
        cursor.insertText(match.group(1), fmt)
        pos = match.end()
    if pos < len(text):
        _insert_code_spans(cursor, text[pos:])


def _insert_code_spans(cursor: QTextCursor, text: str):
    pos = 0
    for match in _CODE_RE.finditer(text):
        if match.start() > pos:
            cursor.insertText(text[pos:match.start()])
        fmt = QTextCharFormat()
        font = _default_font()
        font.setFamily("Menlo")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        fmt.setFont(font)
        cursor.insertText(match.group(1), fmt)
        pos = match.end()
    if pos < len(text):
        cursor.insertText(text[pos:])


def _write_entries(
    cursor: QTextCursor, entries: list[dict], show_original: bool, progress
):
    _insert_heading(cursor, "", 2)  # spacing before the record section
    ts_fmt = QTextCharFormat()
    ts_font = _default_font()
    ts_font.setPointSizeF(_FONT_TIMESTAMP_PT)
    ts_fmt.setFont(ts_font)
    ts_fmt.setForeground(Qt.GlobalColor.gray)

    orig_fmt = QTextCharFormat()
    of = _default_font()
    of.setPointSizeF(9.0)
    of.setItalic(True)
    orig_fmt.setFont(of)
    orig_fmt.setForeground(Qt.GlobalColor.darkGray)

    total = len(entries)
    for i, entry in enumerate(entries):
        translation = entry.get("translation")
        original = entry.get("original")
        if not translation and not original:
            continue
        cursor.insertText(f"[{entry.get('timestamp', '')}] ", ts_fmt)
        _insert_rich_text(cursor, translation or original)
        cursor.insertBlock()
        if show_original and translation and original:
            cursor.insertText("    " + original + "\n", orig_fmt)
        if progress and i % 50 == 0:
            progress(i, total)
    if progress:
        progress(total, total)


# --- printing ----------------------------------------------------------------------

# Space reserved at the bottom of every page for the footer band, in
# millimetres. The content area is the page minus this reserve, so a page's
# last text line can never sit under the footer.
_FOOTER_RESERVE_MM = 12.0


def _print_with_page_numbers(writer: QPdfWriter, doc: QTextDocument, title: str):
    """Two-pass print: paginate once, then draw content + footer per page.

    Pagination and drawing share one content height: the page height minus
    the footer reserve. Each page translates the painter by its slice and
    asks ``drawContents`` for that slice's document rectangle — passing the
    same top-of-document rect every page (the previous version) re-painted
    page 1 on every page, because drawContents clips in *document*
    coordinates and the painter starts every page at the origin.
    """
    reserve = writer.logicalDpiY() * _FOOTER_RESERVE_MM / 25.4
    content_width = writer.width()
    content_height = writer.height() - reserve
    # Lay the document out against the content box (not the full page), so
    # pageCount() and the per-page slices agree on where pages break.
    doc.setPageSize(QSizeF(content_width, content_height))
    page_count = doc.pageCount()
    painter = QPainter(writer)
    try:
        for page in range(page_count):
            if page > 0:
                writer.newPage()
            top = page * content_height
            painter.save()
            # Move this page's slice of the document to the painter origin;
            # drawContents then paints exactly the clipped slice.
            painter.translate(0.0, -top)
            doc.drawContents(
                painter, QRectF(0.0, top, content_width, content_height)
            )
            painter.restore()
            _draw_footer(painter, writer, page + 1, page_count, title)
    finally:
        painter.end()


def _page_rect(writer: QPdfWriter) -> QRectF:
    return QRectF(0, 0, writer.width(), writer.height())


def _draw_footer(painter: QPainter, writer: QPdfWriter, page: int, total: int, title: str):
    font = _default_font()
    font.setPointSizeF(8.0)
    painter.setFont(font)
    painter.setPen(Qt.GlobalColor.gray)
    rect = _page_rect(writer)
    # The footer band sits inside the reserved bottom strip (see
    # _FOOTER_RESERVE_MM), never over content: 8 mm up from the bottom is
    # inside the 18 mm margin band and inside the reserve.
    footer_y = rect.height() - writer.logicalDpiY() * 8.0 / 25.4
    band = QRectF(rect.left(), footer_y, rect.width() / 2, 30)
    painter.drawText(
        band,
        int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
        title,
    )
    painter.drawText(
        QRectF(rect.left() + rect.width() / 2, footer_y, rect.width() / 2 - 20, 30),
        int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
        f"{page} / {total}",
    )


def first_page_renderable(doc: QTextDocument) -> bool:
    """Sanity check used by tests: the document produced real content."""
    return doc.pageCount() > 0 and not doc.isEmpty()


def markdown_to_document(markdown_text: str) -> QTextDocument:
    """Test helper: the same Markdown renderer the PDF path uses."""
    doc = QTextDocument()
    doc.setDefaultFont(_default_font())
    cursor = QTextCursor(doc)
    _write_summary(cursor, markdown_text)
    return doc


def format_duration(seconds) -> str:
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "0:00"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes}:{sec:02d}"
