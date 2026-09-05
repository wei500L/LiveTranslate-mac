"""PDF export: file generation, text searchability, safety of names and content.

The PDF pipeline is Qt-based (QTextDocument + QPdfWriter), so these tests run
offscreen against a real QApplication — no network, no model.
"""

import re

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="pdf_exporter needs Qt")

import pdf_exporter as pe


@pytest.fixture(scope="module")
def app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _decode_pdf_text(path):
    """Extract the ToUnicode-mapped characters a PDF viewer would search."""
    data = path.read_bytes()
    chars = set()
    for match in re.finditer(rb"beginbfrange(.*?)endbfrange", data, re.S):
        for _, _, high in re.findall(
            rb"<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4})>",
            match.group(1),
        ):
            chars.add(chr(int(high, 16)))
    for match in re.finditer(rb"beginbfchar(.*?)endbfchar", data, re.S):
        for _, mapped in re.findall(
            rb"<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4,8})>", match.group(1)
        ):
            try:
                chars.add(chr(int(mapped, 16)))
            except ValueError:
                pass
    return "".join(sorted(chars))


def test_pdf_file_is_generated_with_expected_sections(tmp_path, app):
    out = tmp_path / "minutes.pdf"
    entries = [
        {"timestamp": "09:15:00", "original": "Здравствуйте", "translation": "大家好"},
        {"timestamp": "09:16:01", "original": "second", "translation": None},
    ]
    ok = pe.export_pdf(
        out,
        title="俄语课",
        meta_rows=[("日期", "2026-09-03"), ("时长", "45:12")],
        summary_markdown="# 会议纪要\n\n## 核心内容\n\n- **预算** 12000 元\n",
        entries=entries,
    )
    assert ok and out.exists() and out.stat().st_size > 1000
    assert out.read_bytes().startswith(b"%PDF")


def test_long_record_paginates_at_real_font_size(tmp_path, app):
    """[round-15 regression] The document layout must be bound to the
    QPdfWriter before paginating. Without ``setPaintDevice`` the layout
    interpreted point sizes at ~96 DPI while the page box was in 1200-DPI
    device pixels, so every font rendered ~12x too small and a whole
    meeting collapsed onto a single page (160 bilingual entries -> 1 page).
    A long record must produce multiple pages whose first page is not
    re-painted on the later ones."""
    pytest.importorskip("PyQt6.QtPdf", reason="pagination check reads the PDF back")
    from PyQt6.QtPdf import QPdfDocument

    out = tmp_path / "long.pdf"
    entries = [
        {
            "timestamp": f"{9 + i // 60:02d}:{i % 60:02d}:00",
            "original": f"Запись {i} по повестке дня.",
            "translation": f"第{i}条记录：语音识别系统的性能优化与权衡。",
        }
        for i in range(160)
    ]
    assert pe.export_pdf(
        out, title="长会议", meta_rows=[("条目", "160")],
        summary_markdown=None, entries=entries, show_original=True,
    )
    doc = QPdfDocument(None)
    doc.load(str(out))
    pages = doc.pageCount()
    texts = [doc.getAllText(p).text() for p in range(pages)]
    assert pages > 1, "160 entries collapsed onto a single page (shrink bug)"
    # The first entry lives on page 1 only — the per-page slice must not
    # repaint the top of the document everywhere.
    hits = sum(1 for t in texts if "第0条记录" in t)
    assert hits == 1, f"first entry painted on {hits} pages"
    assert texts[0] != texts[1]


def test_pdf_text_is_searchable_in_chinese_and_russian(tmp_path, app):
    out = tmp_path / "bilingual.pdf"
    ok = pe.export_pdf(
        out,
        title="双语测试",
        meta_rows=[],
        summary_markdown=None,
        entries=[
            {"timestamp": "09:00:01", "original": "Привет всем коллегам",
             "translation": "大家好，同事们"},
            {"timestamp": "09:00:02", "original": "Число сорок два", "translation": "四十二"},
        ],
    )
    assert ok
    text = _decode_pdf_text(out)
    assert any(0x4E00 <= ord(c) <= 0x9FFF for c in text), "no CJK glyphs mapped"
    assert any(0x0400 <= ord(c) <= 0x04FF for c in text), "no Cyrillic glyphs mapped"


def test_pdf_without_summary_and_without_entries_still_works(tmp_path, app):
    out = tmp_path / "empty.pdf"
    assert pe.export_pdf(out, title="空记录", meta_rows=[("日期", "2026-01-01")],
                         summary_markdown=None, entries=None)


def test_pdf_contains_no_credentials(tmp_path, app):
    out = tmp_path / "safe.pdf"
    pe.export_pdf(
        out,
        title="会议",
        meta_rows=[("日期", "2026-09-03")],
        summary_markdown="# 纪要\n内容",
        entries=[{"timestamp": "09:00:00", "original": "text", "translation": "文本"}],
    )
    data = out.read_bytes()
    for secret in (b"api", b"sk-", b"http://127.0.0.1", b"livetrans_"):
        assert secret not in data.lower(), secret


def test_export_failure_returns_false_not_a_crash(tmp_path, app):
    # A directory as target makes the writer fail; export must return False.
    bad = tmp_path / "unwritable.pdf"
    bad.mkdir()
    assert pe.export_pdf(bad, title="x", meta_rows=[], summary_markdown=None) is False


def test_safe_filename_strips_path_separators_and_colons():
    name = pe.safe_filename("俄语课/第一讲: 会议?", "2026-09-03")
    assert "/" not in name and ":" not in name
    assert name.endswith(".pdf")
    assert name == pe.safe_filename("俄语课/第一讲: 会议?", "2026-09-03")
    assert pe.safe_filename("", "2026-09-03").startswith("meeting")


def test_format_duration():
    assert pe.format_duration(0) == "0:00"
    assert pe.format_duration(65) == "1:05"
    assert pe.format_duration(3725) == "1:02:05"
    assert pe.format_duration(None) == "0:00"


def test_markdown_to_document_renders_headings_and_bullets(app):
    doc = pe.markdown_to_document("# 标题\n\n## 小节\n\n- 条目一\n- **粗体**条目\n")
    text = doc.toPlainText()
    assert "标题" in text and "小节" in text
    assert "条目一" in text
    assert "•" in text
