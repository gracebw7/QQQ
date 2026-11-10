"""PDF -> Markdown conversion and page/line mappings."""

import pymupdf
from markdown_it import MarkdownIt
import pytest

from app.documents import convert, download


PAGE_TEXT = {1: "Alpha introduction paragraph.", 3: "Gamma conclusion paragraph."}


@pytest.fixture
def cached_pdf(monkeypatch, tmp_path, paper_data):
    monkeypatch.setattr(download, "CACHE_DIR", tmp_path)
    document = pymupdf.open()
    for page_num in (1, 2, 3):
        page = document.new_page()
        if page_num in PAGE_TEXT:
            page.insert_text((72, 72), f"Heading {page_num}", fontsize=20)
            page.insert_text((72, 120), PAGE_TEXT[page_num])
    document.save(tmp_path / f"{paper_data['document_id']}.pdf")
    document.close()
    return paper_data


def test_pdf_to_markdown_maps_pages_to_lines(cached_pdf):
    result = convert.pdf_to_markdown(cached_pdf)
    lines = result.markdown.split("\n")
    assert result.document_id == cached_pdf["document_id"]
    assert [p.page_num for p in result.page_lines] == [1, 3]
    for page in result.page_lines:
        page_text = "\n".join(lines[page.start_line - 1:page.end_line])
        assert PAGE_TEXT[page.page_num] in page_text
        other = PAGE_TEXT[4 - page.page_num]
        assert other not in page_text
    first, second = result.page_lines
    assert first.end_line < second.start_line


def test_paragraphs_fall_within_one_page_and_pass_ingestion_checks(cached_pdf):
    # Same paragraph parsing as vec_func.markdown_parse (commonmark, 1-based lines).
    result = convert.pdf_to_markdown(cached_pdf)
    paragraphs = [t.map for t in MarkdownIt("commonmark").parse(result.markdown)
                  if t.type == "paragraph_open"]
    assert paragraphs
    for start, end in paragraphs:
        owners = [p for p in result.page_lines if p.start_line <= start + 1 and end <= p.end_line]
        assert len(owners) == 1
    line_count = len(result.markdown.split("\n")) - int(result.markdown.endswith("\n"))
    assert result.page_lines[-1].end_line == line_count


def test_markdown_with_page_lines_normalizes_and_skips_empty_pages():
    markdown, pages = convert.markdown_with_page_lines([
        (1, "# Title \r\n\r\nBody one. \r\n"),
        (2, "  \n\n"),
        (3, "\n\nBody three.\nMore. \n\n"),
    ])
    assert markdown == "# Title\n\nBody one.\n\nBody three.\nMore.\n"
    assert [p.model_dump() for p in pages] == [
        dict(page_num=1, start_line=1, end_line=3),
        dict(page_num=3, start_line=5, end_line=6),
    ]


@pytest.mark.parametrize("pages, expected", [
    ([], ("", [])),
    ([(1, "   \n")], ("", [])),
])
def test_markdown_with_page_lines_empty(pages, expected):
    assert convert.markdown_with_page_lines(pages) == expected


def test_single_page():
    markdown, pages = convert.markdown_with_page_lines([(1, "Only line.")])
    assert markdown == "Only line.\n"
    assert [(p.page_num, p.start_line, p.end_line) for p in pages] == [(1, 1, 1)]


def test_missing_pdf_raises(monkeypatch, tmp_path, paper_data):
    monkeypatch.setattr(download, "CACHE_DIR", tmp_path)
    with pytest.raises(ValueError, match="download_paper"):
        convert.pdf_to_markdown(paper_data)


def test_invalid_pdf_raises(monkeypatch, tmp_path, paper_data):
    monkeypatch.setattr(download, "CACHE_DIR", tmp_path)
    (tmp_path / f"{paper_data['document_id']}.pdf").write_bytes(b"%PDF-broken")
    with pytest.raises(ValueError):
        convert.pdf_to_markdown(paper_data)


def test_invalid_document_id_raises(paper_data):
    with pytest.raises(ValueError, match="Invalid document ID"):
        convert.pdf_to_markdown({**paper_data, "document_id": "../etc/passwd"})
