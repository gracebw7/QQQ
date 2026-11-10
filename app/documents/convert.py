"""Convert a downloaded arXiv PDF to Markdown with PDF page -> Markdown line mappings."""

import re
from dataclasses import dataclass

import pymupdf
import pymupdf4llm

from app.documents import download
from app.vector_schemas import PageLineRange


@dataclass
class MarkdownDocument:
    document_id: str
    markdown: str
    page_lines: list[PageLineRange]


def _clean_page(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def markdown_with_page_lines(
    pages: list[tuple[int, str]],
) -> tuple[str, list[PageLineRange]]:
    """Join per-page Markdown and record the 1-based line range of each page.

    Line numbers match vec_func.parse_and_save: only "\\n" breaks lines and the
    trailing newline does not add a line. Pages are separated by one blank line
    so paragraphs never merge across a page boundary. Empty pages get no mapping.
    """
    blocks: list[str] = []
    page_lines: list[PageLineRange] = []
    current_line = 0
    for page_num, text in pages:
        block = _clean_page(text)
        if not block.strip():
            continue
        if blocks:
            current_line += 1  # blank separator line
        start_line = current_line + 1
        end_line = start_line + block.count("\n")
        page_lines.append(PageLineRange(page_num=page_num, start_line=start_line, end_line=end_line))
        blocks.append(block)
        current_line = end_line
    markdown = "\n\n".join(blocks)
    return (markdown + "\n" if markdown else ""), page_lines


def pdf_to_markdown(paper: dict) -> MarkdownDocument:
    """Convert a paper's cached PDF (from download_paper) to Markdown with page mappings."""
    document_id = paper["document_id"]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", document_id):
        raise ValueError("Invalid document ID")

    path = download.CACHE_DIR / f"{document_id}.pdf"
    if not download.valid_pdf(path):
        raise ValueError(f"No valid cached PDF for {document_id}; run download_paper first")

    with pymupdf.open(path) as document:
        chunks = pymupdf4llm.to_markdown(
            document, page_chunks=True, show_progress=False, use_ocr=False
        )

    pages = [(chunk["metadata"]["page_number"], chunk["text"]) for chunk in chunks]
    markdown, page_lines = markdown_with_page_lines(pages)
    return MarkdownDocument(document_id=document_id, markdown=markdown, page_lines=page_lines)
