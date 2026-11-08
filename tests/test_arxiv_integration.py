"""Live arXiv checks, included in the default test run."""

import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx
import pytest


pytestmark = pytest.mark.integration
PAPER_ID = "2106.09685v1"
EXPECTED_TITLE = "LoRA: Low-Rank Adaptation of Large Language Models"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}


@pytest.fixture(autouse=True)
def pace_live_requests():
    # Space live tests to respect arXiv's API request interval.
    time.sleep(3)


def test_live_arxiv_api_returns_known_paper():
    """Verify real connectivity and usable metadata without requiring app code."""
    response = httpx.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": PAPER_ID, "max_results": 1},
        headers={"User-Agent": "QQQ-integration-test/0.1"},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    entries = ET.fromstring(response.content).findall("atom:entry", ATOM)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.findtext("atom:id", "", ATOM).endswith(f"/{PAPER_ID}")
    title = " ".join(entry.findtext("atom:title", "", ATOM).split())
    assert title == EXPECTED_TITLE
    assert entry.findtext("atom:summary", "", ATOM).strip()
    assert entry.findall("atom:author/atom:name", ATOM)
    pdf_links = [
        link.attrib["href"]
        for link in entry.findall("atom:link", ATOM)
        if link.attrib.get("title") == "pdf"
    ]
    assert len(pdf_links) == 1
    assert urlparse(pdf_links[0]).hostname == "arxiv.org"
    assert urlparse(pdf_links[0]).path == f"/pdf/{PAPER_ID}"


def test_live_download_service_saves_readable_pdf_and_reuses_cache(tmp_path, paper_data, monkeypatch, mock_http):
    """Download a real paper, parse its PDF, and reuse it without another request."""
    from app.documents import download as downloader
    import pymupdf

    # A known URL isolates downloading from the search service.
    paper = {
        **paper_data,
        "arxiv_id": PAPER_ID,
        "document_id": PAPER_ID,
        "pdf_url": f"https://arxiv.org/pdf/{PAPER_ID}",
    }
    cache_dir = tmp_path / "raw"
    monkeypatch.setattr(downloader, "CACHE_DIR", cache_dir)
    result = downloader.download_paper(paper)

    assert result["status"] == "downloaded", result["error"]
    assert result["error"] is None
    pdf_path = cache_dir / f"{PAPER_ID}.pdf"
    downloaded_bytes = pdf_path.read_bytes()
    assert downloaded_bytes.startswith(b"%PDF-")
    with pymupdf.open(pdf_path) as document:
        assert document.page_count > 0
        assert "lora" in document[0].get_text().lower()
    assert list(cache_dir.iterdir()) == [pdf_path]

    def reject_second_download(request):
        pytest.fail("The cached paper must not be downloaded again")

    mock_http(reject_second_download)
    cached = downloader.download_paper(paper)
    assert cached == {"status": "cached", "error": None}
    assert pdf_path.read_bytes() == downloaded_bytes
