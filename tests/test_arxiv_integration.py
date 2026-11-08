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


def test_live_search_service_returns_known_paper():
    """Verify the application's search function against real arXiv results."""
    from app.search.arxiv_client import search_papers

    papers = search_papers(f'id:{PAPER_ID}', 1)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == PAPER_ID
    assert " ".join(paper.title.split()) == EXPECTED_TITLE
    assert paper.authors
    assert paper.abstract.strip()
    assert urlparse(str(paper.abstract_url)).hostname == "arxiv.org"
    assert urlparse(str(paper.pdf_url)).path == f"/pdf/{PAPER_ID}"


def test_live_download_service_saves_readable_pdf_and_reuses_cache(tmp_path, paper_data):
    """Download a real paper, parse its PDF, and reuse it without another request."""
    from app.documents.download import document_id_for, download_paper
    from app.schemas import Paper
    import pymupdf

    # A known URL isolates downloading from the search service.
    paper = Paper(**{
        **paper_data,
        "arxiv_id": PAPER_ID,
        "abstract_url": f"https://arxiv.org/abs/{PAPER_ID}",
        "pdf_url": f"https://arxiv.org/pdf/{PAPER_ID}",
    })
    cache_dir = tmp_path / "raw"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        result = download_paper(paper, cache_dir=cache_dir, client=client)

    assert result.download_status == "downloaded", result.download_error
    assert result.download_error is None
    assert result.document_id == document_id_for(PAPER_ID)
    assert result.local_pdf_url == f"/api/documents/{result.document_id}/pdf"
    pdf_path = cache_dir / f"{result.document_id}.pdf"
    downloaded_bytes = pdf_path.read_bytes()
    assert downloaded_bytes.startswith(b"%PDF-")
    with pymupdf.open(pdf_path) as document:
        assert document.page_count > 0
        assert "lora" in document[0].get_text().lower()
    assert list(cache_dir.iterdir()) == [pdf_path]

    def reject_second_download(request):
        pytest.fail("The cached paper must not be downloaded again")

    with httpx.Client(transport=httpx.MockTransport(reject_second_download)) as client:
        cached = download_paper(paper, cache_dir=cache_dir, client=client)
    assert cached.download_status == "cached"
    assert cached.document_id == result.document_id
    assert cached.local_pdf_url == result.local_pdf_url
    assert pdf_path.read_bytes() == downloaded_bytes
