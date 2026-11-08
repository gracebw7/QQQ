"""PDF download and cache behavior using in-memory HTTP responses."""

import importlib

import httpx
import pytest


@pytest.fixture
def downloader():
    return importlib.import_module("app.documents.download")


@pytest.fixture
def paper(paper_data):
    from app.schemas import Paper

    return Paper(**paper_data)


def download(downloader, paper, cache_dir, handler, **kwargs):
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return downloader.download_paper(
            paper, cache_dir=cache_dir, client=client, **kwargs
        )


def test_document_id_is_stable_safe_and_version_specific(downloader, document_id):
    """Hashing also handles legacy IDs containing slashes."""
    assert downloader.document_id_for("2106.09685v2") == document_id
    assert downloader.document_id_for("2106.09685v1") != document_id
    legacy_id = downloader.document_id_for("hep-th/9901001v1")
    assert len(legacy_id) == 64
    assert all(character in "0123456789abcdef" for character in legacy_id)


def test_download_creates_cache_and_writes_complete_pdf(
    downloader, paper, tmp_path, pdf_bytes, document_id
):
    """A completed download creates exactly one cached PDF."""
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=pdf_bytes, headers={"content-type": "application/pdf"})

    cache = tmp_path / "raw"
    result = download(downloader, paper, cache, respond)
    assert result.download_status == "downloaded"
    assert result.document_id == document_id
    assert result.local_pdf_url == f"/api/documents/{document_id}/pdf"
    assert result.download_error is None
    assert list(cache.iterdir()) == [cache / f"{document_id}.pdf"]
    assert (cache / f"{document_id}.pdf").read_bytes() == pdf_bytes
    assert str(requests[0].url) == str(paper.pdf_url)
    assert requests[0].extensions["timeout"]["read"] <= 30


def test_download_reuses_valid_cache_without_network(
    downloader, paper, tmp_path, pdf_bytes, document_id
):
    """Repeated searches reuse an existing valid PDF."""
    (tmp_path / f"{document_id}.pdf").write_bytes(pdf_bytes)

    def unexpected_request(request):
        pytest.fail("A cache hit must not download again")

    result = download(downloader, paper, tmp_path, unexpected_request)
    assert result.download_status == "cached"
    assert (tmp_path / f"{document_id}.pdf").read_bytes() == pdf_bytes


def test_download_replaces_invalid_cache(downloader, paper, tmp_path, pdf_bytes, document_id):
    """An incomplete cached file must not count as a cache hit."""
    destination = tmp_path / f"{document_id}.pdf"
    destination.write_bytes(b"broken")
    result = download(downloader, paper, tmp_path, lambda request: httpx.Response(200, content=pdf_bytes))
    assert result.download_status == "downloaded"
    assert destination.read_bytes() == pdf_bytes


@pytest.mark.parametrize("status,body", [(200, b"<html>error</html>"), (200, b"%PDF-1.4\nbroken"), (503, b"unavailable")])
def test_download_rejects_invalid_or_failed_response(downloader, paper, tmp_path, status, body):
    """Errors and malformed PDFs leave no cached file."""
    result = download(downloader, paper, tmp_path, lambda request: httpx.Response(status, content=body))
    assert result.download_status == "failed"
    assert result.download_error
    assert result.local_pdf_url is None
    assert not list(tmp_path.iterdir())


class InterruptedPDF(httpx.SyncByteStream):
    def __iter__(self):
        yield b"%PDF-1.4\n"
        raise httpx.ReadTimeout("Download timed out")


def test_download_cleans_up_after_interrupted_stream(downloader, paper, tmp_path):
    """A failure after receiving bytes removes the temporary file."""
    result = download(downloader, paper, tmp_path, lambda request: httpx.Response(200, stream=InterruptedPDF()))
    assert result.download_status == "failed"
    assert result.download_error
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("declared_length", [True, False])
def test_download_enforces_size_limit_with_or_without_header(
    downloader, paper, tmp_path, pdf_bytes, declared_length
):
    """Enforce the byte limit even when Content-Length is absent."""
    def respond(request):
        response = httpx.Response(200, content=pdf_bytes)
        if not declared_length:
            del response.headers["content-length"]
        return response

    result = download(downloader, paper, tmp_path, respond, max_bytes=32)
    assert result.download_status == "failed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("url", ["https://example.com/paper.pdf", "https://arxiv.org.evil.test/paper.pdf"])
def test_download_rejects_non_arxiv_source(downloader, paper, tmp_path, url):
    """Do not use the downloader as an arbitrary URL fetcher."""
    paper = paper.model_copy(update={"pdf_url": url})

    def unexpected_request(request):
        pytest.fail("An unapproved URL must be rejected before requesting it")

    result = download(downloader, paper, tmp_path, unexpected_request)
    assert result.download_status == "failed"
    assert not list(tmp_path.iterdir())


def test_download_rejects_redirect_to_unapproved_host(downloader, paper, tmp_path):
    """Validate redirect targets before following them."""
    requested_urls = []

    def respond(request):
        requested_urls.append(str(request.url))
        if len(requested_urls) > 1:
            pytest.fail("The external redirect must not be followed")
        return httpx.Response(302, headers={"location": "https://example.com/paper.pdf"})

    result = download(downloader, paper, tmp_path, respond)
    assert result.download_status == "failed"
    assert len(requested_urls) == 1
    assert not list(tmp_path.iterdir())
