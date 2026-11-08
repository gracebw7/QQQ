"""HTTP contracts with search and download services replaced by fakes."""

import hashlib
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_setup(monkeypatch, tmp_path, paper_data, response_data):
    from app import api
    from app.schemas import Paper, PaperResult

    search = Mock(return_value=[Paper(**paper_data)])
    download = Mock(return_value=PaperResult(**response_data["papers"][0]))
    monkeypatch.setattr(api, "search_papers", search)
    monkeypatch.setattr(api, "download_paper", download)
    monkeypatch.setattr(api, "CACHE_DIR", tmp_path)
    with TestClient(api.app) as client:
        yield client, search, download


def test_health_does_not_call_external_services(api_setup):
    """Health checks stay fast and independent of arXiv."""
    client, search, download = api_setup
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    search.assert_not_called()
    download.assert_not_called()


def test_search_returns_metadata_and_downloaded_pdf_link(api_setup, response_data, tmp_path):
    """Search combines normalized metadata with a completed download."""
    client, search, download = api_setup
    response = client.post("/api/search", json={"query": "low-rank adaptation", "max_results": 3})
    assert response.status_code == 200
    assert response.json() == response_data
    assert str(tmp_path) not in response.text
    search.assert_called_once_with("low-rank adaptation", 3)
    download.assert_called_once()
    assert download.call_args.args[0] == search.return_value[0]
    assert download.call_args.kwargs["cache_dir"] == tmp_path


@pytest.mark.parametrize("payload", [{}, {"query": " "}, {"query": "LoRA", "max_results": 0}, {"query": "LoRA", "max_results": 6}])
def test_search_rejects_invalid_input_before_calling_services(api_setup, payload):
    """Invalid input must not trigger search or downloads."""
    client, search, download = api_setup
    assert client.post("/api/search", json=payload).status_code == 422
    search.assert_not_called()
    download.assert_not_called()


def test_search_returns_empty_results_without_downloading(api_setup):
    """No matches is a successful empty response."""
    client, search, download = api_setup
    search.return_value = []
    response = client.post("/api/search", json={"query": "unknown topic"})
    assert response.status_code == 200
    assert response.json()["papers"] == []
    download.assert_not_called()


@pytest.mark.parametrize("all_failed", [False, True])
def test_search_attempts_every_download_and_preserves_failures(api_setup, response_data, all_failed):
    """A failed PDF does not remove its metadata or stop the batch."""
    from app.schemas import PaperResult

    client, search, download = api_setup
    first = search.return_value[0]
    second_id = "2106.09685v1"
    second = first.model_copy(update={
        "arxiv_id": second_id,
        "abstract_url": f"https://arxiv.org/abs/{second_id}",
        "pdf_url": f"https://arxiv.org/pdf/{second_id}",
    })
    search.return_value = [first, second]
    failed = {
        **response_data["papers"][0],
        "download_status": "failed",
        "local_pdf_url": None,
        "download_error": "Download timed out",
    }
    second_document_id = hashlib.sha256(second_id.encode()).hexdigest()
    second_result = {
        **failed,
        **second.model_dump(mode="json"),
        "document_id": second_document_id,
    }
    if not all_failed:
        second_result.update(
            download_status="downloaded",
            download_error=None,
            local_pdf_url=f"/api/documents/{second_document_id}/pdf",
        )
    results = [PaperResult(**failed), PaperResult(**second_result)]
    download.side_effect = results
    response = client.post("/api/search", json={"query": "LoRA"})
    assert response.status_code == 200
    assert response.json()["papers"] == [result.model_dump(mode="json") for result in results]
    assert [call.args[0] for call in download.call_args_list] == [first, second]


def test_search_translates_upstream_failure_to_502(api_setup):
    """An arXiv outage must not look like an empty search."""
    from app.search.arxiv_client import ArxivSearchError

    client, search, download = api_setup
    search.side_effect = ArxivSearchError("arXiv is unavailable")
    response = client.post("/api/search", json={"query": "LoRA"})
    assert response.status_code == 502
    assert response.json()["detail"]
    download.assert_not_called()


def test_pdf_endpoint_serves_cached_bytes(api_setup, tmp_path, pdf_bytes, document_id):
    """The viewer receives the cached PDF, not a filesystem path."""
    client, search, download = api_setup
    (tmp_path / f"{document_id}.pdf").write_bytes(pdf_bytes)
    response = client.get(f"/api/documents/{document_id}/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content == pdf_bytes
    search.assert_not_called()
    download.assert_not_called()


@pytest.mark.parametrize("identifier", ["a" * 64, "invalid", "..%5Coutside", "%2e%2e%2foutside"])
def test_pdf_endpoint_rejects_missing_or_unsafe_paths(api_setup, identifier, tmp_path, pdf_bytes):
    """Missing IDs and traversal attempts cannot expose other files."""
    client, _, _ = api_setup
    (tmp_path.parent / "outside.pdf").write_bytes(pdf_bytes)
    assert client.get(f"/api/documents/{identifier}/pdf").status_code == 404
