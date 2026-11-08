"""Search behavior without contacting arXiv."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import arxiv
import pytest


@pytest.fixture
def search_client(monkeypatch, paper_data):
    result = SimpleNamespace(
        entry_id=f"https://arxiv.org/abs/{paper_data["arxiv_id"]}",
        published=datetime.fromisoformat(paper_data["published"]),
        title=paper_data["title"],
        authors=[SimpleNamespace(name=name) for name in paper_data["authors"]],
        summary=paper_data["abstract"],
        pdf_url=paper_data["pdf_url"],
        get_short_id=lambda: paper_data["arxiv_id"],
    )
    client = Mock()
    client.results.return_value = iter([result])
    from app.search import arxiv_client
    monkeypatch.setattr(arxiv_client, "client", client)
    return client, result


def test_search_passes_query_limit_and_relevance_sort(search_client):
    """The user's requested search reaches the client unchanged."""
    from app.search.arxiv_client import search_papers

    client, _ = search_client
    search_papers("low-rank adaptation", 3)
    search = client.results.call_args.args[0]
    assert search.query == "low-rank adaptation"
    assert search.max_results == 3
    assert search.sort_by == arxiv.SortCriterion.Relevance


def test_search_normalizes_metadata_and_preserves_version(search_client, paper_data):
    """Retain source URLs and the exact paper version."""
    from app.search.arxiv_client import search_papers

    papers = search_papers("LoRA", 3)
    assert len(papers) == 1
    assert papers[0] == paper_data


def test_search_removes_duplicate_papers(search_client):
    """Repeated entries should not cause repeated downloads."""
    from app.search.arxiv_client import search_papers

    client, result = search_client
    client.results.return_value = iter([result, result])
    assert len(search_papers("LoRA", 3)) == 1


@pytest.mark.parametrize("arxiv_id,expected_id", [
    ("2106.09685v1", "2106.09685v1"),
    ("2106.09685v2", "2106.09685v2"),
    ("hep-th/9901001v1", "hep-th_9901001v1"),
])
def test_search_creates_stable_versioned_filename_ids(search_client, arxiv_id, expected_id):
    """Paper versions remain distinct and legacy slashes are filename-safe."""
    from app.search.arxiv_client import search_papers

    client, result = search_client
    result.entry_id = f"https://arxiv.org/abs/{arxiv_id}"
    client.results.return_value = [result]
    first = search_papers("LoRA", 1)[0]
    second = search_papers("LoRA", 1)[0]
    assert first["arxiv_id"] == arxiv_id
    assert first["document_id"] == second["document_id"] == expected_id


def test_search_returns_empty_list_when_no_papers_match(search_client):
    """An empty feed is not an upstream failure."""
    from app.search.arxiv_client import search_papers

    client, _ = search_client
    client.results.return_value = iter([])
    assert search_papers("unknown topic", 3) == []


def test_search_reports_upstream_failure(search_client):
    """Give the API a predictable error to translate to HTTP 502."""
    from app.search.arxiv_client import search_papers

    client, _ = search_client
    client.results.side_effect = arxiv.HTTPError("https://export.arxiv.org/api/query", 1, 503)
    with pytest.raises(arxiv.HTTPError):
        search_papers("LoRA", 3)
