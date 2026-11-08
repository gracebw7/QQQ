"""Validation contracts for API inputs and paper results."""

import pytest
from pydantic import ValidationError


def test_search_request_trims_query_and_defaults_to_three_results():
    """Normalize useful input before it reaches arXiv."""
    from app.schemas import SearchRequest

    request = SearchRequest(query="  low-rank adaptation  ")
    assert request.query == "low-rank adaptation"
    assert request.max_results == 3


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_search_request_rejects_blank_query(query):
    """Whitespace alone is not a search."""
    from app.schemas import SearchRequest

    with pytest.raises(ValidationError):
        SearchRequest(query=query)


@pytest.mark.parametrize("count", [0, -1, 6])
def test_search_request_rejects_result_count_outside_limits(count):
    """Keep automatic downloads within the agreed limit."""
    from app.schemas import SearchRequest

    with pytest.raises(ValidationError):
        SearchRequest(query="LoRA", max_results=count)


@pytest.mark.parametrize("count", [1, 5])
def test_search_request_accepts_result_count_boundaries(count):
    """Both ends of the supported range are valid."""
    from app.schemas import SearchRequest

    assert SearchRequest(query="LoRA", max_results=count).max_results == count


@pytest.mark.parametrize("status", ["downloaded", "cached", "failed"])
def test_paper_result_accepts_download_outcomes(status, response_data):
    """Successful and failed papers share one response shape."""
    from app.schemas import PaperResult

    data = response_data["papers"][0]
    data["download_status"] = status
    if status == "failed":
        data.update(local_pdf_url=None, download_error="Download timed out")
    result = PaperResult(**data)
    assert result.model_dump(mode="json") == data


def test_paper_result_rejects_unknown_download_status(response_data):
    """Catch misspelled or unsupported statuses."""
    from app.schemas import PaperResult

    data = response_data["papers"][0]
    data["download_status"] = "maybe"
    with pytest.raises(ValidationError):
        PaperResult(**data)


def test_search_response_accepts_empty_results():
    """No matches is a normal response."""
    from app.schemas import SearchResponse

    result = SearchResponse(query="unknown topic", papers=[])
    assert result.papers == []
