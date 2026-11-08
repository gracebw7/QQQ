"""User-facing search behavior in Streamlit's simulated app."""

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    path = Path(__file__).resolve().parents[1] / "main.py"
    return AppTest.from_file(str(path), default_timeout=10)


def submit(ui, query="low-rank adaptation"):
    ui.text_input(key="query").set_value(query)
    ui.button(key="search_submit").click().run()
    assert not ui.exception


def visible_text(ui):
    elements = ("markdown", "text", "caption", "title", "header", "subheader", "error", "warning", "info")
    return "\n".join(str(element.value) for kind in elements for element in ui.get(kind))


def link_urls(ui):
    return {element.proto.url for element in ui.get("link_button")}


def test_ui_renders_search_controls_without_requesting_data(ui, mock_http):
    """Opening the app must not start a search."""
    requests = mock_http(lambda request: pytest.fail("Search ran before submission"))
    ui.run()
    assert not ui.exception
    assert ui.text_input(key="query").label == "Search arXiv"
    assert ui.selectbox(key="max_results").value == 3
    assert ui.selectbox(key="max_results").options == ["1", "2", "3", "4", "5"]
    assert ui.button(key="search_submit").label == "Search"
    assert requests == []


def test_ui_submits_selected_limit_and_displays_results(ui, mock_http, response_data):
    """The UI sends the selected controls and displays returned paper details."""
    requests = mock_http(lambda request: httpx.Response(200, json=response_data))
    ui.run()
    ui.selectbox(key="max_results").select(2)
    submit(ui)
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:8000/api/search"
    assert json.loads(request.content) == {"query": "low-rank adaptation", "max_results": 2}
    assert request.extensions["timeout"]["read"] == 240
    paper = response_data["papers"][0]
    text = visible_text(ui)
    assert paper["title"] in text
    assert all(author in text for author in paper["authors"])
    assert paper["abstract"] in text
    assert len(ui.expander) >= 1
    assert paper["abstract_url"] in link_urls(ui)
    assert "http://127.0.0.1:8000" + paper["local_pdf_url"] in link_urls(ui)


def test_ui_rerender_keeps_results_without_repeating_search(ui, mock_http, response_data):
    """Session state prevents duplicate searches after ordinary rerenders."""
    requests = mock_http(lambda request: httpx.Response(200, json=response_data))
    ui.run()
    submit(ui)
    ui.run()
    assert not ui.exception
    assert len(requests) == 1
    assert response_data["papers"][0]["title"] in visible_text(ui)


def test_ui_blank_input_shows_validation_without_request(ui, mock_http):
    """Give immediate feedback before contacting the backend."""
    requests = mock_http(lambda request: pytest.fail("Blank input reached the backend"))
    ui.run()
    submit(ui, "   ")
    assert len(ui.error) + len(ui.warning) > 0
    assert requests == []


def test_ui_empty_results_shows_no_results_message(ui, mock_http):
    """A successful empty search needs an explicit message."""
    mock_http(lambda request: httpx.Response(200, json={"query": "LoRA", "papers": []}))
    ui.run()
    submit(ui, "LoRA")
    assert "no papers found" in visible_text(ui).lower()


def test_ui_failed_download_keeps_metadata_and_source_link(ui, mock_http, response_data):
    """Users can still access the source when local downloading fails."""
    paper = response_data["papers"][0]
    paper.update(download_status="failed", download_error="Download timed out", local_pdf_url=None)
    mock_http(lambda request: httpx.Response(200, json=response_data))
    ui.run()
    submit(ui)
    assert paper["title"] in visible_text(ui)
    assert "Download timed out" in visible_text(ui)
    assert paper["pdf_url"] in link_urls(ui)
    assert not any("/api/documents/" in url for url in link_urls(ui))


@pytest.mark.parametrize("failure", ["connection", "timeout", "upstream"])
def test_ui_shows_backend_failure_without_crashing(ui, mock_http, failure):
    """Connection errors, timeouts, and failed searches are visible errors."""
    def respond(request):
        if failure == "connection":
            raise httpx.ConnectError("Backend unavailable", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("Backend timed out", request=request)
        return httpx.Response(502, json={"detail": "arXiv is unavailable"})

    mock_http(respond)
    ui.run()
    submit(ui)
    assert len(ui.error) > 0


def test_ui_uses_configured_backend_url(ui, monkeypatch, mock_http, response_data):
    """API_BASE_URL controls both search requests and cached PDF links."""
    monkeypatch.setenv("API_BASE_URL", "http://localhost:9000")
    requests = mock_http(lambda request: httpx.Response(200, json=response_data))
    ui.run()
    submit(ui)
    assert str(requests[0].url) == "http://localhost:9000/api/search"
    assert "http://localhost:9000" + response_data["papers"][0]["local_pdf_url"] in link_urls(ui)
