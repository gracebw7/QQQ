"""Shared fixtures; only integration tests may contact live services."""

import json
import urllib.request
from pathlib import Path

import httpx
import pytest
import requests


@pytest.fixture(autouse=True)
def block_network(monkeypatch, request):
    """Block real HTTP requests in unit tests, but allow integration tests."""
    if request.node.get_closest_marker("integration"):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Real network access is disabled in tests")

    # Leave local sockets available for Streamlit and Windows event loops.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", blocked)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", blocked)


@pytest.fixture
def paper_data():
    path = Path(__file__).parent / "fixtures" / "paper.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def document_id(paper_data):
    return paper_data["document_id"]


@pytest.fixture
def response_data(paper_data, document_id):
    return {
        "query": "low-rank adaptation",
        "papers": [{
            **paper_data,
            "published": paper_data["published"].replace("+00:00", "Z"),
            "document_id": document_id,
            "download_status": "downloaded",
            "download_error": None,
        }],
    }


@pytest.fixture
def pdf_bytes():
    """Build a valid one-page PDF with offsets calculated from its bytes."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_offset = len(content)
    content.extend(b"xref\n0 4\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(content)


@pytest.fixture
def mock_http(monkeypatch):
    """Replace HTTPX's network transport while preserving request handling."""
    requests = []

    def install(handler):
        def handle_request(transport, request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
        return requests

    return install
