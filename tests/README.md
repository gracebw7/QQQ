# Tests

Run from `QQQ/` with the project dependencies installed:

Live arXiv tests run by default and require internet access. Tests for unfinished
application code will fail until that code is implemented.

```powershell
# Offline unit tests only
.venv/Scripts/python.exe -m pytest -m "not integration"

# Live integration tests only
.venv/Scripts/python.exe -m pytest -m integration

# One test file
.venv/Scripts/python.exe -m pytest tests/test_schemas.py
```

## Files

| File | Coverage |
|---|---|
| `test_schemas.py` | Input validation, result limits, and response models. |
| `test_arxiv_client.py` | Mocked searches, metadata, deduplication, and upstream errors. |
| `test_download.py` | Mocked PDF downloads, caching, size limits, failures, and URL validation. |
| `test_api.py` | Health, search, and cached-PDF endpoints, including error responses. |
| `test_ui.py` | Search controls, results, errors, links, and session persistence. |
| `test_arxiv_integration.py` | Real arXiv API calls, application search, PDF downloading, and cache reuse. |
| `conftest.py` | Shared fixtures and network blocking for unit tests. |
| `fixtures/paper.json` | Sample paper metadata for offline tests. |
| `requirements.txt` | Additional test dependencies. |
