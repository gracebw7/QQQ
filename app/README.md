## API Documentation

Start the FastAPI backend:

```bash
uv run uvicorn app.api:app --reload
```

API documentation: http://127.0.0.1:8000/docs

### Endpoints

| Endpoint                               | Description                    |
| -------------------------------------- | ------------------------------ |
| `GET /health`                          | Check backend status           |
| `POST /api/search`                     | Search arXiv and download PDFs |
| `GET /api/documents/{document_id}/pdf` | Retrieve a cached PDF          |

### Example Requests

**Health check**

```bash
curl http://127.0.0.1:8000/health
```

**Search arXiv**

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"machine learning","max_results":5}'
```

`query` is required. `max_results` defaults to 5 (allowed: 1–20).

**Retrieve PDF**

```bash
curl http://127.0.0.1:8000/api/documents/1706.03762v7/pdf -o paper.pdf
```

Replace the ID with a `document_id` returned by the search endpoint.

Downloaded PDFs are stored in `data/raw/`. Each search result includes a download status: `downloaded`, `cached`, or `failed`.
