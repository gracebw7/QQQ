
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.schemas import SearchRequest, SearchResponse, PaperResult
from app.search.arxiv_client import search_papers
from app.documents.download import (
    download_paper,
    CACHE_DIR,
    valid_pdf,
)

app = FastAPI(title="QQQ Research API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest):

    try:
        papers = search_papers(request.query, request.max_results)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to search arXiv right now."
        )

    results = []

    for paper in papers:
        download = download_paper(paper)

        status = download["status"]

        local_url = None

        if status in ("downloaded", "cached"):
            local_url = (
                f"/api/documents/{paper['document_id']}/pdf"
            )

        results.append(PaperResult(
            **paper,
            download_status=status,
            local_pdf_url=local_url,
            download_error=download["error"],
        ))

    return SearchResponse(
        query=request.query,
        papers=results,
    )


@app.get("/api/documents/{document_id}/pdf")
def get_pdf(document_id: str):

    if not re.fullmatch(r"[A-Za-z0-9._-]+", document_id):
        raise HTTPException(status_code=404)

    pdf_path = CACHE_DIR / f"{document_id}.pdf"

    if not valid_pdf(pdf_path):
        raise HTTPException(status_code=404)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        content_disposition_type="inline",
    )
