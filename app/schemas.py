from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)
    @field_validator("query")
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be empty")
        return value

class PaperResult(BaseModel):
    # Internal ID for stored pdf in data/raw/{id}.pdf
    document_id: str
    # Provided arXiv ID (can be used to find abstract liek:  https://arxiv.org/abs/{id})
    arxiv_id: str

    # Paper metadata
    title: str
    authors: list[str]
    abstract: str
    published: datetime

    # Source link to download from
    pdf_url: str

    # Download information
    download_status: Literal["downloaded", "cached", "failed"]
    download_error: str | None = None


class SearchResponse(BaseModel):
    query: str
    papers: list[PaperResult]

