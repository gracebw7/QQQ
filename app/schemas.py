from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
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


class RouterStatus(str, Enum):
    DISTILLED = "distilled"
    NEEDS_REFRAME = "needs_reframe"


class RouterDecision(BaseModel):
    """Output of the single-question router ("distill question or ask user to reframe")."""

    status: RouterStatus
    distilled_question: str | None = Field(
        None,
        description="One self-contained question. Required when status == distilled.",
    )
    reframe_message: str | None = Field(
        None,
        description="Message shown to the user. Required when status == needs_reframe.",
    )
    reason: str = Field("", description="Short internal rationale (logged, not shown).")

    @field_validator("distilled_question", "reframe_message")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        v = v.strip() if v else v
        return v or None

    def model_post_init(self, _ctx) -> None:
        if self.status is RouterStatus.DISTILLED and not self.distilled_question:
            raise ValueError("distilled_question required when status == distilled")
        if self.status is RouterStatus.NEEDS_REFRAME and not self.reframe_message:
            raise ValueError("reframe_message required when status == needs_reframe")


class SearchQuery(BaseModel):
    """One arXiv search pass produced by the search planner."""

    search_term: str = Field(description="Short keyword phrase a human would type.")
    arxiv_query: str = Field(description="Valid arXiv `search_query` string.")
    max_results: int = Field(3, ge=1, le=20, description="Papers to fetch for this pass.")

    @field_validator("search_term", "arxiv_query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class SearchPlan(BaseModel):
    """Search planner output: the distilled question and 1+ arXiv queries."""

    distilled_question: str = Field(
        "", description="Filled in by the application; the model may leave it empty."
    )
    queries: list[SearchQuery] = Field(min_length=1)
