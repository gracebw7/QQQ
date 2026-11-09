"""Validation for document line references, embeddings, and PDF page mappings."""

import pytest
from pydantic import ValidationError

from app.vector_schemas import ChunkCreate, PageLineCreate


@pytest.mark.parametrize("model", [ChunkCreate, PageLineCreate])
@pytest.mark.parametrize("changes", [
    {"start_line": 0}, {"end_line": 0}, {"start_line": 3, "end_line": 2},
    {"start_line": None}, {"end_line": None},
])
def test_invalid_line_ranges(model, changes):
    data = dict(document_id=1, start_line=1, end_line=2)
    data.update(content="Text.", embedding=[0.1] * 384) if model is ChunkCreate else data.update(page_num=1)
    with pytest.raises(ValidationError):
        model(**(data | changes))


@pytest.mark.parametrize("field", ["start_line", "end_line"])
@pytest.mark.parametrize("model", [ChunkCreate, PageLineCreate])
def test_line_numbers_are_required(model, field):
    data = dict(document_id=1, start_line=1, end_line=2)
    data.update(content="Text.", embedding=[0.1] * 384) if model is ChunkCreate else data.update(page_num=1)
    del data[field]
    with pytest.raises(ValidationError):
        model(**data)


@pytest.mark.parametrize("page_num", [0, -1, None, 2**31])
def test_page_numbers_are_positive_integers(page_num):
    with pytest.raises(ValidationError):
        PageLineCreate(document_id=1, page_num=page_num, start_line=1, end_line=1)


@pytest.mark.parametrize("embedding", [[0.1] * 383, [float('nan')] * 384, [float('inf')] * 384])
def test_embedding_must_have_384_finite_values(embedding):
    with pytest.raises(ValidationError):
        ChunkCreate(document_id=1, start_line=1, end_line=1, content="Text.", embedding=embedding)


@pytest.mark.parametrize("field", ["heading_id", "position", "paragraph_number"])
def test_chunk_rejects_removed_fields(field):
    with pytest.raises(ValidationError):
        ChunkCreate(document_id=1, start_line=1, end_line=1, content="Text.", embedding=[0.1] * 384, **{field: 1})


def test_single_line_references_are_valid():
    chunk = ChunkCreate(document_id=1, start_line=1, end_line=1, content="Text.", embedding=[0.1] * 384)
    page = PageLineCreate(document_id=1, page_num=1, start_line=1, end_line=1)
    assert chunk.start_line == chunk.end_line == page.start_line == page.end_line == 1


def test_chunk_requires_content():
    with pytest.raises(ValidationError):
        ChunkCreate(document_id=1, start_line=1, end_line=1, embedding=[0.1] * 384)
