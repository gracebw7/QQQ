"""Real Markdown parsing and Pydantic validation with fake embeddings/database writes."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def vector_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=MagicMock()))
    return importlib.import_module("app.vector_store.vec_func")


@pytest.fixture
def ingestion(vector_module, monkeypatch):
    module = vector_module
    connection = MagicMock()
    connection.__exit__.return_value = False
    monkeypatch.setattr(module, "connect_db", lambda: connection)
    monkeypatch.setattr(module, "insert_document", MagicMock(return_value=10))
    monkeypatch.setattr(module, "insert_chunk", MagicMock())
    monkeypatch.setattr(module, "insert_page_line", MagicMock())
    monkeypatch.setattr(module, "embed_chunk", MagicMock(side_effect=lambda texts: [
        SimpleNamespace(tolist=lambda: [0.1] * 384) for _ in texts
    ]))
    return module, connection


def test_parse_and_save_preserves_lines_and_page_mappings(ingestion):
    module, connection = ingestion
    text = "# Title\n\nFirst sentence.\nSecond sentence! Third sentence?\n\nLast.\n"
    pages = [dict(page_num=1, start_line=1, end_line=3), dict(page_num=2, start_line=4, end_line=6)]
    assert module.parse_and_save(text, "arxiv:example", sentences_per_chunk=2, page_lines=pages) == 10
    document = module.insert_document.call_args.args[1]
    assert document.content == text and document.external_id == "arxiv:example"
    module.embed_chunk.assert_called_once_with(["First sentence.\nSecond sentence!", "Third sentence?", "Last."])
    chunks = [call.args[1] for call in module.insert_chunk.call_args_list]
    assert [(c.start_line, c.end_line) for c in chunks] == [(3, 4), (4, 4), (6, 6)]
    assert [c.content for c in chunks] == module.embed_chunk.call_args.args[0]
    assert all(c.document_id == 10 and len(c.embedding) == 384 for c in chunks)
    saved_pages = [call.args[1].model_dump() for call in module.insert_page_line.call_args_list]
    assert saved_pages == [dict(document_id=10, **page) for page in pages]
    assert connection.__exit__.call_args.args == (None, None, None)


@pytest.mark.parametrize("text, chunk_count", [("No heading here.", 1), ("", 0), ("# Heading only", 0)])
def test_documents_without_paragraphs_or_headers(ingestion, text, chunk_count):
    module, _ = ingestion
    assert module.parse_and_save(text, "arxiv:example") == 10
    assert module.insert_chunk.call_count == chunk_count
    module.insert_page_line.assert_not_called()
    if not chunk_count:
        module.embed_chunk.assert_not_called()


def test_parser_keeps_original_markdown_lines(vector_module):
    paragraphs = vector_module.markdown_parse('# Heading\r\n\r\n**Bold** text.\r\nContinued.\r\n\r\nLast.')
    assert [(p.text, p.start_line, p.end_line) for p in paragraphs] == [
        ("**Bold** text.\nContinued.", 3, 4), ("Last.", 6, 6)
    ]


def test_ingestion_normalizes_line_endings(ingestion):
    module, _ = ingestion
    text = "Café text.\r\nSecond line."
    module.parse_and_save(text, "example")
    assert module.insert_document.call_args.args[1].content == "Café text.\nSecond line."
    chunk = module.insert_chunk.call_args.args[1]
    assert (chunk.start_line, chunk.end_line) == (1, 2)


@pytest.mark.parametrize("separator", ["\v", "\f", "\x85", "\u2028", "\u2029"])
def test_non_markdown_separators_do_not_drop_text_or_shift_lines(ingestion, separator):
    module, _ = ingestion
    text = f"Alpha.{separator}Beta.\n\nLast."
    paragraphs = module.markdown_parse(text)
    assert [(p.text, p.start_line, p.end_line) for p in paragraphs] == [
        (f"Alpha.{separator}Beta.", 1, 1), ("Last.", 3, 3)
    ]
    module.parse_and_save(text, "example")
    assert module.insert_document.call_args.args[1].content == text
    chunks = [call.args[1] for call in module.insert_chunk.call_args_list]
    assert [c.content for c in chunks] == [f"Alpha.{separator}Beta.", "Last."]
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 1), (3, 3)]


def test_mixed_newlines_use_the_same_coordinates_for_parsing_and_pages(ingestion):
    module, _ = ingestion
    text = "# Title\r\n\rFirst.\nSecond.\r\n\rLast.\r"
    pages = [dict(page_num=1, start_line=1, end_line=6)]
    module.parse_and_save(text, "example", page_lines=pages)
    assert module.insert_document.call_args.args[1].content == "# Title\n\nFirst.\nSecond.\n\nLast.\n"
    chunks = [call.args[1] for call in module.insert_chunk.call_args_list]
    assert [(c.start_line, c.end_line) for c in chunks] == [(3, 4), (6, 6)]
    assert module.insert_page_line.call_args.args[1].end_line == 6


@pytest.mark.parametrize("text, end_line", [
    ("Alpha.\fBeta.", 2), ("One.\n", 2), ("", 1),
])
def test_page_validation_does_not_count_special_separators_or_phantom_lines(ingestion, text, end_line):
    module, _ = ingestion
    with pytest.raises(ValueError, match="within the document"):
        module.parse_and_save(text, "example", page_lines=[dict(page_num=1, start_line=1, end_line=end_line)])
    module.insert_document.assert_not_called()


@pytest.mark.parametrize("text, expected", [
    ("One. Two! Three? Four.", ["One. Two!", "Three? Four."]),
    ("A sentence without punctuation", ["A sentence without punctuation"]),
    ("   ", []), ("", []),
])
def test_sentence_grouping(vector_module, text, expected):
    paragraph = vector_module.ParseParagraph(text, 1, max(1, len(text.splitlines())))
    assert [c.text for c in vector_module.chunk_paragraph(paragraph, 2)] == expected


@pytest.mark.parametrize("text, expected", [
    ("Dr. Smith scored 92.5%. Next result.", ["Dr. Smith scored 92.5%.", "Next result."]),
    ('She said, "It works." Then left.', ['She said, "It works."', 'Then left.']),
])
def test_sentence_boundaries_handle_abbreviations_decimals_and_quotes(vector_module, text, expected):
    paragraph = vector_module.ParseParagraph(text, 1, 1)
    assert [c.text for c in vector_module.chunk_paragraph(paragraph, 1)] == expected


def test_sentence_spans_preserve_text_and_line_references(vector_module):
    paragraph = vector_module.ParseParagraph('  Dr. Smith tested\nthe model.\n"It works!"\nNext result.', 10, 13)
    chunks = vector_module.chunk_paragraph(paragraph, 1)
    assert [(c.text, c.start_line, c.end_line) for c in chunks] == [
        ('Dr. Smith tested\nthe model.', 10, 11),
        ('"It works!"', 12, 12),
        ('Next result.', 13, 13),
    ]


def test_page_mapping_outside_document_is_rejected_before_writes(ingestion):
    module, _ = ingestion
    with pytest.raises(ValueError, match="within the document"):
        module.parse_and_save("One.", "example", page_lines=[dict(page_num=1, start_line=1, end_line=2)])
    module.insert_document.assert_not_called()
    module.embed_chunk.assert_not_called()


@pytest.mark.parametrize("helper", ["insert_chunk", "insert_page_line"])
def test_insert_failure_reaches_transaction_context(ingestion, helper):
    module, connection = ingestion
    getattr(module, helper).side_effect = RuntimeError("insert failed")
    with pytest.raises(RuntimeError, match="insert failed"):
        module.parse_and_save("A sentence.", "example", page_lines=[dict(page_num=1, start_line=1, end_line=1)])
    assert connection.__exit__.call_args.args[0] is RuntimeError


def test_insert_helpers_use_new_columns(vector_module):
    from app import vector_schemas as db
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"id": 42}
    chunk = db.ChunkCreate(document_id=10, start_line=1, end_line=2, content="Exact chunk text.", embedding=[0.1] * 384)
    assert vector_module.insert_chunk(conn, chunk) == 42
    query, params = conn.execute.call_args.args
    assert "::vector" in query and params['embedding'] == str(chunk.embedding)
    assert set(params) == {"document_id", "start_line", "end_line", "content", "embedding"}
    assert params["content"] == chunk.content and "%(content)s" in query
    page = db.PageLineCreate(document_id=10, page_num=2, start_line=1, end_line=2)
    assert vector_module.insert_page_line(conn, page) == 42
    query, params = conn.execute.call_args.args
    assert "INSERT INTO page_lines" in query and params == page.model_dump()


def test_page_lookup_uses_inclusive_overlap(vector_module):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [dict(id=42, document_id=10, page_num=2, start_line=5, end_line=9)]
    pages = vector_module.get_page_lines(conn, 10, 3, 5)
    assert pages[0].page_num == 2
    query, params = conn.execute.call_args.args
    assert "start_line <= %s AND end_line >= %s" in query
    assert params == (10, 5, 3)


def test_search_returns_exact_chunk_text_and_overlapping_pages(vector_module):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        dict(chunk_id=1, document_id=10, start_line=2, end_line=2,
             content="First sentence.", page_nums=[1, 2], similarity=1.0),
        dict(chunk_id=2, document_id=10, start_line=2, end_line=2,
             content="Second sentence.", page_nums=[], similarity=0.5),
    ]
    rows = vector_module.search_chunks(conn, [0.1] * 384, document_id=10)
    assert [row['content'] for row in rows] == ["First sentence.", "Second sentence."]
    assert [row['page_nums'] for row in rows] == [[1, 2], []]
    assert all('document_content' not in row for row in rows)
    query, params = conn.execute.call_args.args
    assert "c.content" in query and "d.content" not in query
    assert "p.start_line <= c.end_line AND p.end_line >= c.start_line" in query
    assert "headings" not in query and params[1:3] == (10, 10)
