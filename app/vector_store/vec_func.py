from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import os

from markdown_it import MarkdownIt
import psycopg
from psycopg.rows import dict_row
from pydantic import TypeAdapter
from sentence_transformers import SentenceTransformer

from app import vector_schemas as db

import spacy


@dataclass
class ParseParagraph:
    text: str
    start_line: int
    end_line: int


def markdown_parse(content: str) -> list[ParseParagraph]:
    """Parse the same text saved in documents.content, using one-based lines."""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.split("\n")
    paragraphs = []
    for token in MarkdownIt("commonmark").parse(content):
        if token.type == "paragraph_open":
            start, end = token.map
            paragraphs.append(ParseParagraph("\n".join(lines[start:end]), start + 1, end))
    return paragraphs


@lru_cache(maxsize=1)
def _sentence_splitter():

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def chunk_paragraph(paragraph: ParseParagraph, sentences_per_chunk: int = 3) -> list[ParseParagraph]:
    """Group sentences and keep the Markdown lines each group touches."""
    if sentences_per_chunk < 1:
        raise ValueError("sentences_per_chunk must be at least 1")
    sentences = [s for s in _sentence_splitter()(paragraph.text).sents if s.text.strip()]
    chunks = []
    for index in range(0, len(sentences), sentences_per_chunk):
        group = sentences[index:index + sentences_per_chunk]
        first = group[0]
        start = first.start_char + len(first.text) - len(first.text.lstrip())
        text = paragraph.text[start:group[-1].end_char].rstrip()
        end = start + len(text)
        chunks.append(ParseParagraph(
            text=text,
            start_line=paragraph.start_line + paragraph.text[:start].count("\n"),
            end_line=paragraph.start_line + paragraph.text[:end].count("\n"),
        ))
    return chunks


def embed_chunk(chunks: list[str]):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # MiniLM cuts off text past 256 word pieces, so very long chunks may lose some context.
    return model.encode(chunks, normalize_embeddings=True)


def parse_and_save(markdown: str, external_id: str, title: str | None = None,
                   source: str | None = None, sentences_per_chunk: int = 3,
                   page_lines: list[db.PageLineRange | dict] | None = None) -> int:
    """Save a document, embedded chunks, and optional PDF page mappings atomically.

    CRLF and CR line endings are normalized to LF before storing and parsing.
    Page mappings must refer to these normalized Markdown lines. They must be
    supplied by PDF extraction; page numbers cannot be inferred from Markdown.
    """
    if sentences_per_chunk < 1:
        raise ValueError("sentences_per_chunk must be at least 1")
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    document = db.DocumentCreate(
        external_id=external_id, title=title, source=source,
        content=markdown,
    )
    pages = [db.PageLineRange.model_validate(page) for page in (page_lines or [])]
    # A final newline terminates the last line; it does not add another one.
    line_count = len(markdown.split("\n")) - int(not markdown or markdown.endswith("\n"))
    if any(page.end_line > line_count for page in pages):
        raise ValueError("Page mappings must be within the document's Markdown lines")
    chunks = [
        chunk
        for paragraph in markdown_parse(document.content)
        for chunk in chunk_paragraph(paragraph, sentences_per_chunk)
    ]
    embeddings = embed_chunk([chunk.text for chunk in chunks]) if chunks else []

    with connect_db() as conn:
        document_id = insert_document(conn, document)
        for page in pages:
            insert_page_line(conn, db.PageLineCreate(document_id=document_id, **page.model_dump()))
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            insert_chunk(conn, db.ChunkCreate(
                document_id=document_id,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.text,
                embedding=embedding.tolist(),
            ))
    return document_id


@contextmanager
def connect_db():
    """Commit on success, roll back on error, and always close the connection."""
    settings = dict(row_factory=dict_row, connect_timeout=10)
    if os.getenv("DATABASE_URL"):
        connection = psycopg.connect(os.environ["DATABASE_URL"], **settings)
    else:
        connection = psycopg.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
            dbname=os.getenv("PGDATABASE", "qqq"),
            user=os.getenv("PGUSER", "qqq"),
            password=os.getenv("POSTGRES_PASSWORD"),
            **settings,
        )
    with connection:
        yield connection


def insert_document(conn, document: db.DocumentCreate) -> int:
    return conn.execute(
        """INSERT INTO documents (external_id, title, source, content)
           VALUES (%(external_id)s, %(title)s, %(source)s, %(content)s)
           RETURNING id""",
        document.model_dump(),
    ).fetchone()["id"]


def get_document(conn, external_id: str) -> db.Document | None:
    row = conn.execute(
        "SELECT * FROM documents WHERE external_id = %s", (external_id,)
    ).fetchone()
    return db.Document.model_validate(row) if row else None


def insert_chunk(conn, chunk: db.ChunkCreate) -> int:
    values = chunk.model_dump()
    values["embedding"] = str(chunk.embedding)
    return conn.execute(
        """INSERT INTO chunks (document_id, start_line, end_line, content, embedding)
           VALUES (%(document_id)s, %(start_line)s, %(end_line)s, %(content)s,
                   %(embedding)s::vector)
           RETURNING id""",
        values,
    ).fetchone()["id"]


def insert_page_line(conn, page: db.PageLineCreate) -> int:
    return conn.execute(
        """INSERT INTO page_lines (document_id, page_num, start_line, end_line)
           VALUES (%(document_id)s, %(page_num)s, %(start_line)s, %(end_line)s)
           RETURNING id""",
        page.model_dump(),
    ).fetchone()["id"]


def get_page_lines(conn, document_id: int, start_line: int, end_line: int) -> list[db.PageLine]:
    """Find PDF pages overlapping a Markdown line range."""
    db.LineRange(start_line=start_line, end_line=end_line)
    rows = conn.execute(
        """SELECT * FROM page_lines
           WHERE document_id = %s AND start_line <= %s AND end_line >= %s
           ORDER BY page_num, start_line""",
        (document_id, end_line, start_line),
    ).fetchall()
    return [db.PageLine.model_validate(row) for row in rows]


def search_chunks(conn, embedding: db.Embedding, limit: int = 6,
                  document_id: int | None = None) -> list[dict]:
    """Return nearest chunks with their exact text, source lines, and PDF pages."""
    vector = TypeAdapter(db.Embedding).validate_python(embedding)
    if not any(vector):
        raise ValueError("Cosine search requires a nonzero embedding")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return conn.execute(
        """SELECT c.id AS chunk_id, c.document_id, c.start_line, c.end_line,
                  c.content, d.external_id, d.title, d.source,
                  ARRAY(SELECT DISTINCT p.page_num FROM page_lines p
                        WHERE p.document_id = c.document_id
                          AND p.start_line <= c.end_line AND p.end_line >= c.start_line
                        ORDER BY p.page_num) AS page_nums,
                  1 - (c.embedding <=> %s::vector) AS similarity
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE (%s::integer IS NULL OR c.document_id = %s)
           ORDER BY c.embedding <=> %s::vector
           LIMIT %s""",
        (str(vector), document_id, document_id, str(vector), limit),
    ).fetchall()
