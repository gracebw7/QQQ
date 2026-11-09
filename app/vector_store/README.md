# PostgreSQL vector store

This is where we store a paper's Markdown, the chunks we search over, and the
mapping back to PDF pages. PostgreSQL holds the data, and pgvector compares the
embeddings. Python handles parsing, sentence grouping, and embedding generation.

The main files are:

- [pg-vd.sql](pg-vd.sql): the database tables and constraints.
- [vec_func.py](vec_func.py): parsing, chunking, ingestion, and database helpers.
- [compose.yaml](compose.yaml): the local PostgreSQL container.
- [vector_schemas.py](../vector_schemas.py): the Pydantic models used before inserts.

The commented old models are kept for reference. The active schema has no headings table.

## What the database stores

### `documents`

One row represents one document.

| Column | What it holds |
| --- | --- |
| `id` | Database-generated integer ID. |
| `external_id` | Required, unique source identifier, such as `arxiv:1706.03762v7`. |
| `title` | Optional document title. |
| `source` | Optional source name or URL. |
| `content` | The complete Markdown text. Windows and old Mac line endings are normalized to `\n`. |

### `chunks`

Each row contains a group of sentences from one paragraph.

| Column | What it holds |
| --- | --- |
| `id` | Database-generated integer ID. |
| `document_id` | The document this chunk belongs to. |
| `start_line` | First Markdown line touched by the chunk. |
| `end_line` | Last Markdown line touched by the chunk. |
| `content` | The exact chunk text passed to the embedding model. |
| `embedding` | A 384-dimensional pgvector embedding. |

Line numbers start at **1**, and both ends of the range are included. Two chunks
can share a line range if their sentences appear on the same Markdown line.
Search returns the saved chunk text, so it does not have to reconstruct sentences
from that range.

### `page_lines`

These rows connect Markdown lines to the original PDF pages.

| Column | What it holds |
| --- | --- |
| `id` | Database-generated integer ID. |
| `document_id` | The document the page belongs to. |
| `page_num` | PDF page number, starting at 1. |
| `start_line` | First Markdown line covered by that page. |
| `end_line` | Last Markdown line covered by that page, included in the range. |

Page mappings are optional. They need to come from PDF extraction because plain
Markdown does not reliably tell us where PDF pages begin. You can supply them
during ingestion or add them afterward with `insert_page_line()`.

Both chunks and page mappings reference `documents.id`. Deleting a document also
deletes its chunks and page mappings. Line ranges must be positive and ordered.
There are indexes on the two `document_id` columns; vector search currently uses
an exact scan, with no HNSW or IVFFlat index.

## What `vec_func.py` does

The usual entry point is `parse_and_save()`. It takes a Markdown **string**, parses
paragraphs, groups their sentences, embeds the resulting chunks, and saves the
document and related rows. It returns the new document ID.

Embeddings are generated before opening the database connection. The inserts then
share one transaction: either all of them succeed, or all are rolled back.

| Function or class | Purpose |
| --- | --- |
| `ParseParagraph` | A small container for text and its Markdown line range. Also used for parsed chunks. |
| `markdown_parse(content)` | Returns paragraph blocks and their line ranges. It reads paragraph tokens, so headings, fenced code, and other non-paragraph blocks are not embedded. |
| `_sentence_splitter()` | Builds and caches spaCy's English sentencizer. No downloaded spaCy language model is needed. |
| `chunk_paragraph(paragraph, sentences_per_chunk=3)` | Groups detected sentences in order, within one paragraph, and calculates each group's line range. It keeps sentences whole. |
| `embed_chunk(chunks)` | Generates normalized embeddings with `all-MiniLM-L6-v2`. The embedding model is loaded on each call and may need downloading on first use. |
| `parse_and_save(markdown, external_id, ...)` | Runs ingestion. Optional arguments are `title`, `source`, `sentences_per_chunk`, and `page_lines`. |
| `connect_db()` | Opens a connection for a `with` block. Commits on success, rolls back on error, and closes the connection. |
| `insert_document(conn, document)` | Inserts a `DocumentCreate` model and returns its ID. |
| `get_document(conn, external_id)` | Returns a `Document` model, or `None` if it is not found. |
| `insert_chunk(conn, chunk)` | Inserts a `ChunkCreate` model and returns its ID. |
| `insert_page_line(conn, page)` | Inserts a `PageLineCreate` model and returns its ID. |
| `get_page_lines(conn, document_id, start_line, end_line)` | Returns page mappings that overlap the supplied Markdown range. |
| `search_chunks(conn, embedding, limit=6, document_id=None)` | Finds chunks by cosine distance. Returns dictionaries with chunk text, line ranges, source metadata, matching page numbers, and similarity. |

Search looks across all documents unless `document_id` is supplied. A chunk can
match more than one PDF page. When there is no matching page mapping, `page_nums`
is an empty list. Lower-level insert helpers use the caller's transaction rather
than committing individually.

`ChunkCreate` validates that embeddings contain 384 finite numbers. The range
models validate line numbers, and the page models also validate `page_num`.
PostgreSQL enforces document references and unique external IDs.

There are two practical limits to keep in mind: importing the same `external_id`
twice raises a duplicate error, and MiniLM truncates input beyond 256 word pieces.
A very long chunk can therefore have some text left out of its embedding even
though its complete text is saved. Sentence boundaries are rule-based and can
occasionally be wrong.

## Start the database with Docker

Use PowerShell from the **repository root**, where `pyproject.toml` lives. Docker
Desktop should be running with Linux containers. The commands below use the
Compose project name `qqq-citations`; keep that name when managing this database.

First install the Python dependencies:

```powershell
uv sync
```

Set a local development password in the same terminal:

```powershell
$env:POSTGRES_PASSWORD = 'replace-with-your-password'
```

Then start PostgreSQL:

```powershell
docker compose -p qqq-citations -f app/vector_store/compose.yaml up -d
docker compose -p qqq-citations -f app/vector_store/compose.yaml ps
docker compose -p qqq-citations -f app/vector_store/compose.yaml logs --tail 80 pg-vd
```

Wait for the service to show `healthy`. On the first startup with an empty data
volume, PostgreSQL runs `pg-vd.sql` and creates the extension and tables.

The Compose file uses `pgvector/pgvector:pg18-trixie`, which includes PostgreSQL 18
and pgvector. Its named volume is mounted at `/var/lib/postgresql`. The SQL file
is mounted read-only from beside `compose.yaml`.
See the [pgvector Docker instructions](https://github.com/pgvector/pgvector#docker)
and [Postgres image documentation](https://hub.docker.com/_/postgres).

The connection settings are:

| Setting | Default |
| --- | --- |
| Host | `localhost` |
| Port | `5432` |
| Database | `qqq` |
| Username | `qqq` |
| Password | The value of `POSTGRES_PASSWORD` above. |

`connect_db()` uses `DATABASE_URL` if it is set. Otherwise it uses `PGHOST`,
`PGPORT`, `PGDATABASE`, and `PGUSER`, with the defaults above, plus
`POSTGRES_PASSWORD`. Python does not automatically load a `.env` file here.
Using the shell variable above makes the password available to both Compose and
Python commands started from that terminal. Set it again in a new terminal.

If another database already uses port 5432, stop its container or change the
Compose port mapping to `127.0.0.1:5433:5432`. If you use 5433, also set
`$env:PGPORT = '5433'` for Python.

### Check the tables

Open a SQL session:

```powershell
docker compose -p qqq-citations -f app/vector_store/compose.yaml exec pg-vd psql -U qqq -d qqq
```

Inside `psql`, run:

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
\dt
\d chunks
\d page_lines
```

You should see the vector extension and the `documents`, `chunks`, and
`page_lines` tables. Check that `chunks` includes both `content` and
`embedding vector(384)`. Type `\q` to exit.

### Stop it without losing data

```powershell
docker compose -p qqq-citations -f app/vector_store/compose.yaml down
```

The named volume remains. Running `up -d` with the same project name brings the
same database back.

Editing `pg-vd.sql` does **not** update an existing database. Initialization scripts
only run against an empty data directory. For later schema changes, apply a
migration or start a fresh database under a different Compose project name after
stopping the previous container. Changing the environment password also does not
change the password of an already-created database user.
See the [initialization documentation](https://hub.docker.com/_/postgres#initialization-scripts).


