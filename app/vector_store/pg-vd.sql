


BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    title TEXT,
    source TEXT,
    content TEXT NOT NULL
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    -- One-based, inclusive lines in documents.content.
    start_line INTEGER NOT NULL CHECK (start_line >= 1),
    end_line INTEGER NOT NULL CHECK (end_line >= start_line),
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

CREATE TABLE page_lines (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_num INTEGER NOT NULL CHECK (page_num >= 1),
    -- PDF page coverage in the same Markdown line coordinates as chunks.
    start_line INTEGER NOT NULL CHECK (start_line >= 1),
    end_line INTEGER NOT NULL CHECK (end_line >= start_line)
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_page_lines_document ON page_lines(document_id);

COMMIT;

-- OLD DATA MODEL

-- BEGIN;

-- CREATE EXTENSION IF NOT EXISTS vector;

-- -- Documents
-- CREATE TABLE documents (
--     id SERIAL PRIMARY KEY,
--     -- Stable, namespaced source ID, including the version when available
--     -- (for example, arxiv:1706.03762v7).
--     external_id TEXT NOT NULL UNIQUE,
--     title TEXT,
--     source TEXT,
--     content TEXT NOT NULL
-- );

-- -- Headings
-- CREATE TABLE headings (
--     id SERIAL PRIMARY KEY,

--     document_id INTEGER NOT NULL
--         REFERENCES documents(id) ON DELETE CASCADE,

--     parent_id INTEGER,

--     level INTEGER NOT NULL
--         CHECK (level BETWEEN 1 AND 6),

--     title TEXT NOT NULL,
--     position INTEGER NOT NULL,

--     -- One-based, inclusive Markdown lines covering this section and subsections.
--     start_line INTEGER NOT NULL CHECK (start_line >= 1),
--     end_line INTEGER NOT NULL CHECK (end_line >= start_line),

--     UNIQUE (document_id, id),
--     FOREIGN KEY (document_id, parent_id)
--         REFERENCES headings(document_id, id) ON DELETE CASCADE
-- );

-- -- Chunks
-- CREATE TABLE chunks (
--     id SERIAL PRIMARY KEY,

--     document_id INTEGER NOT NULL
--         REFERENCES documents(id) ON DELETE CASCADE,

--     heading_id INTEGER,

--     position INTEGER NOT NULL,
--     -- One-based paragraph number within the heading; shared by its chunks.
--     -- With heading_id NULL, numbering is within the document's opening text.
--     paragraph_number INTEGER NOT NULL CHECK (paragraph_number >= 1),
--     content TEXT NOT NULL,

--     embedding VECTOR(384) NOT NULL,

--     -- Clear only heading_id when a heading is removed; keep the document.
--     -- Column-specific ON DELETE SET NULL requires PostgreSQL 15 or later.
--     FOREIGN KEY (document_id, heading_id)
--         REFERENCES headings(document_id, id)
--         ON DELETE SET NULL (heading_id)
-- );

-- -- Indexes for relational lookups
-- CREATE INDEX idx_headings_document
--     ON headings(document_id);

-- CREATE INDEX idx_headings_parent
--     ON headings(parent_id);

-- CREATE INDEX idx_chunks_document
--     ON chunks(document_id);

-- CREATE INDEX idx_chunks_heading
--     ON chunks(heading_id);

-- -- potential indexes for vector similarity search ivfflat or hnsw

-- COMMIT;
