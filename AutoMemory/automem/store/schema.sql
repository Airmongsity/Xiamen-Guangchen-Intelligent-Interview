-- AutoMemory SQLite schema. The vec0 virtual table is created separately in db.py
-- (only when the sqlite-vec extension loads successfully).

CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    memory_kind   TEXT NOT NULL DEFAULT 'fact',      -- fact | experience | summary
    source        TEXT NOT NULL DEFAULT 'extracted', -- extracted | self | import
    user_id       TEXT NOT NULL DEFAULT 'default',
    agent_id      TEXT,
    importance    REAL NOT NULL DEFAULT 0.5,         -- [0,1]
    utility       REAL NOT NULL DEFAULT 0.0,         -- [-1,1] feedback EMA
    feedback_count INTEGER NOT NULL DEFAULT 0,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    is_deleted    INTEGER NOT NULL DEFAULT 0,
    metadata      TEXT,
    -- hybrid-granularity storage: a compressed fact points back to the verbatim
    -- passage it was extracted from (sources.id), so recall can expand evidence.
    source_id     TEXT,
    -- bi-temporal validity: a superseded fact stays retrievable (is_deleted=0)
    -- but is marked historically valid. valid_to IS NULL  => currently true.
    valid_from    TEXT,                              -- world-time the fact became true
    valid_to      TEXT,                              -- world-time it stopped being true
    superseded_by TEXT,                              -- id of the memory that replaced it
    -- attribute key for value-changing facts (employer, home_city, pet_name...),
    -- used to find the prior value during reconcile even at moderate similarity.
    slot          TEXT
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, is_deleted);
CREATE INDEX IF NOT EXISTS idx_mem_slot ON memories(user_id, slot);

-- Verbatim source passages (the "raw-text pointer" target). One row per ingested
-- conversation chunk; many extracted facts may reference the same source.
CREATE TABLE IF NOT EXISTS sources (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL DEFAULT 'default',
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata   TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_user ON sources(user_id);

-- Fallback embedding store (always populated; brute-force path reads from here).
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id),
    embedding BLOB NOT NULL                          -- float32 little-endian
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories', content_rowid='rowid',
    tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS memory_links (
    src_id    TEXT NOT NULL,
    dst_id    TEXT NOT NULL,
    link_kind TEXT NOT NULL DEFAULT 'related',  -- related | derived_from | contradicts
    weight    REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id)
);
CREATE INDEX IF NOT EXISTS idx_links_dst ON memory_links(dst_id);

CREATE TABLE IF NOT EXISTS retrievals (
    retrieval_id TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    query        TEXT NOT NULL,
    memory_ids   TEXT NOT NULL,   -- JSON array, final ranking order
    scores       TEXT NOT NULL,   -- JSON array of final scores
    created_at   TEXT NOT NULL,
    feedback     REAL,            -- NULL until report_outcome
    feedback_at  TEXT
);

CREATE TABLE IF NOT EXISTS stm_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    user_id    TEXT NOT NULL DEFAULT 'default',
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_stm_user ON stm_events(user_id, consolidated);
