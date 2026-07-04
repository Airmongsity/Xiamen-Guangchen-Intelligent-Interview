"""SQLite connection management with optional sqlite-vec acceleration.

Embeddings are always stored as float32 BLOBs in `memory_embeddings` (source of
truth). If the sqlite-vec extension loads, a vec0 virtual table is kept in sync
and used for KNN; otherwise KNN falls back to brute-force numpy cosine, which is
fine at demo scale (<10k memories).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Columns added after the original release; ALTER them onto pre-existing DBs
# (CREATE TABLE IF NOT EXISTS won't touch a table that already exists).
_MEM_MIGRATIONS = {
    "source_id": "TEXT",
    "valid_from": "TEXT",
    "valid_to": "TEXT",
    "superseded_by": "TEXT",
    "slot": "TEXT",
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotently bring an existing `memories` table up to the current schema."""
    have = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
    for col, decl in _MEM_MIGRATIONS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")


def vec_to_blob(vec: list[float] | np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Database:
    """Thread-local connection manager.

    Python's sqlite3 connections are not safe for concurrent use from multiple
    threads even in serialized mode (the statement cache trips on
    InterfaceError), so each thread gets its own connection; WAL mode gives
    concurrent readers + a single writer across connections.

    Note: db_path=":memory:" would create one private DB per thread — use a
    file path (or a tempfile) when threads must share state.
    """

    def __init__(self, db_path: str, embed_dim: int, *, readonly: bool = False):
        self._db_path = db_path
        self._embed_dim = embed_dim
        self._readonly = readonly
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._track_lock = threading.Lock()
        # open eagerly on the creating thread: creates schema (RW) / detects vec.
        # readonly=True opens an OS-level read-only connection: any stray write
        # (e.g. a forgotten touch_accessed during evaluation) raises instead of
        # silently mutating the store — a structural guard, not a convention.
        if readonly:
            self.vec_available = self._probe_vec_ro(self.conn)
        else:
            self.vec_available = self._detect_vec(self.conn)

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open()
            self._local.conn = conn
            with self._track_lock:
                self._all_conns.append(conn)
        return conn

    def _open(self) -> sqlite3.Connection:
        # check_same_thread=False only so close() can run from the main thread;
        # by construction each connection is used by a single thread otherwise
        if self._readonly:
            # OS-level read-only: no DDL/migrate/WAL (the DB already has them);
            # query_only also blocks writes from this connection.
            conn = sqlite3.connect(
                f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._load_vec(conn)
            return conn
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        _migrate(conn)
        self._load_vec(conn)
        conn.commit()
        return conn

    def _probe_vec_ro(self, conn: sqlite3.Connection) -> bool:
        """Read-only vec detection: the vec0 table must exist in the DB AND the
        extension must have loaded for this connection. Otherwise fall back to the
        brute-force numpy path (also read-only-safe)."""
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'memories_vec'"
        ).fetchone() is not None
        if not has_table:
            return False
        try:
            conn.execute("SELECT memory_id FROM memories_vec LIMIT 1").fetchone()
            return True
        except sqlite3.OperationalError:
            return False  # module not loaded -> brute-force

    def _load_vec(self, conn: sqlite3.Connection) -> bool:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return True
        except Exception:
            return False  # brute-force fallback handles KNN

    def _detect_vec(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0("
                f"memory_id TEXT PRIMARY KEY, embedding FLOAT[{self._embed_dim}])"
            )
            conn.commit()
            return True
        except sqlite3.OperationalError:
            return False

    def close(self) -> None:
        with self._track_lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except sqlite3.ProgrammingError:
                    pass
            self._all_conns.clear()
        self._local = threading.local()


class VectorIndex:
    """KNN interface that is identical for the vec0 and brute-force backends."""

    def __init__(self, db: Database):
        self._db = db
        self._vec = db.vec_available

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._db.conn

    def upsert(self, memory_id: str, embedding: list[float] | np.ndarray) -> None:
        blob = vec_to_blob(embedding)
        self._conn.execute(
            "INSERT INTO memory_embeddings(memory_id, embedding) VALUES (?, ?) "
            "ON CONFLICT(memory_id) DO UPDATE SET embedding = excluded.embedding",
            (memory_id, blob),
        )
        if self._vec:
            self._conn.execute(
                "DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,)
            )
            self._conn.execute(
                "INSERT INTO memories_vec(memory_id, embedding) VALUES (?, ?)",
                (memory_id, blob),
            )

    def delete(self, memory_id: str) -> None:
        self._conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        )
        if self._vec:
            self._conn.execute(
                "DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,)
            )

    def get(self, memory_id: str) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT embedding FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return blob_to_vec(row["embedding"]) if row else None

    def knn(
        self, query_vec: list[float] | np.ndarray, k: int, *, user_id: str
    ) -> list[tuple[str, float]]:
        """Return [(memory_id, cosine_similarity)] of non-deleted memories, best first."""
        if self._vec:
            return self._knn_vec(query_vec, k, user_id)
        return self._knn_brute(query_vec, k, user_id)

    def _knn_vec(self, query_vec, k, user_id) -> list[tuple[str, float]]:
        # Over-fetch because the deleted/user filter is applied after the ANN query.
        rows = self._conn.execute(
            "SELECT v.memory_id, v.distance FROM memories_vec v "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (vec_to_blob(query_vec), min(k * 4, 4096)),
        ).fetchall()
        out: list[tuple[str, float]] = []
        for row in rows:
            mem = self._conn.execute(
                "SELECT user_id, is_deleted FROM memories WHERE id = ?",
                (row["memory_id"],),
            ).fetchone()
            if mem and mem["user_id"] == user_id and not mem["is_deleted"]:
                # vec0 default metric is L2; recompute exact cosine from the blob
                # so both backends return the same similarity scale.
                vec = self.get(row["memory_id"])
                out.append((row["memory_id"], _cosine(np.asarray(query_vec), vec)))
            if len(out) >= k:
                break
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def _knn_brute(self, query_vec, k, user_id) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT e.memory_id, e.embedding FROM memory_embeddings e "
            "JOIN memories m ON m.id = e.memory_id "
            "WHERE m.user_id = ? AND m.is_deleted = 0",
            (user_id,),
        ).fetchall()
        if not rows:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        ids = [r["memory_id"] for r in rows]
        mat = np.vstack([blob_to_vec(r["embedding"]) for r in rows])
        norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0)
        norms[norms == 0] = 1.0
        sims = (mat @ q) / norms
        order = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in order]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)
