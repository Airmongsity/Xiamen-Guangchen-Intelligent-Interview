"""MemoryStore: all persistence operations for AutoMemory."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from ..models import Memory, STMEvent, utcnow_iso
from .db import Database, VectorIndex


def _new_id() -> str:
    return uuid.uuid4().hex


class MemoryStore:
    def __init__(self, db_path: str, embed_dim: int, *, readonly: bool = False):
        self._db = Database(db_path, embed_dim, readonly=readonly)
        self.readonly = readonly
        self.vec_available = self._db.vec_available
        self.index = VectorIndex(self._db)
        # serializes multi-statement write transactions across threads so WAL's
        # single-writer rule doesn't surface as SQLITE_BUSY
        self._write_lock = threading.RLock()

    @property
    def conn(self):
        return self._db.conn

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------- memories

    def insert_memory(
        self,
        content: str,
        embedding: list[float] | np.ndarray,
        *,
        memory_kind: str = "fact",
        source: str = "extracted",
        user_id: str = "default",
        agent_id: str | None = None,
        importance: float = 0.5,
        created_at: str | None = None,
        metadata: dict | None = None,
        memory_id: str | None = None,
        source_id: str | None = None,
        valid_from: str | None = None,
        slot: str | None = None,
    ) -> Memory:
        now = utcnow_iso()
        mem = Memory(
            id=memory_id or _new_id(),
            content=content,
            memory_kind=memory_kind,
            source=source,
            user_id=user_id,
            agent_id=agent_id,
            importance=importance,
            created_at=created_at or now,
            updated_at=now,
            metadata=metadata,
            source_id=source_id,
            valid_from=valid_from or created_at or now,
            slot=slot,
        )
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO memories (id, content, memory_kind, source, user_id, agent_id,"
                " importance, created_at, updated_at, metadata, source_id, valid_from, slot)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mem.id, mem.content, mem.memory_kind, mem.source, mem.user_id,
                    mem.agent_id, mem.importance, mem.created_at, mem.updated_at,
                    json.dumps(metadata, ensure_ascii=False) if metadata else None,
                    mem.source_id, mem.valid_from, mem.slot,
                ),
            )
            self.index.upsert(mem.id, embedding)
            self.conn.commit()
        return mem

    def get_by_slot(
        self, slot: str, *, user_id: str = "default", current_only: bool = True
    ) -> list[Memory]:
        """Memories filling the same attribute slot for a user (reconcile lookup)."""
        if not slot:
            return []
        sql = (
            "SELECT * FROM memories WHERE user_id = ? AND slot = ? AND is_deleted = 0"
        )
        if current_only:
            sql += " AND valid_to IS NULL"
        rows = self.conn.execute(sql, (user_id, slot)).fetchall()
        return [Memory.from_row(r) for r in rows]

    # ------------------------------------------------------------- sources

    def insert_source(
        self, content: str, *, user_id: str = "default",
        created_at: str | None = None, metadata: dict | None = None,
    ) -> str:
        """Store a verbatim passage; return its id (the raw-text pointer target)."""
        sid = "s-" + _new_id()
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO sources (id, user_id, content, created_at, metadata)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    sid, user_id, content, created_at or utcnow_iso(),
                    json.dumps(metadata, ensure_ascii=False) if metadata else None,
                ),
            )
            self.conn.commit()
        return sid

    def get_sources(self, source_ids: list[str]) -> dict[str, str]:
        ids = [s for s in dict.fromkeys(source_ids) if s]
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, content FROM sources WHERE id IN ({ph})", ids
        ).fetchall()
        return {r["id"]: r["content"] for r in rows}

    def supersede_memory(self, old_id: str, new_id: str, *, valid_to: str) -> None:
        """Bi-temporal invalidation: mark `old_id` historically valid (no longer
        current) without deleting it, so it stays retrievable for history."""
        self.update_memory_fields(
            old_id, valid_to=valid_to, superseded_by=new_id, updated_at=utcnow_iso()
        )

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return Memory.from_row(row) if row else None

    def get_memories(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        ph = ",".join("?" * len(memory_ids))
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE id IN ({ph})", memory_ids
        ).fetchall()
        by_id = {r["id"]: Memory.from_row(r) for r in rows}
        return [by_id[i] for i in memory_ids if i in by_id]

    def list_memories(
        self, *, user_id: str = "default", limit: int = 100, include_deleted: bool = False
    ) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE user_id = ?"
        if not include_deleted:
            sql += " AND is_deleted = 0"
        sql += " ORDER BY created_at DESC LIMIT ?"
        rows = self.conn.execute(sql, (user_id, limit)).fetchall()
        return [Memory.from_row(r) for r in rows]

    def update_memory_content(
        self, memory_id: str, content: str, embedding: list[float] | np.ndarray
    ) -> None:
        with self._write_lock:
            self.conn.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                (content, utcnow_iso(), memory_id),
            )
            self.index.upsert(memory_id, embedding)
            self.conn.commit()

    def update_memory_fields(self, memory_id: str, **fields) -> None:
        allowed = {
            "importance", "utility", "feedback_count", "access_count",
            "last_accessed", "is_deleted", "updated_at", "metadata",
            "source_id", "valid_from", "valid_to", "superseded_by", "slot",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update fields: {bad}")
        if "metadata" in fields and isinstance(fields["metadata"], dict):
            fields["metadata"] = json.dumps(fields["metadata"], ensure_ascii=False)
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE memories SET {cols} WHERE id = ?", (*fields.values(), memory_id)
        )
        self.conn.commit()

    def soft_delete(self, memory_id: str) -> None:
        self.update_memory_fields(memory_id, is_deleted=1, updated_at=utcnow_iso())

    def hard_delete(self, memory_id: str) -> None:
        with self._write_lock:
            self.index.delete(memory_id)
            self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.execute(
                "DELETE FROM memory_links WHERE src_id = ? OR dst_id = ?",
                (memory_id, memory_id),
            )
            self.conn.commit()

    def touch_accessed(self, memory_ids: list[str]) -> None:
        """access_count += 1 and refresh last_accessed for retrieved memories."""
        now = utcnow_iso()
        self.conn.executemany(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ?"
            " WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )
        self.conn.commit()

    def count_memories(self, *, user_id: str = "default") -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE user_id = ? AND is_deleted = 0",
            (user_id,),
        ).fetchone()
        return row["n"]

    # ------------------------------------------------------------- search

    def knn(
        self, query_vec, k: int, *, user_id: str = "default"
    ) -> list[tuple[str, float]]:
        return self.index.knn(query_vec, k, user_id=user_id)

    def fts_search(
        self, query: str, k: int, *, user_id: str = "default"
    ) -> list[tuple[str, float]]:
        """BM25 search. Returns [(memory_id, bm25_raw)] where higher = better."""
        sanitized = _fts_sanitize(query)
        if not sanitized:
            return []
        try:
            rows = self.conn.execute(
                "SELECT m.id, bm25(memories_fts) AS rank FROM memories_fts f"
                " JOIN memories m ON m.rowid = f.rowid"
                " WHERE memories_fts MATCH ? AND m.user_id = ? AND m.is_deleted = 0"
                " ORDER BY rank LIMIT ?",
                (sanitized, user_id, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # malformed FTS query; vector channel still covers recall
        # SQLite bm25() returns lower-is-better (usually negative); flip sign.
        return [(r["id"], -r["rank"]) for r in rows]

    # ------------------------------------------------------------- links

    def add_link(
        self, src_id: str, dst_id: str, *, link_kind: str = "related", weight: float = 1.0
    ) -> None:
        if src_id == dst_id:
            return
        self.conn.execute(
            "INSERT INTO memory_links (src_id, dst_id, link_kind, weight, created_at)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(src_id, dst_id) DO UPDATE SET"
            " link_kind = excluded.link_kind, weight = excluded.weight",
            (src_id, dst_id, link_kind, weight, utcnow_iso()),
        )
        self.conn.commit()

    def neighbors(self, memory_id: str) -> list[tuple[str, str, float]]:
        """Undirected neighbor view: [(neighbor_id, link_kind, weight)]."""
        rows = self.conn.execute(
            "SELECT dst_id AS nid, link_kind, weight FROM memory_links WHERE src_id = ?"
            " UNION ALL "
            "SELECT src_id AS nid, link_kind, weight FROM memory_links WHERE dst_id = ?",
            (memory_id, memory_id),
        ).fetchall()
        return [(r["nid"], r["link_kind"], r["weight"]) for r in rows]

    # ------------------------------------------------------------- retrievals

    def log_retrieval(
        self, *, user_id: str, query: str, memory_ids: list[str], scores: list[float]
    ) -> str:
        rid = "r-" + uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO retrievals (retrieval_id, user_id, query, memory_ids, scores,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                rid, user_id, query,
                json.dumps(memory_ids), json.dumps(scores), utcnow_iso(),
            ),
        )
        self.conn.commit()
        return rid

    def get_retrieval(self, retrieval_id: str):
        return self.conn.execute(
            "SELECT * FROM retrievals WHERE retrieval_id = ?", (retrieval_id,)
        ).fetchone()

    def mark_retrieval_feedback(self, retrieval_id: str, feedback: float) -> None:
        self.conn.execute(
            "UPDATE retrievals SET feedback = ?, feedback_at = ? WHERE retrieval_id = ?",
            (feedback, utcnow_iso(), retrieval_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------- STM

    def add_stm_event(
        self, role: str, content: str, *, session_id: str = "default",
        user_id: str = "default", created_at: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO stm_events (session_id, user_id, role, content, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, role, content, created_at or utcnow_iso()),
        )
        self.conn.commit()

    def get_stm_window(
        self, *, user_id: str = "default", max_items: int = 20, ttl_hours: float = 6.0
    ) -> list[STMEvent]:
        """Most recent unconsolidated events: newest max_items AND within TTL."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        ).isoformat()
        rows = self.conn.execute(
            "SELECT * FROM stm_events WHERE user_id = ? AND consolidated = 0"
            " AND created_at >= ? ORDER BY id DESC LIMIT ?",
            (user_id, cutoff, max_items),
        ).fetchall()
        return [STMEvent.from_row(r) for r in reversed(rows)]

    def get_stm_overflow(
        self, *, user_id: str = "default", max_items: int = 20, ttl_hours: float = 6.0
    ) -> list[STMEvent]:
        """Unconsolidated events that fell out of the STM window (to consolidate)."""
        window_ids = {
            ev.id
            for ev in self.get_stm_window(
                user_id=user_id, max_items=max_items, ttl_hours=ttl_hours
            )
        }
        rows = self.conn.execute(
            "SELECT * FROM stm_events WHERE user_id = ? AND consolidated = 0 ORDER BY id",
            (user_id,),
        ).fetchall()
        return [STMEvent.from_row(r) for r in rows if r["id"] not in window_ids]

    def mark_consolidated(self, event_ids: list[int]) -> None:
        self.conn.executemany(
            "UPDATE stm_events SET consolidated = 1 WHERE id = ?",
            [(i,) for i in event_ids],
        )
        self.conn.commit()

    # ------------------------------------------------------------- stats

    def stats(self, *, user_id: str = "default") -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, AVG(importance) AS avg_imp, AVG(utility) AS avg_util,"
            " SUM(is_deleted) AS forgotten FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        links = self.conn.execute("SELECT COUNT(*) AS n FROM memory_links").fetchone()
        pending = self.conn.execute(
            "SELECT COUNT(*) AS n FROM retrievals WHERE user_id = ? AND feedback IS NULL",
            (user_id,),
        ).fetchone()
        return {
            "memories": row["n"] or 0,
            "forgotten": row["forgotten"] or 0,
            "avg_importance": round(row["avg_imp"], 3) if row["avg_imp"] else None,
            "avg_utility": round(row["avg_util"], 3) if row["avg_util"] else None,
            "links": links["n"] or 0,
            "retrievals_awaiting_feedback": pending["n"] or 0,
            "vector_backend": "sqlite-vec" if self.vec_available else "numpy-bruteforce",
        }


def _fts_sanitize(query: str) -> str:
    """Make an arbitrary query safe for FTS5 MATCH by quoting tokens, OR-joined."""
    tokens = [t for t in query.replace('"', " ").split() if t.strip()]
    return " OR ".join(f'"{t}"' for t in tokens[:32])
