"""Data models for AutoMemory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

MemoryKind = Literal["fact", "experience", "summary"]
MemorySource = Literal["extracted", "self", "import"]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _col(row, key: str, default=None):
    """sqlite3.Row has no .get(); read a column only if the query selected it."""
    return row[key] if key in row.keys() else default


@dataclass
class Memory:
    id: str
    content: str
    memory_kind: str = "fact"
    source: str = "extracted"
    user_id: str = "default"
    agent_id: str | None = None
    importance: float = 0.5
    utility: float = 0.0
    feedback_count: int = 0
    access_count: int = 0
    last_accessed: str | None = None
    created_at: str = ""
    updated_at: str = ""
    is_deleted: bool = False
    metadata: dict[str, Any] | None = None
    source_id: str | None = None       # pointer to the verbatim passage (sources.id)
    valid_from: str | None = None      # world-time the fact became true
    valid_to: str | None = None        # world-time it stopped being true; None = current
    superseded_by: str | None = None   # id of the memory that replaced it
    slot: str | None = None            # attribute key for value-changing facts

    @classmethod
    def from_row(cls, row) -> "Memory":
        return cls(
            id=row["id"],
            content=row["content"],
            memory_kind=row["memory_kind"],
            source=row["source"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            importance=row["importance"],
            utility=row["utility"],
            feedback_count=row["feedback_count"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_deleted=bool(row["is_deleted"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            source_id=_col(row, "source_id"),
            valid_from=_col(row, "valid_from"),
            valid_to=_col(row, "valid_to"),
            superseded_by=_col(row, "superseded_by"),
            slot=_col(row, "slot"),
        )


@dataclass
class ScoredMemory:
    memory: Memory
    relevance: float = 0.0      # hybrid vec+bm25 relevance
    retention: float = 1.0      # Ebbinghaus retention at query time
    priority: float = 1.0       # decay/importance/utility multiplier
    score_pre: float = 0.0      # relevance * priority (+ spreading)
    rerank_score: float | None = None
    final_score: float = 0.0
    via_link: bool = False      # pulled in by activation spreading


@dataclass
class STMEvent:
    id: int
    session_id: str
    user_id: str
    role: str
    content: str
    created_at: str
    consolidated: bool = False

    @classmethod
    def from_row(cls, row) -> "STMEvent":
        return cls(
            id=row["id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            consolidated=bool(row["consolidated"]),
        )


@dataclass
class MemoryEvent:
    """Result of a write-pipeline decision, mem0-style."""

    event: Literal["ADD", "UPDATE", "DELETE", "NONE"]
    memory: Memory | None = None
    previous_content: str | None = None


def _age_str(ts: str, now: datetime) -> str:
    delta = now - parse_iso(ts)
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}min ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


@dataclass
class RecallResult:
    retrieval_id: str
    query: str
    short_term: list[STMEvent] = field(default_factory=list)
    long_term: list[ScoredMemory] = field(default_factory=list)
    history: list[ScoredMemory] = field(default_factory=list)  # superseded, separate quota
    sources: dict[str, str] = field(default_factory=dict)  # source_id -> verbatim passage

    def to_prompt(self) -> str:
        """Render as a context-injectable block."""
        now = datetime.now(timezone.utc)
        lines = [f'<memory retrieval_id="{self.retrieval_id}">']
        if self.short_term:
            lines.append(f"[short-term | last {len(self.short_term)} messages]")
            for ev in self.short_term:
                lines.append(f"- ({ev.role}, {_age_str(ev.created_at, now)}) {ev.content}")
        if self.long_term:
            lines.append(f"[long-term | top {len(self.long_term)} relevant]")
            for i, sm in enumerate(self.long_term, 1):
                m = sm.memory
                # bi-temporal annotation: flag facts that are no longer current.
                validity = (
                    f", superseded {m.valid_to[:10]}" if m.valid_to else ""
                )
                src = f", src:{m.source_id[:6]}" if m.source_id else ""
                lines.append(
                    f"{i}. [{m.id[:8]}] ({m.memory_kind}, imp={m.importance:.2f}, "
                    f"{_age_str(m.created_at, now)}{validity}{src}) {m.content}"
                )
        if self.history:
            # NB: history is only ever populated under the experimental separate-quota
            # config (off by default); in the shipped "compete" config self.history is
            # empty and this block does not render. See config.py / retrieval.py.
            # Prior values, on a separate quota; never displace the current facts.
            # The "not current" guidance lives in this section header (once), so the
            # lines can stay neutral dated intervals — preserving the date anchors a
            # timeline question needs without a per-line "discard" signal.
            lines.append(
                f"[earlier | {len(self.history)} prior value(s) that were later "
                f"replaced; not current, but their date ranges are valid evidence "
                f"for past-state / timeline questions]"
            )
            for sm in self.history:
                m = sm.memory
                src = f" src:{m.source_id[:6]}" if m.source_id else ""
                frm = (m.valid_from or m.created_at or "")[:10]
                to = (m.valid_to or "")[:10]
                lines.append(f"- [{frm} until {to}]{src} {m.content}")
        if self.sources:
            # hybrid granularity: the verbatim passages behind the facts above,
            # for questions that need exact names / wording / dates.
            lines.append("[evidence | verbatim source passages]")
            for sid, text in self.sources.items():
                lines.append(f"(src:{sid[:6]}) {text}")
        if self.long_term:
            lines.append(
                f'If these memories helped (or hurt) your answer, call '
                f'report_outcome("{self.retrieval_id}", quality) with quality in [0,1].'
            )
        lines.append("</memory>")
        return "\n".join(lines)
