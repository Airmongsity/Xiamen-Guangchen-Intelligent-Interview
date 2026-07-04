"""LLM-driven extraction and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..providers.llm import DeepSeekLLM
from .prompts import (
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
    RECONCILE_SYSTEM,
    RECONCILE_USER_TEMPLATE,
)

VALID_KINDS = {"fact", "experience", "summary"}


@dataclass
class ExtractedMemory:
    content: str
    kind: str = "fact"
    importance: float = 0.5
    slot: str | None = None


@dataclass
class ReconcileDecision:
    op: str  # ADD | UPDATE | DELETE | NONE
    target_id: str | None = None
    text: str | None = None


def format_conversation(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


class MemoryExtractor:
    def __init__(self, llm: DeepSeekLLM):
        self._llm = llm

    def extract(
        self, messages: list[dict], *, conversation_date: str | None = None
    ) -> list[ExtractedMemory]:
        data = self._llm.complete_json(
            EXTRACTION_SYSTEM,
            EXTRACTION_USER_TEMPLATE.format(
                conversation_date=conversation_date or date.today().isoformat(),
                conversation=format_conversation(messages),
            ),
        )
        out: list[ExtractedMemory] = []
        for item in data.get("memories", []):
            content = (item.get("content") or "").strip()
            if not content:
                continue
            kind = item.get("kind", "fact")
            if kind not in VALID_KINDS:
                kind = "fact"
            try:
                importance = min(max(float(item.get("importance", 0.5)), 0.0), 1.0)
            except (TypeError, ValueError):
                importance = 0.5
            slot = item.get("slot")
            slot = slot.strip().lower() if isinstance(slot, str) and slot.strip() else None
            out.append(
                ExtractedMemory(
                    content=content, kind=kind, importance=importance, slot=slot
                )
            )
        return out

    def reconcile(
        self, candidate: str, existing: list[tuple[str, str]]
    ) -> ReconcileDecision:
        """existing: [(memory_id, content)]. Decide ADD/UPDATE/DELETE/NONE."""
        if not existing:
            return ReconcileDecision(op="ADD")
        existing_block = "\n".join(f'- id="{mid}": {content}' for mid, content in existing)
        data = self._llm.complete_json(
            RECONCILE_SYSTEM,
            RECONCILE_USER_TEMPLATE.format(candidate=candidate, existing=existing_block),
        )
        op = str(data.get("op", "ADD")).upper()
        if op not in {"ADD", "UPDATE", "DELETE", "NONE"}:
            op = "ADD"
        target_id = data.get("id")
        valid_ids = {mid for mid, _ in existing}
        if op in {"UPDATE", "DELETE"} and target_id not in valid_ids:
            op = "ADD"  # hallucinated id; fail safe toward keeping information
            target_id = None
        return ReconcileDecision(op=op, target_id=target_id, text=data.get("text"))
