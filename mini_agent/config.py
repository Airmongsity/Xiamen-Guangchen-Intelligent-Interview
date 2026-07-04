"""Configuration and a dependency-free .env loader.

The root `.env` only needs `DEEPSEEK_API_KEY`. Everything else has a default so
the agent runs against the DeepSeek OpenAI-compatible endpoint out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env(path: str | os.PathLike | None = None) -> None:
    """Load a `.env` file into ``os.environ`` without any third-party dependency.

    Existing environment variables are never overwritten (``setdefault``), so a
    real shell export always wins over the file. If ``path`` is omitted we walk
    up from the current working directory until a ``.env`` is found.
    """
    if path is None:
        for base in [Path.cwd(), *Path.cwd().parents]:
            candidate = base / ".env"
            if candidate.exists():
                path = candidate
                break
    if path is None or not Path(path).exists():
        return
    # utf-8-sig transparently strips a BOM if a Windows editor added one.
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _chat_base_url(url: str | None) -> str | None:
    """Turn a possibly-full chat endpoint into an OpenAI SDK ``base_url``.

    ``https://host/v1/chat/completions`` -> ``https://host/v1``. A bare base URL
    is returned unchanged. Empty/None passes through as None.
    """
    if not url:
        return None
    url = url.rstrip("/")
    suffix = "/chat/completions"
    return url[: -len(suffix)] if url.endswith(suffix) else url


@dataclass
class Settings:
    """Runtime knobs, all overridable via environment variables."""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"

    # Agent loop / context limits.
    max_steps: int = 8              # hard cap on tool-calling iterations per turn
    max_history_messages: int = 20  # compress once history grows past this
    keep_recent_messages: int = 8   # messages kept verbatim after compression

    @classmethod
    def from_env(cls) -> "Settings":
        """Read config from the environment, supporting two `.env` conventions:

        * a generic OpenAI-compatible one: ``CHAT_API_URL`` / ``CHAT_API_KEY`` /
          ``CHAT_MODEL`` (``CHAT_API_URL`` may be a full endpoint, e.g. ending in
          ``/chat/completions`` -- we strip that to get the SDK ``base_url``);
        * a DeepSeek one: ``DEEPSEEK_API_KEY`` / ``DEEPSEEK_BASE_URL``.

        ``CHAT_*`` wins when present; otherwise we fall back to DeepSeek defaults.
        """
        load_env()
        api_key = (os.environ.get("CHAT_API_KEY")
                   or os.environ.get("DEEPSEEK_API_KEY")
                   or os.environ.get("OPENAI_API_KEY", ""))
        base_url = (_chat_base_url(os.environ.get("CHAT_API_URL"))
                    or os.environ.get("DEEPSEEK_BASE_URL", cls.base_url))
        model = (os.environ.get("CHAT_MODEL")
                 or os.environ.get("MINI_AGENT_MODEL", cls.model))
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_steps=int(os.environ.get("MINI_AGENT_MAX_STEPS", cls.max_steps)),
            max_history_messages=int(
                os.environ.get("MINI_AGENT_MAX_HISTORY", cls.max_history_messages)
            ),
            keep_recent_messages=int(
                os.environ.get("MINI_AGENT_KEEP_RECENT", cls.keep_recent_messages)
            ),
        )
