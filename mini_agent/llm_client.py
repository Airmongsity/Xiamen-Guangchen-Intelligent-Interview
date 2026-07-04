"""Thin wrapper over an OpenAI-compatible Chat Completions endpoint.

Kept deliberately small: it only owns credentials, the model name, and retry on
transient errors. The agent loop (``runtime.py``) owns all orchestration logic so
that ``LLMClient`` can be swapped for a fake in tests.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from .config import Settings

logger = logging.getLogger("mini_agent.llm")


class LLMClient:
    def __init__(self, settings: Settings | None = None, *, timeout: float = 60.0,
                 max_retries: int = 5):
        self.settings = settings or Settings.from_env()
        if not self.settings.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Put it in a root .env file or export it."
            )
        self.model = self.settings.model
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=timeout,
        )

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             tool_choice: str = "auto", temperature: float = 0.2) -> Any:
        """Call chat.completions with exponential backoff on transient errors.

        Returns the raw SDK response; the caller reads ``resp.choices[0].message``.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APITimeoutError, APIError) as err:
                last_err = err
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s -- retrying in %ss",
                    attempt + 1, self.max_retries, err, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_err}"
        )
