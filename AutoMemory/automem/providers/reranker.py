"""SiliconFlow reranker. The OpenAI SDK has no rerank endpoint, so we POST directly."""

from __future__ import annotations

import time

import httpx

from ..config import AutoMemConfig


class SiliconFlowReranker:
    def __init__(self, config: AutoMemConfig, *, retries: int = 3):
        if not config.siliconflow_api_key:
            raise ValueError("SILICONFLOW_API_KEY is not set")
        self._url = config.siliconflow_base_url.rstrip("/") + "/rerank"
        self._headers = {"Authorization": f"Bearer {config.siliconflow_api_key}"}
        self._model = config.rerank_model
        self._retries = retries

    def rerank(self, query: str, documents: list[str], *, top_n: int | None = None) -> list[tuple[int, float]]:
        """Return [(document_index, relevance_score), ...] sorted by score desc."""
        if not documents:
            return []
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                resp = httpx.post(
                    self._url,
                    headers=self._headers,
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n or len(documents),
                        "return_documents": False,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                results = resp.json()["results"]
                return [(r["index"], r["relevance_score"]) for r in results]
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                # retry transient server/network errors; client errors fail fast
                if (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code < 500
                ):
                    raise
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        raise last_exc
