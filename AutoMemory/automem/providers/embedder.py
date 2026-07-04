"""SiliconFlow embeddings via the OpenAI-compatible API."""

from __future__ import annotations

from openai import OpenAI

from ..config import AutoMemConfig


class SiliconFlowEmbedder:
    def __init__(self, config: AutoMemConfig):
        if not config.siliconflow_api_key:
            raise ValueError("SILICONFLOW_API_KEY is not set")
        self._client = OpenAI(
            api_key=config.siliconflow_api_key,
            base_url=config.siliconflow_base_url,
            max_retries=6,  # ride out 429 TPM windows (reset ~once per minute)
            timeout=60.0,
        )
        self._model = config.embed_model
        self._batch_size = config.embed_batch_size
        self.dim = config.embed_dim

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            # API preserves input order; sort by index defensively
            items = sorted(resp.data, key=lambda d: d.index)
            out.extend(item.embedding for item in items)
        return out
