"""Live API smoke tests. Require DEEPSEEK_API_KEY / SILICONFLOW_API_KEY (or .env).

Run with: pytest -m api tests/test_providers_smoke.py
"""

import os

import pytest

from automem.config import AutoMemConfig

cfg = AutoMemConfig.from_env(env_file=os.path.join(os.path.dirname(__file__), "..", ".env"))

needs_siliconflow = pytest.mark.skipif(
    not cfg.siliconflow_api_key, reason="SILICONFLOW_API_KEY not set"
)
needs_deepseek = pytest.mark.skipif(
    not cfg.deepseek_api_key, reason="DEEPSEEK_API_KEY not set"
)


@pytest.mark.api
@needs_siliconflow
def test_embed_smoke():
    from automem.providers import SiliconFlowEmbedder

    emb = SiliconFlowEmbedder(cfg)
    vecs = emb.embed_batch(["你好，世界", "hello world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == cfg.embed_dim


@pytest.mark.api
@needs_siliconflow
def test_rerank_smoke():
    from automem.providers import SiliconFlowReranker

    rr = SiliconFlowReranker(cfg)
    results = rr.rerank("用户喜欢什么编程语言", ["用户最喜欢 Python", "今天天气不错"])
    assert len(results) == 2
    top_idx, top_score = results[0]
    assert top_idx == 0
    assert 0.0 <= top_score <= 1.0


@pytest.mark.api
@needs_deepseek
def test_llm_json_smoke():
    from automem.providers import DeepSeekLLM

    llm = DeepSeekLLM(cfg)
    data = llm.complete_json(
        "You output JSON only.",
        'Return {"facts": ["..."]} with one short fact extracted from: "我叫小明，住在上海"',
    )
    assert isinstance(data, dict)
    assert "facts" in data
