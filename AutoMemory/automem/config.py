"""Configuration for AutoMemory. All API keys are read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ScoringParams:
    """All tunable scoring parameters, centralized for ablation experiments."""

    # --- Ebbinghaus decay: retention = exp(-dt_days / S_eff) ---
    # S_eff = S_base(kind) * (1 + rho*ln(1+access_count)) * (1 + a*importance + b*max(utility,0))
    s_base: dict[str, float] = field(
        default_factory=lambda: {"fact": 14.0, "experience": 30.0, "summary": 30.0}
    )
    self_source_multiplier: float = 1.5  # source='self' memories decay slower
    rho: float = 0.3   # access-count reinforcement (PowerMem's reinforcement_factor)
    a: float = 1.0     # importance contribution to strength
    b: float = 1.0     # utility contribution to strength (ablation: set 0)

    # --- hybrid relevance = w_vec*cos_norm + w_bm25*bm25_sigmoid ---
    w_vec: float = 0.7
    w_bm25: float = 0.3
    bm25_sigmoid_center: float = 5.0
    bm25_sigmoid_scale: float = 2.0

    # --- priority = (lam + (1-lam)*retention) * (1 + alpha*importance + beta*utility) ---
    lam: float = 0.2     # decay floor; lam=1 disables time decay (ablation)
    alpha: float = 0.6   # importance weight
    beta: float = 0.8    # utility weight; beta=0 disables feedback (ablation)
    # bi-temporal: a superseded fact (valid_to set) stays retrievable for history
    # questions but is down-weighted so current truth ranks above it.
    superseded_penalty: float = 0.4  # multiply priority by this when valid_to is set

    # --- activation spreading (one hop) ---
    gamma: float = 0.35          # spread strength; gamma=0 disables spreading (ablation)
    spread_max_contribs: int = 2  # max contributions accumulated per neighbor
    spread_cap_frac: float = 0.5  # contribution capped at this fraction of source score

    # --- final fusion after rerank ---
    w_rerank: float = 0.7
    w_pre: float = 0.3

    # --- feedback update: utility += eta * w_rank * (signal - utility) ---
    eta: float = 0.3

    # --- maintenance (soft-forget) ---
    forget_retention_threshold: float = 0.05

    # --- knowledge-update detection: which neighbors enter the reconcile
    # (ADD/UPDATE/DELETE) step. A fixed top-N quota with a similarity floor,
    # instead of a hard high threshold, so moderate-similarity contradictions
    # (e.g. "works at Acme" vs "started at Globex") still get judged. ---
    reconcile_top_n: int = 5       # consider up to this many nearest neighbors
    # Each candidate must clear this cosine floor to enter the (LLM) reconcile step.
    # Chosen by a MECHANISM probe (probe_supersession.py), not the dev overall:
    # supersession triggers are flat from 0.4->0.65 (UPDATE 228->229, superseded
    # 230->233) but collapse at 0.8 (->96), where the lost merges become duplicate
    # ADDs that dilute top-k recall (60-q overall .800->.700, multi-session .923->
    # .615). So 0.65 drops only the WASTED reconcile calls (unrelated neighbors that
    # the LLM would just ADD) — ~24% cheaper ingest with the dedup/supersession
    # behaviour intact. Logged per-run via dump_run_config.
    reconcile_floor: float = 0.65


@dataclass
class AutoMemConfig:
    # storage
    db_path: str = "automem.db"

    # providers (OpenAI-compatible endpoints)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"

    # chat LLM selection: "deepseek" (shipped default) | "mimo" (Xiaomi MiMo).
    # MiMo is a reasoning model served over an OpenAI-compatible endpoint; its
    # credentials default to the SiliconFlow open-model endpoint (see from_env).
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    mimo_api_key: str = ""
    mimo_base_url: str = "https://api.siliconflow.cn/v1"
    mimo_model: str = "XiaomiMiMo/MiMo-7B-RL"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embed_dim: int = 1024
    embed_batch_size: int = 32

    # retrieval
    recall_candidates: int = 50  # per-channel recall (vector / FTS)
    rerank_pool: int = 30        # candidates sent to the reranker
    top_k: int = 8

    # hybrid-granularity evidence expansion: after top-k facts are chosen, attach
    # the verbatim source passages behind the highest-ranked ones so the caller
    # can answer questions that need exact wording / names / dates.
    expand_sources: bool = True
    expand_top_n: int = 5          # expand sources for at most this many top memories
    source_max_chars: int = 1200   # truncate each passage to keep the prompt bounded

    # === EXPERIMENTAL / ABLATION-ONLY — INERT IN THE DEFAULT (shipped) CONFIG ===
    # The three fields below drive the "separate-quota" history subsystem, which the
    # clean read-only + as-of evaluation found to be WORSE than the default (it
    # scored .733 vs the shipped compete config's .800 on the 60-q dev; see the
    # README ablation, −6.7pp). They are kept ONLY so that ablation is reproducible
    # via `search_runner.py --separate-quota`. With the default
    # `history_separate_quota=False`, superseded history competes in the main top_k
    # pool (down-weighted by superseded_penalty) and `history_mode`/`history_quota`
    # have NO effect — do not assume the gate below runs in normal operation.
    #
    # history_mode (only consulted when history_separate_quota=True):
    #   "off"  - never attach the separate history quota
    #   "on"   - always attach it
    #   "auto" - attach only on retrospective queries (used to / before / 之前 ...),
    #            via is_retrospective() in retrieval.py
    history_mode: str = "auto"
    history_quota: int = 5
    # Default False = "compete": mingled history is first-class evidence for
    # multi-session/temporal synthesis; a separate appendix section makes the reader
    # under-use it. Flip True ONLY to reproduce the separate-quota ablation.
    history_separate_quota: bool = False

    # short-term memory window (both constraints apply)
    stm_max_items: int = 20
    stm_ttl_hours: float = 6.0

    # dedup / linking thresholds (cosine similarity)
    remember_merge_threshold: float = 0.92  # self-memory direct merge
    link_min_cos: float = 0.55
    link_max_cos: float = 0.9
    link_max_count: int = 3

    scoring: ScoringParams = field(default_factory=ScoringParams)

    @classmethod
    def from_env(cls, env_file: str | None = ".env", **overrides) -> "AutoMemConfig":
        if env_file:
            _load_env_file(env_file)
        cfg = cls(
            db_path=os.environ.get("AUTOMEM_DB_PATH", cls.db_path),
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", cls.deepseek_base_url),
            siliconflow_api_key=os.environ.get("SILICONFLOW_API_KEY", ""),
            siliconflow_base_url=os.environ.get(
                "SILICONFLOW_BASE_URL", cls.siliconflow_base_url
            ),
            llm_provider=os.environ.get("AUTOMEM_LLM_PROVIDER", cls.llm_provider),
            # MiMo defaults to the SiliconFlow open-model endpoint/key when its own
            # MIMO_* vars are unset, since that is the endpoint AutoMem already uses.
            mimo_api_key=os.environ.get("MIMO_API_KEY")
            or os.environ.get("SILICONFLOW_API_KEY", ""),
            mimo_base_url=os.environ.get("MIMO_BASE_URL")
            or os.environ.get("SILICONFLOW_BASE_URL", cls.mimo_base_url),
            mimo_model=os.environ.get("MIMO_MODEL", cls.mimo_model),
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"Unknown config field: {key}")
            setattr(cfg, key, value)
        return cfg


def _load_env_file(path: str) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing env vars take precedence."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
