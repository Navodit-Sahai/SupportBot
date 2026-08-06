"""
Central configuration.

All model names, thresholds, paths, and knobs live here.
Change models / thresholds here — never in the code.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ---------- Paths ----------
ROOT_DIR: Path = Path(__file__).parent.resolve()
DATA_DIR: Path = ROOT_DIR / "data"
KB_DIR: Path = DATA_DIR / "kb"
RESOLVED_CASES_PATH: Path = DATA_DIR / "resolved_cases.json"
METADATA_STORE_PATH: Path = DATA_DIR / "interaction_metadata.jsonl"
LOG_DIR: Path = ROOT_DIR / "logs"


# ---------- Models ----------
@dataclass(frozen=True)
class ModelConfig:
    # Embedding model — used for dense retrieval.
    # MiniLM-L6-v2: 384 dim, ~80MB, fast on CPU, solid quality baseline.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "main"

    # Cross-encoder reranker — MS MARCO tuned, 6-layer MiniLM.
    # ~90MB, ~50ms/pair on CPU. Second-stage precision boost.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_revision: str = "main"

    # Generation LLM — small instruction-tuned model that runs on CPU.
    # Qwen2.5-1.5B-Instruct: strong quality/size tradeoff, Apache 2.0.
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_revision: str = "main"

    # Zero-shot NLI model — used by verification node for entailment check.
    # Optional; can be disabled via VERIFICATION_USE_NLI below.
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_revision: str = "main"

    # Device: "cpu", "cuda", "mps", or "auto".
    device: str = "auto"

    # dtype for LLM. "auto" picks fp16 on GPU, fp32 on CPU.
    llm_dtype: str = "float16"


MODEL = ModelConfig()


# ---------- Retrieval ----------
@dataclass(frozen=True)
class RetrievalConfig:
    # Chunking
    chunk_strategy: str = "header"  # "header" | "fixed"
    chunk_size_tokens: int = 300     # only for "fixed"
    chunk_overlap_tokens: int = 40

    # Hybrid search
    bm25_top_k: int = 20
    vector_top_k: int = 20
    hybrid_alpha: float = 0.6   # 0.0 = pure BM25, 1.0 = pure vector
    candidates_before_rerank: int = 20
    top_k_after_rerank: int = 5

    # Source priority — added as a bias to reranker score.
    kb_priority_boost: float = 0.10
    resolved_case_boost: float = 0.0


RETRIEVAL = RetrievalConfig()


# ---------- Confidence thresholds ----------
@dataclass(frozen=True)
class ConfidenceConfig:
    # Reranker top-1 score thresholds (raw cross-encoder logits, roughly -10..+10).
    # Sigmoided into 0..1 before comparison.
    answerable_min: float = 0.55
    clarification_min: float = 0.30


CONFIDENCE = ConfidenceConfig()


# ---------- Generation ----------
@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 400
    temperature: float = 0.2
    top_p: float = 0.9
    context_token_budget: int = 2200 
    system_prompt: str = (
        "You are a technical support assistant. Answer ONLY from the provided sources. "
        "If the sources do not contain the answer, say so. "
        "Cite every claim with the source id in square brackets, e.g. [S1]. "
        "Respond in valid JSON matching the schema the caller provides."
    )


GENERATION = GenerationConfig()


# ---------- Verification ----------
@dataclass(frozen=True)
class VerificationConfig:
    require_citations: bool = True
    require_valid_json: bool = True
    verification_use_nli: bool = False   # heavy on CPU — off by default
    nli_entailment_threshold: float = 0.5
    max_retries: int = 1                 # hard cap; graph must not loop infinitely


VERIFICATION = VerificationConfig()


# ---------- Graph ----------
@dataclass(frozen=True)
class GraphConfig:
    max_node_visits: int = 15  
    log_node_transitions: bool = True


GRAPH = GraphConfig()


# ---------- Triage ----------
@dataclass(frozen=True)
class TriageConfig:
    # Keywords that force "escalation" classification.
    escalation_keywords: List[str] = field(default_factory=lambda: [
        "refund", "cancel my subscription", "speak to a human",
        "talk to agent", "billing dispute", "legal",
    ])
    # If top retrieval score is below this, treat as out-of-scope even
    # if triage LLM says answerable.
    out_of_scope_retrieval_floor: float = 0.15


TRIAGE = TriageConfig()