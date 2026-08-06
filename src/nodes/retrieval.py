"""
Retrieval node: hybrid search -> cross-encoder rerank -> top-k.

Downgrades classification to 'out_of_scope' if nothing sufficiently relevant
was found — deterministic safety net around the triage LLM.
"""
from __future__ import annotations
import numpy as np

from config import RETRIEVAL, TRIAGE
from src.models.reranker import rerank_scores, sigmoid
from src.retrievers.corpus import get_retriever
from src.state import AgentState
from src.utils.logger import Timer, log_event


def retrieval_node(state: AgentState) -> dict:
    state.setdefault("node_trace", []).append("retrieval")
    latencies = state.setdefault("latencies_ms", {})

    query = state["query"]
    retriever = get_retriever()

    with Timer(latencies, "hybrid_ms"):
        candidates = retriever.search(query)

    log_event("retrieval", "hybrid_done", candidates=len(candidates))

    if not candidates:
        return {
            "candidates": [],
            "reranked": [],
            "top_score": 0.0,
            "classification": "out_of_scope",
        }

    # Rerank the top-N candidates.
    with Timer(latencies, "rerank_ms"):
        scores = rerank_scores(query, [c["text"] for c in candidates])

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    top = reranked[: RETRIEVAL.top_k_after_rerank]
    top_score = sigmoid(float(top[0]["rerank_score"])) if top else 0.0

    log_event("retrieval", "reranked",
              kept=len(top), top_score=round(top_score, 3),
              top_source=top[0]["source_id"] if top else None)

    updates: dict = {
        "candidates": candidates,
        "reranked": top,
        "top_score": top_score,
    }

    # Deterministic override: nothing relevant enough -> out_of_scope,
    # regardless of what triage said.
    if top_score < TRIAGE.out_of_scope_retrieval_floor:
        updates["classification"] = "out_of_scope"
        log_event("retrieval", "downgraded_to_out_of_scope",
                  top_score=round(top_score, 3))

    return updates
