"""
Terminal nodes that produce the final AgentResponse for each path.

Each returns a dict with a single "final_response" key so downstream
persistence knows the run is complete.
"""
from __future__ import annotations
from typing import Any, Dict, List

from src.models.reranker import sigmoid
from src.state import AgentState
from src.utils.logger import log_event
from src.utils.metadata_store import store_interaction


# Weights for the retrieval-based confidence composite (answerable path).
# Reranker gets the larger share because it directly measures query↔chunk
# relevance with a cross-encoder that sees both texts jointly. The hybrid
# score is a shallower BM25+vector signal that's useful as a secondary

_CONF_W_RERANK = 0.6
_CONF_W_HYBRID = 0.4


def _sources_from_manifest(state: AgentState) -> List[Dict[str, str]]:
    manifest = state.get("source_manifest") or []
    return [
        {"source_id": m["source_id"], "passage": m["chunk_id"]}
        for m in manifest
    ]


def _answerable_confidence(state: AgentState) -> float:
    """Composite confidence for the answerable path.
    confidence = 0.7 * sigmoid(top rerank_score) + 0.3 * sigmoid(top hybrid_score)
    """
    reranked = state.get("reranked") or []
    if not reranked:
        return 0.0
    top = reranked[0]
    rerank_component = sigmoid(float(top.get("rerank_score", 0.0)))
    hybrid_component = sigmoid(float(top.get("hybrid_score", 0.0)))
    composite = _CONF_W_RERANK * rerank_component + _CONF_W_HYBRID * hybrid_component
    # Clamp — sigmoid outputs are always in (0,1)
    # clamping protects against future weight changes that might sum to != 1.0.
    return max(0.0, min(1.0, round(composite, 3)))


def finalize_answer_node(state: AgentState) -> dict:
    """Answerable path — verification succeeded."""
    state.setdefault("node_trace", []).append("finalize_answer")
    aj = state.get("answer_json") or {}
    confidence = _answerable_confidence(state)
    resp = {
        "classification": "answerable",
        "answer": aj.get("answer", ""),
        "sources": _sources_from_manifest(state),
        "confidence": confidence,
        "requires_human": False,
        "reason": aj.get("reason", "answered from knowledge base"),
    }
    reranked = state.get("reranked") or []
    top = reranked[0] if reranked else {}
    log_event(
        "finalize_answer", "done",
        confidence=confidence,
        rerank_component=round(sigmoid(float(top.get("rerank_score", 0.0))), 3) if reranked else None,
        hybrid_component=round(sigmoid(float(top.get("hybrid_score", 0.0))), 3) if reranked else None,
    )
    out = {"final_response": resp}
    store_interaction({**state, **out})
    return out


def finalize_clarification_node(state: AgentState) -> dict:
    state.setdefault("node_trace", []).append("finalize_clarification")
    resp = {
        "classification": "requires_clarification",
        "answer": (
            "I need a bit more information to help. "
            "Could you share the exact feature or workflow you're using, "
            "any error messages, and what you expected to happen?"
        ),
        "sources": _sources_from_manifest(state),
        "confidence": float(state.get("top_score", 0.0)),
        "requires_human": False,
        "reason": "query was too ambiguous to answer confidently",
    }
    log_event("finalize_clarification", "done")
    out = {"final_response": resp}
    store_interaction({**state, **out})
    return out


def finalize_escalation_node(state: AgentState) -> dict:
    state.setdefault("node_trace", []).append("finalize_escalation")
    resp = {
        "classification": "requires_escalation",
        "answer": (
            "This request needs a human agent. I've flagged it for the "
            "support team along with your original question."
        ),
        "sources": [],
        "confidence": 1.0,   # We are confident this needs a human.
        "requires_human": True,
        "reason": "requires actions or policies outside the automated agent's scope",
    }
    log_event("finalize_escalation", "done")
    out = {"final_response": resp}
    store_interaction({**state, **out})
    return out


def finalize_out_of_scope_node(state: AgentState) -> dict:
    state.setdefault("node_trace", []).append("finalize_out_of_scope")
    resp = {
        "classification": "out_of_scope",
        "answer": (
            "That request is outside what I can help with. "
            "I can answer questions about the product's features, settings, "
            "and troubleshooting."
        ),
        "sources": [],
        "confidence": float(state.get("top_score", 0.0)),
        "requires_human": False,
        "reason": "query is not covered by the knowledge base",
    }
    log_event("finalize_out_of_scope", "done")
    out = {"final_response": resp}
    store_interaction({**state, **out})
    return out


def finalize_safe_failure_node(state: AgentState) -> dict:
    """Reached when verification fails twice."""
    state.setdefault("node_trace", []).append("finalize_safe_failure")
    resp: Dict[str, Any] = {
        "classification": "safe_failure",
        "answer": (
            "I couldn't produce a verified answer for this. "
            "I'm handing this off to a human support agent."
        ),
        "sources": _sources_from_manifest(state),
        "confidence": 0.0,
        "requires_human": True,
        "reason": "verification failed after retry: "
                  + "; ".join(state.get("verification_errors", []) or ["unknown"]),
    }
    log_event("finalize_safe_failure", "done",
              errors=state.get("verification_errors", []))
    out = {"final_response": resp}
    store_interaction({**state, **out})
    return out