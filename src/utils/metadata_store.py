"""
Interaction metadata store.

Deliberately separate from the retrieval corpus — we NEVER feed AI-generated
answers back into the KB. This file is for analytics, debugging, and future
supervised fine-tuning only.
"""
from __future__ import annotations
import json
import time
from typing import Any, Dict

from config import METADATA_STORE_PATH


def store_interaction(state: Dict[str, Any]) -> None:
    """Append a single interaction record as JSON Lines."""
    record = {
        "ts": time.time(),
        "query": state.get("query"),
        "classification": state.get("classification"),
        "top_score": state.get("top_score"),
        "retrieved": [
            {"source_id": c["source_id"], "rerank_score": c.get("rerank_score", 0.0)}
            for c in state.get("reranked", [])
        ],
        "answer": (state.get("final_response") or {}).get("answer"),
        "confidence": (state.get("final_response") or {}).get("confidence"),
        "verification_passed": state.get("verification_passed"),
        "verification_errors": state.get("verification_errors", []),
        "retry_count": state.get("retry_count", 0),
        "node_trace": state.get("node_trace", []),
        "latencies_ms": state.get("latencies_ms", {}),
    }
    METADATA_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_STORE_PATH.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
