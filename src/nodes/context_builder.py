"""
Build the prompt context from reranked chunks:
  - Prefix each with an [S1], [S2] ... label the LLM must cite
  - Trim to approximate token budget (word-count proxy)
"""
from __future__ import annotations
from typing import List

from config import GENERATION
from src.state import AgentState, RetrievedChunk
from src.utils.logger import log_event


def _approx_tokens(text: str) -> int:
    # 1 token ~ 0.75 words for English. Cheap proxy — no tokenizer needed here.
    return int(len(text.split()) / 0.75)


def _format(chunks: List[RetrievedChunk]) -> tuple[str, list[dict]]:
    lines: list[str] = []
    manifest: list[dict] = []
    budget = GENERATION.context_token_budget
    used = 0
    for i, c in enumerate(chunks, 1):
        label = f"S{i}"
        body = c["text"].strip()
        block = f"[{label}]\n{body}"
        cost = _approx_tokens(block)
        if used + cost > budget:
            break
        lines.append(block)
        used += cost
        manifest.append({
            "label": label,
            "source_id": c["source_id"],
            "chunk_id": c["chunk_id"],
            "source_type": c["source_type"],
            "rerank_score": c["rerank_score"],
        })
    return "\n\n".join(lines), manifest


def context_builder_node(state: AgentState) -> dict:
    state.setdefault("node_trace", []).append("context_builder")
    context, manifest = _format(state.get("reranked", []))
    log_event("context_builder", "built",
              blocks=len(manifest), approx_tokens=_approx_tokens(context))
    return {"context": context, "source_manifest": manifest}