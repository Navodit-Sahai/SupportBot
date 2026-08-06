"""
Hybrid retrieval: BM25 + dense, blended by hybrid_alpha.

Also applies the KB-priority boost so that when the KB and resolved cases
disagree, the KB wins on ties.
"""
from __future__ import annotations
from typing import Dict, List

from config import RETRIEVAL
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.vector import VectorRetriever
from src.state import RetrievedChunk
from src.utils.chunker import Chunk


class HybridRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(chunks)

    def search(self, query: str) -> List[RetrievedChunk]:
        bm25_hits = dict(
            (c.chunk_id, s) for c, s in self.bm25.search(query, RETRIEVAL.bm25_top_k)
        )
        vec_hits = dict(
            (c.chunk_id, s) for c, s in self.vector.search(query, RETRIEVAL.vector_top_k)
        )
        alpha = RETRIEVAL.hybrid_alpha

        merged: Dict[str, RetrievedChunk] = {}
        for cid in set(bm25_hits) | set(vec_hits):
            bm = bm25_hits.get(cid, 0.0)
            ve = vec_hits.get(cid, 0.0)
            hybrid = alpha * ve + (1 - alpha) * bm

            # Source priority boost — KB is trusted more than tickets.
            chunk = self._by_id[cid]
            if chunk.source_type == "kb":
                hybrid += RETRIEVAL.kb_priority_boost
            else:
                hybrid += RETRIEVAL.resolved_case_boost

            merged[cid] = RetrievedChunk(
                chunk_id=cid,
                source_id=chunk.source_id,
                source_type=chunk.source_type,   # type: ignore[typeddict-item]
                text=chunk.text,
                bm25_score=float(bm),
                vector_score=float(ve),
                hybrid_score=float(hybrid),
                rerank_score=0.0,
            )
        ranked = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[: RETRIEVAL.candidates_before_rerank]
