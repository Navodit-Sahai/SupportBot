"""Dense retriever: cosine similarity over sentence-transformer embeddings."""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from src.models.embedder import embed
from src.utils.chunker import Chunk


class VectorRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        if chunks:
            self._matrix = embed([c.text for c in chunks])  # (N, D), L2-normed
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = embed([query])                       # (1, D)
        sims = (self._matrix @ q.T).squeeze(-1)  # (N,)
        # Cosine sim of normed vectors is in [-1,1]; map to [0,1].
        sims = (sims + 1.0) / 2.0
        order = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in order]
