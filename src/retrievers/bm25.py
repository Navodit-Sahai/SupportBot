"""BM25 keyword retriever built over pre-chunked documents."""
from __future__ import annotations
import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.utils.chunker import Chunk


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # Normalize to 0..1 so we can blend with cosine similarity later.
        s_max = float(scores.max()) if len(scores) else 0.0
        normed = [float(s) / s_max if s_max > 0 else 0.0 for s in scores]
        ranked = sorted(
            zip(self.chunks, normed), key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_k]
