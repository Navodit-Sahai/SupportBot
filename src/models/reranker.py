"""Cross-encoder reranker. Scores (query, passage) pairs."""
from __future__ import annotations
import time
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from config import MODEL
from src.models.device import resolve_device
from src.utils.logger import log_event


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    device = resolve_device()
    t0 = time.perf_counter()
    model = CrossEncoder(
        MODEL.reranker_model,
        revision=MODEL.reranker_revision,
        device=device,
    )
    log_event("reranker", "loaded",
              model=MODEL.reranker_model, device=device,
              load_ms=round((time.perf_counter() - t0) * 1000, 1))
    return model


def rerank_scores(query: str, passages: List[str]) -> np.ndarray:
    """Return raw cross-encoder logits for each (query, passage) pair."""
    if not passages:
        return np.zeros(0, dtype=np.float32)
    model = get_reranker()
    pairs: List[Tuple[str, str]] = [(query, p) for p in passages]
    return np.asarray(model.predict(pairs, show_progress_bar=False), dtype=np.float32)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))
