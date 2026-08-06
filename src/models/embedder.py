"""Sentence-transformers embedding model, loaded once."""
from __future__ import annotations
import time
from functools import lru_cache
from typing import List

import numpy as np

from config import MODEL
from src.models.device import resolve_device
from src.utils.logger import log_event


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    device = resolve_device()
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL.embedding_model,
        revision=MODEL.embedding_revision,
        device=device,
    )
    log_event("embedder", "loaded",
              model=MODEL.embedding_model, device=device,
              load_ms=round((time.perf_counter() - t0) * 1000, 1))
    return model


def embed(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
