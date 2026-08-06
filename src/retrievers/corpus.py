"""
Load the KB + resolved cases from disk and build the retriever exactly once.
"""
from __future__ import annotations
import json
from functools import lru_cache
from typing import List

from config import KB_DIR, RESOLVED_CASES_PATH
from src.retrievers.hybrid import HybridRetriever
from src.utils.chunker import Chunk, chunk_markdown, chunk_resolved_case
from src.utils.logger import log_event


def _load_kb_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    if not KB_DIR.exists():
        return chunks
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(source_id=path.name, text=text, source_type="kb"))
    return chunks


def _load_case_chunks() -> List[Chunk]:
    if not RESOLVED_CASES_PATH.exists():
        return []
    data = json.loads(RESOLVED_CASES_PATH.read_text(encoding="utf-8"))
    # Real export is an object like {"product", "exported_at", "notes", "cases": [...]}.
    # Fall back to a bare list for older/simple fixtures.
    cases = data["cases"] if isinstance(data, dict) and "cases" in data else data
    return [chunk_resolved_case(c) for c in cases]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    kb = _load_kb_chunks()
    cases = _load_case_chunks()
    all_chunks = kb + cases
    log_event("corpus", "loaded",
              kb_chunks=len(kb), case_chunks=len(cases), total=len(all_chunks))
    return HybridRetriever(all_chunks)