"""
Chunking utilities.

Header-based splitting is preferred for markdown KB — it preserves
semantic boundaries better than fixed-token windows on short docs.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

from config import RETRIEVAL


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    source_type: str   # "kb" | "resolved_case"
    text: str


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Remove leading YAML frontmatter so metadata (document_id, tags, ...)
    doesn't leak into BM25 as ordinary tokens and cause spurious matches."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def chunk_markdown(source_id: str, text: str, source_type: str = "kb") -> List[Chunk]:
    """Split markdown by headers."""
    text = _strip_frontmatter(text)

    if RETRIEVAL.chunk_strategy == "fixed":
        return _fixed_window(source_id, text, source_type)

    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [Chunk(chunk_id=f"{source_id}#0", source_id=source_id,
                     source_type=source_type, text=text.strip())]

    chunks: List[Chunk] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chunks.append(Chunk(
                chunk_id=f"{source_id}#{i}",
                source_id=source_id,
                source_type=source_type,
                text=body,
            ))
    return chunks


def _fixed_window(source_id: str, text: str, source_type: str) -> List[Chunk]:
    """Fallback: rough word-based windowing."""
    words = text.split()
    size = RETRIEVAL.chunk_size_tokens
    overlap = RETRIEVAL.chunk_overlap_tokens
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        window = " ".join(words[i:i + size])
        if window.strip():
            chunks.append(Chunk(
                chunk_id=f"{source_id}#{i}",
                source_id=source_id,
                source_type=source_type,
                text=window,
            ))
    return chunks


def _list_to_sentences(items) -> str:
    """Join a list of short bullet-like strings into flowing prose.
    Each item becomes its own sentence so the paragraph reads naturally
    to a language model instead of like a database dump."""
    if items is None:
        return ""
    if isinstance(items, str):
        return items.strip()
    parts = [str(x).strip().rstrip(".") for x in items if str(x).strip()]
    if not parts:
        return ""
    return ". ".join(parts) + "."


def chunk_resolved_case(case: dict) -> Chunk:
    """Compose a resolved case as one natural-language paragraph.

    The old format used 'Title: ...\\nSymptoms: ...\\nResolution: ...' lines,
    which read to the LLM as a metadata record rather than prose. That made
    an in-answer citation tag like [S1] statistically unlikely because the
    tag never appeared adjacent to prose in-context. Storing (and passing)
    each case as a paragraph puts [S?] above prose in the retrieved block,
    which is a shape the model can associate with prose citation.

    Metadata that the *system* still needs (case_id, status, superseded
    reason, source_documents) either stays on the Chunk object (chunk_id,
    source_id) or is woven into the prose in a way that a human reader
    would also want to see — the superseded warning in particular is
    kept prominent and non-optional.
    """
    cid = str(case.get("case_id", case.get("id", "case")))
    status = str(case.get("status", "")).lower()

    sentences: list[str] = []

    # Superseded warning first — visible even if the block is later truncated,
    # and reads as an inline editorial note rather than a metadata field.
    if status == "superseded":
        reason = str(case.get(
            "superseded_reason", "this resolution is no longer current"
        )).strip().rstrip(".")
        sentences.append(
            f"[This case is SUPERSEDED and must not be presented as current "
            f"guidance. {reason}.]"
        )

    # Setup: what the user was experiencing.
    title = str(case.get("title", "")).strip().rstrip(".")
    symptoms_prose = _list_to_sentences(case.get("symptoms"))
    if title and symptoms_prose:
        sentences.append(f"A user reported an issue described as: {title}. {symptoms_prose}")
    elif title:
        sentences.append(f"A user reported an issue described as: {title}.")
    elif symptoms_prose:
        sentences.append(f"A user reported the following: {symptoms_prose}")

    # Legacy simple-fixture field.
    if "question" in case and not title and not symptoms_prose:
        q = str(case["question"]).strip().rstrip(".")
        sentences.append(f"A user asked: {q}.")

    # Resolution — phrased as narrative, not as a numbered checklist.
    resolution_prose = _list_to_sentences(case.get("resolution"))
    if resolution_prose:
        verb = (
            "The historical resolution was"
            if status == "superseded"
            else "The support team resolved this as follows"
        )
        sentences.append(f"{verb}: {resolution_prose}")

    # Important limit — the safety caveat, kept inline so any downstream
    # summary of the resolution cannot drop it silently.
    limit = case.get("important_limit")
    if limit:
        sentences.append(f"Important limit: {str(limit).strip()}")

    # Legacy notes field.
    if "notes" in case and case["notes"]:
        sentences.append(f"Additional notes: {str(case['notes']).strip()}")

    text = " ".join(sentences) if sentences else str(case)

    return Chunk(
        chunk_id=f"case:{cid}",
        source_id=cid,
        source_type="resolved_case",
        text=text,
    )