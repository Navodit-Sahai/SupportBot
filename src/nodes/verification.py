"""
Verification node.

Checks (in order — cheap ones first):
  1. Response is parseable JSON.
  2. Required fields exist and types are correct.
  3. Every citation label [S?] in the answer maps to a real source in the manifest.
  4. Every source_used claim is real.
  5. (Optional, off by default) NLI entailment: each sentence of the answer
     is entailed by at least one cited source chunk.

Design rule: verification must NEVER raise. Whatever the LLM produces —
missing keys, wrong types, dicts where strings were expected — becomes an
entry in verification_errors so the retry path can attempt a correction
and, failing that, the safe_failure terminal can hand off gracefully.
"""
from __future__ import annotations
import re
from typing import List

from config import VERIFICATION
from src.state import AgentState
from src.utils.logger import Timer, log_event


_CITATION_RE = re.compile(r"\[S(\d+)\]")


def _validate_schema(answer_json: dict) -> tuple[bool, List[str]]:
    errors: List[str] = []
    required = {"answer", "sources_used", "confidence", "reason"}
    missing = required - set(answer_json.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    try:
        conf = float(answer_json.get("confidence", 0.0))
        if not 0.0 <= conf <= 1.0:
            errors.append("confidence out of [0,1]")
    except (TypeError, ValueError):
        errors.append("confidence not a number")
    if not isinstance(answer_json.get("answer", ""), str):
        errors.append("answer not a string")
    # sources_used should be a list of strings — defer content check to
    # _validate_citations so we produce a single, coherent error message.
    if "sources_used" in answer_json and not isinstance(answer_json["sources_used"], list):
        errors.append("sources_used not a list")
    return (len(errors) == 0), errors


def _coerce_label(item) -> str | None:
    """Best-effort extraction of a source label like 'S1' from whatever
    the LLM decided to emit for a sources_used entry.

    Handles:
      - "S1"              -> "S1"
      - "[S1]"            -> "S1"
      - 1                 -> "S1"
      - {"label": "S1"}   -> "S1"
      - {"S1": "..."}     -> "S1"   (single-key dict, key is the label)
      - anything else     -> None
    """
    if isinstance(item, str):
        m = _CITATION_RE.search(item)
        if m:
            return f"S{m.group(1)}"
        s = item.strip().lstrip("[").rstrip("]")
        if s.startswith("S") and s[1:].isdigit():
            return s
        return None
    if isinstance(item, int):
        return f"S{item}"
    if isinstance(item, dict):
        # {"label": "S1"} pattern
        for key in ("label", "id", "source", "ref"):
            if key in item and isinstance(item[key], str):
                return _coerce_label(item[key])
        # {"S1": "..."} pattern — single-key dict where key IS the label
        if len(item) == 1:
            k = next(iter(item))
            if isinstance(k, str):
                return _coerce_label(k)
    return None


def _validate_citations(answer_json: dict, manifest: list[dict]) -> List[str]:
    errors: List[str] = []
    if not manifest:
        return errors
    valid_labels = {m["label"] for m in manifest}

    # In-text citations must be valid.
    text = str(answer_json.get("answer", ""))
    cited_in_text = {f"S{m.group(1)}" for m in _CITATION_RE.finditer(text)}
    if VERIFICATION.require_citations and not cited_in_text:
        errors.append("answer contains no citations")
    unknown = cited_in_text - valid_labels
    if unknown:
        errors.append(f"unknown citation labels: {sorted(unknown)}")

    # sources_used field: be defensive about item shape. Small models
    # frequently return list-of-dicts here despite prompt instructions.
    used_field = answer_json.get("sources_used")
    if used_field is None:
        return errors
    if not isinstance(used_field, list):
        errors.append(f"sources_used has wrong type: {type(used_field).__name__}")
        return errors

    bad_shape: list[str] = []
    unknown_labels: list[str] = []
    for item in used_field:
        label = _coerce_label(item)
        if label is None:
            bad_shape.append(repr(item)[:40])
        elif label not in valid_labels:
            unknown_labels.append(label)
    if bad_shape:
        errors.append(f"sources_used has unparseable entries: {bad_shape}")
    if unknown_labels:
        errors.append(f"sources_used has unknown labels: {sorted(set(unknown_labels))}")
    return errors


def _nli_check(answer_text: str, manifest: list[dict], reranked: list[dict]) -> List[str]:
    """Optional entailment check using a cross-encoder NLI model."""
    from sentence_transformers import CrossEncoder
    from config import MODEL

    nli = CrossEncoder(MODEL.nli_model, revision=MODEL.nli_revision)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    errors: List[str] = []
    id_to_text = {c["chunk_id"]: c["text"] for c in reranked}
    label_to_chunk = {m["label"]: id_to_text.get(m["chunk_id"], "") for m in manifest}
    for sent in sentences:
        cited = _CITATION_RE.findall(sent)
        if not cited:
            continue
        premises = [label_to_chunk[f"S{n}"] for n in cited if f"S{n}" in label_to_chunk]
        if not premises:
            continue
        pairs = [(p, sent) for p in premises]
        logits = nli.predict(pairs, show_progress_bar=False)
        import numpy as np
        entail = np.asarray(logits)[:, 1] if getattr(logits, "ndim", 1) > 1 else [0.0]
        if float(max(entail)) < VERIFICATION.nli_entailment_threshold:
            errors.append(f"unsupported claim: {sent[:60]}...")
    return errors


def verification_node(state: AgentState) -> dict:
    state.setdefault("node_trace", []).append("verification")
    latencies = state.setdefault("latencies_ms", {})
    errors: List[str] = []

    with Timer(latencies, "verification_ms"):
        answer_json = state.get("answer_json") or {}
        manifest = state.get("source_manifest") or []


        try:
            if VERIFICATION.require_valid_json and not answer_json:
                errors.append("response was not valid JSON")
            else:
                ok, schema_errs = _validate_schema(answer_json)
                errors.extend(schema_errs)
                errors.extend(_validate_citations(answer_json, manifest))
                if not errors and VERIFICATION.verification_use_nli:
                    errors.extend(_nli_check(
                        str(answer_json.get("answer", "")),
                        manifest,
                        state.get("reranked", []),
                    ))
        except Exception as e:  
            errors.append(f"verification internal error: {type(e).__name__}: {e}")

    passed = len(errors) == 0
    log_event("verification", "checked",
              passed=passed, errors=errors[:3])

    return {
        "verification_passed": passed,
        "verification_errors": errors,
    }