"""
Generation node.

Produces a structured JSON answer citing the source labels ([S1], [S2], ...)
that the context builder assigned. On a retry, the previous verification
errors are appended to the prompt as a corrective instruction.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple

from config import GENERATION
from src.models.llm import generate
from src.state import AgentState
from src.utils.logger import Timer, log_event


_USER_TEMPLATE = """Answer the support question using ONLY the numbered Sources below. Return valid JSON with EXACTLY four keys and nothing else.

HOW TO ANSWER — pick the pattern that matches the question:

• Troubleshooting question ("X stopped after Y" / "I'm seeing error Z" / "why isn't X working")
  → Look for a CASE-* source whose Symptoms match the user's situation.
  → The answer is the Resolution steps of that case, in the order given, each step ending with the [S?] tag of the case it came from.
  → If the case has an "Important limit:", include it as the final sentence (also cited) — a weak resolution without its caveat is unsafe.

• Permission / capability question ("Can role X do Y?" / "Who is allowed to Y?")
  → Look for the role definitions in the roles-and-permissions source.
  → State YES or NO plainly in the first sentence, ending with [S?].
  → If NO, name which role(s) DO have that permission, ending that sentence with [S?] too.

• Procedural / info-gathering question ("What do I need to prepare for X" / "What info should I collect before Y")
  → Look for a checklist or numbered list in the relevant KB source.
  → List the items in the order the KB gives them, each item ending with [S?].

• Explanation question ("What is X" / "How does Y work")
  → Paraphrase the KB definition in one or two sentences, each ending with [S?].

RULES that apply to every answer:

1. Every fact-bearing sentence in "answer" MUST end with a citation like [S1] or [S2] pointing to the source it came from. The [S?] tag goes INSIDE the "answer" string, not only in "sources_used".
2. If a Source is marked "SUPERSEDED" or "DO NOT PRESENT AS CURRENT GUIDANCE", IGNORE it — it is outdated and must not appear in the answer or sources_used.
3. Never invent facts. Use only what the Sources actually say.
4. If none of the Sources actually answer the question, write a single sentence saying so (no citation is needed on that sentence because it makes no factual claim), leave "sources_used" as [], and set confidence to 0.2 or lower.
5. "sources_used" must be a list of plain strings like ["S1","S2"], not objects and not dictionaries. It must contain every label that appears as [S?] inside "answer", and no others.

OUTPUT FORMAT — exactly these four keys, valid JSON, no prose before or after.

Correct shape (note the [S?] tags INSIDE the "answer" string — this is required, not optional):
{{"answer": "Read-only users cannot create API credentials [S1]. Only Admins and Developers may create them [S1].", "sources_used": ["S1"], "confidence": 0.9, "reason": "roles-and-permissions source states this directly"}}

Incorrect (missing inline [S?] — do NOT do this):
{{"answer": "Read-only users cannot create API credentials.", "sources_used": ["S1"], "confidence": 0.9, "reason": "..."}}

Sources:
{context}

Question: {query}

Remember: every fact-bearing sentence in "answer" ends with [S?]. Return JSON only.

JSON:"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CITE_RE = re.compile(r"\[S\d+\]")
_LABEL_RE = re.compile(r"^\[?(S\d+)\]?$")


def _parse_json(raw: str) -> Dict[str, Any] | None:
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _extract_labels(sources_used: Any, manifest_labels: set[str]) -> List[str]:
    """Coerce sources_used entries into '[S?]' tags, filtered by the manifest.

    Only labels the retriever actually produced are trusted. If the model
    hallucinated 'S9' but the manifest is S1..S5, S9 is dropped — this
    prevents the repair path from injecting bogus citations that the
    verifier would then reject as unknown labels.
    """
    if not isinstance(sources_used, list):
        return []
    tags: List[str] = []
    for item in sources_used:
        if not isinstance(item, str):
            continue
        m = _LABEL_RE.match(item.strip())
        if not m:
            continue
        label = m.group(1)
        if manifest_labels and label not in manifest_labels:
            continue
        tag = f"[{label}]"
        if tag not in tags:
            tags.append(tag)
    return tags


def _repair_citations(parsed: Dict[str, Any], manifest_labels: set[str]) -> Tuple[Dict[str, Any], bool]:
    """Deterministically append [S?] tags to answer sentences."""
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return parsed, False
    if _CITE_RE.search(answer):
        return parsed, False

    tags = _extract_labels(parsed.get("sources_used"), manifest_labels)
    if not tags:
        return parsed, False

    suffix = " " + " ".join(tags)
    # Split on sentence-ending punctuation. Then separate each sentence's
    # body from its terminal punctuation so we can insert [S?] BEFORE the
    # period, matching the shape shown in the prompt example.
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    rebuilt: List[str] = []
    for part in parts:
        if not part:
            continue
        m = re.match(r"^(.*?)([.!?]+)?\s*$", part, re.DOTALL)
        body = (m.group(1) if m else part).rstrip()
        punct = (m.group(2) if m else "") or ""
        if not body:
            continue
        rebuilt.append(f"{body}{suffix}{punct}")

    new_parsed = dict(parsed)
    new_parsed["answer"] = " ".join(rebuilt) if rebuilt else answer
    return new_parsed, True


def generation_node(state: AgentState) -> dict:
    state.setdefault("node_trace", []).append("generation")
    latencies = state.setdefault("latencies_ms", {})

    query = state["query"]
    context = state.get("context", "")

    messages = [
        {"role": "system", "content": GENERATION.system_prompt},
        {"role": "user", "content": _USER_TEMPLATE.format(query=query, context=context)},
    ]

    # ─── RETRY BRANCH ─────────────────────────────────────────────

    if state.get("retry_count", 0) > 0 and state.get("verification_errors"):
        prior = state.get("raw_answer", "").strip()
        if prior:
            messages.append({"role": "assistant", "content": prior})
        messages.append({
            "role": "user",
            "content": (
                "That JSON failed verification with these errors: "
                + "; ".join(state["verification_errors"])
                + ". Rewrite the SAME JSON, keeping the same answer content "
                "and the same facts, but fix the errors. Add a [S?] citation "
                "at the end of every fact-bearing sentence in the \"answer\" "
                "field (for example: \"Fact one [S1]. Fact two [S2].\"). "
                "Do NOT shorten the answer. Do NOT replace the answer "
                "text with just a citation tag like \"[S1]\" — the answer must "
                "still be full sentences. Keep all four keys "
                "(answer, sources_used, confidence, reason). "
                "Return valid JSON only."
            ),
        })

    with Timer(latencies, "generation_ms"):
        raw = generate(messages)

    parsed = _parse_json(raw)

    # ─── DETERMINISTIC CITATION REPAIR ─────────────────────────────
    # Root-cause fix for the "answer contains no citations" failure.
    # Prompt engineering alone cannot override Qwen2.5-1.5B-Instruct's
    # pretraining prior that keeps prose (answer) and metadata
    # (sources_used) in separate fields — evidence from this run: the
    # model produced correct sources_used=["S3"] but wrote a bare
    # "Only Admins can..." sentence with no [S3] inline, forcing a
    # 128s retry. Since retrieval already knows which sources support
    # this answer, injecting the label deterministically is more
    # accurate than asking the model to reproduce information we
    # already have. Only fires when sources_used is manifest-backed
    # and non-empty, so compliant outputs are untouched.
    # ──────────────────────────────────────────────────────────────
    repaired = False
    if parsed:
        manifest_labels = {m["label"] for m in (state.get("source_manifest") or [])}
        parsed, repaired = _repair_citations(parsed, manifest_labels)

    log_event("generation", "produced",
              parsed=parsed is not None, chars=len(raw),
              citations_repaired=repaired)

    return {
        "raw_answer": raw,
        "answer_json": parsed or {},
    }