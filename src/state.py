from typing import TypedDict, List, Literal, Dict, Any


Classification = Literal[
    "answerable", "clarification", "escalation", "out_of_scope"
]


class RetrievedChunk(TypedDict):
    chunk_id: str
    source_id: str   
    source_type: Literal["kb", "resolved_case"]
    text: str
    bm25_score: float
    vector_score: float
    hybrid_score: float
    rerank_score: float


class SourceRef(TypedDict):
    document: str
    passage: str


class AgentState(TypedDict, total=False):
    # --- Input ---
    query: str

    # --- Triage ---
    classification: Classification
    triage_reason: str

    # --- Retrieval ---
    candidates: List[RetrievedChunk]        # after hybrid, before rerank
    reranked: List[RetrievedChunk]          # after rerank, top-k
    top_score: float                        # sigmoided top-1 rerank score

    # --- Context builder ---
    context: str
    # Manifest of source labels [S1], [S2] ... -> chunk metadata.
    source_manifest: List[Dict[str, Any]]

    # --- Generation ---
    raw_answer: str
    answer_json: Dict[str, Any]

    # --- Verification ---
    verification_passed: bool
    verification_errors: List[str]
    retry_count: int

    # --- Output ---
    final_response: Dict[str, Any]

    # --- Tracing ---
    node_trace: List[str]
    latencies_ms: Dict[str, float]


def new_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        candidates=[],
        reranked=[],
        top_score=0.0,
        source_manifest=[],
        retry_count=0,
        verification_passed=False,
        verification_errors=[],
        node_trace=[],
        latencies_ms={},
    )