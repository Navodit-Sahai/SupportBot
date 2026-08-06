from langgraph.graph import StateGraph, END

from config import VERIFICATION, GRAPH
from src.state import AgentState

from src.nodes.triage import triage_node
from src.nodes.retrieval import retrieval_node
from src.nodes.context_builder import context_builder_node
from src.nodes.generation import generation_node
from src.nodes.verification import verification_node
from src.nodes.finalize import (
    finalize_answer_node,
    finalize_clarification_node,
    finalize_escalation_node,
    finalize_out_of_scope_node,
    finalize_safe_failure_node,
)


# --- Conditional routers ------------------------------------------------

def route_after_triage(state: AgentState) -> str:
    label = state.get("classification", "clarification")
    return {
        "answerable": "retrieval",
        "clarification": "finalize_clarification",
        "escalation": "finalize_escalation",
        "out_of_scope": "finalize_out_of_scope",
    }.get(label, "finalize_clarification")


def route_after_retrieval(state: AgentState) -> str:
    # Retrieval may have downgraded classification to out_of_scope.
    if state.get("classification") == "out_of_scope":
        return "finalize_out_of_scope"
    return "context_builder"


def route_after_verification(state: AgentState) -> str:
    if state.get("verification_passed"):
        return "finalize_answer"
    if state.get("retry_count", 0) < VERIFICATION.max_retries:
        return "retry_generation"
    return "finalize_safe_failure"


def retry_bump(state: AgentState) -> dict:
    """Deterministic counter bump — separate node so the loop is visible in traces."""
    state.setdefault("node_trace", []).append("retry_bump")
    return {"retry_count": state.get("retry_count", 0) + 1}


# --- Graph builder ------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("triage", triage_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("context_builder", context_builder_node)
    g.add_node("generation", generation_node)
    g.add_node("verification", verification_node)
    g.add_node("retry_bump", retry_bump)

    g.add_node("finalize_answer", finalize_answer_node)
    g.add_node("finalize_clarification", finalize_clarification_node)
    g.add_node("finalize_escalation", finalize_escalation_node)
    g.add_node("finalize_out_of_scope", finalize_out_of_scope_node)
    g.add_node("finalize_safe_failure", finalize_safe_failure_node)

    g.set_entry_point("triage")

    g.add_conditional_edges("triage", route_after_triage, {
        "retrieval": "retrieval",
        "finalize_clarification": "finalize_clarification",
        "finalize_escalation": "finalize_escalation",
        "finalize_out_of_scope": "finalize_out_of_scope",
    })

    g.add_conditional_edges("retrieval", route_after_retrieval, {
        "context_builder": "context_builder",
        "finalize_out_of_scope": "finalize_out_of_scope",
    })

    g.add_edge("context_builder", "generation")
    g.add_edge("generation", "verification")

    g.add_conditional_edges("verification", route_after_verification, {
        "finalize_answer": "finalize_answer",
        "retry_generation": "retry_bump",
        "finalize_safe_failure": "finalize_safe_failure",
    })
    # Retry path — bump the counter, regenerate, re-verify.
    g.add_edge("retry_bump", "generation")

    for terminal in [
        "finalize_answer", "finalize_clarification",
        "finalize_escalation", "finalize_out_of_scope",
        "finalize_safe_failure",
    ]:
        g.add_edge(terminal, END)

    # Recursion limit is our hard loop guard even if routing had a bug.
    return g.compile().with_config(recursion_limit=GRAPH.max_node_visits)
