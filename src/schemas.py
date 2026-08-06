"""
Pydantic schemas for the structured response.
Used for validation in the verification node.
"""
from typing import List, Literal
from pydantic import BaseModel, Field


class SourceRefModel(BaseModel):
    source_id: str
    passage: str


class AgentResponse(BaseModel):
    classification: Literal[
        "answerable", "requires_clarification", "requires_escalation",
        "out_of_scope", "safe_failure",
    ]
    answer: str
    sources: List[SourceRefModel] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human: bool = False
    reason: str = ""