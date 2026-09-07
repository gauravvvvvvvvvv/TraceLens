from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryMode(str, Enum):
    none = "none"
    short = "short"
    long = "long"


class Verdict(str, Enum):
    supported = "supported"
    refuted = "refuted"
    mixed = "mixed"
    insufficient_evidence = "insufficient_evidence"


class InvestigationRequest(BaseModel):
    subject: str = Field(min_length=2, max_length=300)
    question: str = Field(min_length=5, max_length=2000)
    subject_type: Literal["company", "person", "claim"] = "claim"
    memory_mode: MemoryMode = MemoryMode.short
    max_steps: int = Field(default=8, ge=2, le=20)
    max_tool_calls: int = Field(default=12, ge=2, le=40)


class AtomicClaim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    required: bool = True


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    query: str


class RetrievedDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    url: str
    final_url: str
    title: str
    text: str
    content_type: str
    content_hash: str
    retrieved_at: datetime = Field(default_factory=utc_now)


class EvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    atomic_claim_id: str
    stance: Literal["supports", "refutes", "neutral"]
    excerpt: str
    interpretation: str
    source_url: str
    source_title: str
    source_type: str = "unknown"
    source_reliability: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    independence_group: str
    excerpt_verified: bool = False


class Conflict(BaseModel):
    claim_id: str
    description: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    refuting_evidence_ids: list[str] = Field(default_factory=list)
    resolved: bool = False


class ModelUsage(BaseModel):
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int
    estimated_cost_usd: float | None = None


class TraceEvent(BaseModel):
    sequence: int
    timestamp: datetime = Field(default_factory=utc_now)
    stage: str
    action: str
    summary: str
    tool: str | None = None
    latency_ms: int | None = None
    confidence: float | None = None
    verdict: Verdict | None = None
    error: str | None = None


class InvestigationReport(BaseModel):
    investigation_id: str
    subject: str
    question: str
    status: Literal["queued", "running", "completed", "failed"]
    verdict: Verdict = Verdict.insufficient_evidence
    confidence: float = Field(default=0, ge=0, le=1)
    summary: str = ""
    atomic_claims: list[AtomicClaim] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    usage: list[ModelUsage] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class PlanOutput(BaseModel):
    atomic_claims: list[str] = Field(min_length=1, max_length=8)
    search_queries: list[str] = Field(min_length=1, max_length=6)
    evidence_requirements: list[str] = Field(default_factory=list)


class EvidenceExtractionOutput(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)


class ReflectionOutput(BaseModel):
    verdict: Verdict
    summary: str
    unresolved_questions: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list, max_length=4)
    limitations: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    id: str
    subject: str
    subject_type: Literal["company", "person", "claim"]
    claim: str
    expected_verdict: Verdict
    required_facts: list[str]
    notes: str = ""

