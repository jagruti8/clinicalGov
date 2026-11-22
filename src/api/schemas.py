"""Request and response schemas for the REST API (published as OpenAPI at /openapi.json)."""
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.tools import Phase, Status


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = {}


class Evidence(BaseModel):
    claim: str
    nct_id: str
    quote: str
    supported: bool
    reason: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    intent: Optional[str] = None
    confidence: Optional[float] = Field(None, description="Mean pairwise agreement across models; null with one model")
    agreement_models: int = 0
    citations: list[str] = []
    verified_citations: list[str] = []
    unverified_citations: list[str] = []
    evidence_support_rate: Optional[float] = None
    evidence: list[Evidence] = []
    tool_calls_llm: list[ToolCall] = []
    tool_runs: list[dict[str, Any]] = []
    plan_source: Optional[str] = None
    models_used: dict[str, str] = {}
    critique: dict[str, Any] = {}
    revision_count: int = 0
    reformulation_count: int = 0
    timings: dict[str, float] = {}
    total_latency_s: Optional[float] = None
    warnings: list[str] = []
    errors: list[str] = []


class PlanResponse(BaseModel):
    question: str
    intent: Optional[str] = None
    tool_calls_llm: list[ToolCall] = []
    tool_plan: list[ToolCall] = []
    plan_source: Optional[str] = None
    plan_json_ok: bool = False
    models_used: dict[str, str] = {}
    errors: list[str] = []


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    condition: Optional[str] = None
    intervention: Optional[str] = None
    phase: Optional[Phase] = None
    status: Optional[Status] = None
    term: Optional[str] = None
    max_results: int = Field(25, ge=1, le=100)


class RetrievedTrial(BaseModel):
    nct_id: str
    score: float
    rank_score: float
    text: str


class RetrieveResponse(BaseModel):
    n_fetched: int
    results: list[RetrievedTrial]


class ToolRequest(BaseModel):
    arguments: dict[str, Any] = {}


class ToolResponse(BaseModel):
    tool: str
    n_trials: int
    trials: list[dict[str, Any]]


class JudgeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    answer: str = Field(min_length=1)
    contexts: list[str] = Field(min_length=1)


class JudgeResponse(BaseModel):
    faithfulness: Optional[float] = None
    citation_accuracy: Optional[float] = None
    faithfulness_reasoning: str = ""
    citation_reasoning: str = ""
    error: Optional[str] = None
