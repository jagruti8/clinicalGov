"""Tool registry for the retrieval agent.

Tools are plain functions over the ClinicalTrials.gov client, described by Pydantic
argument models. Two ways to let a model pick them:
  json   - the model returns {"tool_calls": [{"name": ..., "args": {...}}]} as text
  native - the model uses the provider's tool-calling API (Ollama `tools`)
Both paths validate arguments with the same Pydantic models.
"""
import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from src.retrieval.ctgov_client import NCT_PATTERN, build_query_params

MAX_TOOL_CALLS = 3
Phase = Literal["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
Status = Literal["RECRUITING", "COMPLETED", "NOT_YET_RECRUITING"]
NCT_RE = r"^NCT\d{8}$"


class SearchTrialsArgs(BaseModel):
    condition: Optional[str] = Field(None, description="Disease or condition, e.g. 'type 2 diabetes'")
    intervention: Optional[str] = Field(None, description="Drug, treatment, or procedure")
    phase: Optional[Phase] = None
    status: Optional[Status] = None
    term: Optional[str] = Field(None, description="Free-text search across all fields")
    max_results: int = Field(25, ge=1, le=100)


class GetTrialArgs(BaseModel):
    nct_id: str = Field(pattern=NCT_RE)


class CompareTrialsArgs(BaseModel):
    nct_ids: list[str] = Field(min_length=2, max_length=5)


class GetEligibilityArgs(BaseModel):
    nct_id: str = Field(pattern=NCT_RE)


TOOLS = {
    "search_trials": (
        "Search ClinicalTrials.gov by condition, intervention, phase, status, or free text. "
        "Use for open questions like 'which trials study X'.",
        SearchTrialsArgs,
    ),
    "get_trial": (
        "Fetch one trial by NCT ID. Use when the question is about a specific trial.",
        GetTrialArgs,
    ),
    "compare_trials": (
        "Fetch 2 to 5 trials by NCT ID for comparison. Use when asked to compare named trials.",
        CompareTrialsArgs,
    ),
    "get_eligibility": (
        "Fetch a trial by NCT ID for its inclusion/exclusion criteria.",
        GetEligibilityArgs,
    ),
}

INTENT_BY_TOOL = {
    "search_trials": "search",
    "get_trial": "detail",
    "compare_trials": "compare",
    "get_eligibility": "eligibility",
}


def tool_schemas() -> list[dict]:
    """OpenAI-style function schemas (used for native tool calling and the JSON prompt)."""
    return [
        {"type": "function", "function": {
            "name": name, "description": description, "parameters": model.model_json_schema(),
        }}
        for name, (description, model) in TOOLS.items()
    ]


def tool_descriptions() -> str:
    """Compact tool list for the JSON-routing prompt."""
    lines = []
    for name, (description, model) in TOOLS.items():
        fields = ", ".join(
            f"{field}{'' if info.is_required() else '?'}" for field, info in model.model_fields.items()
        )
        lines.append(f"- {name}({fields}): {description}")
    return "\n".join(lines)


def parse_json_loose(raw: str) -> Optional[dict]:
    """Parse a JSON object from model output: tolerates code fences, <think> blocks,
    and prose around the object."""
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", raw or "", flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    for candidate in (cleaned, _first_json_object(cleaned)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find("{", start + 1)
    return None


def validate_call(name: str, args: dict) -> Optional[dict]:
    """Return {"name", "args"} with validated, normalized args, or None if invalid."""
    if name not in TOOLS or not isinstance(args, dict):
        return None
    try:
        validated = TOOLS[name][1](**args)
    except ValidationError:
        return None
    return {"name": name, "args": validated.model_dump(exclude_none=True)}


def calls_from_json(raw: str) -> tuple[list[dict], bool]:
    """Valid tool calls from JSON-routing output, and whether the JSON itself parsed."""
    parsed = parse_json_loose(raw)
    if parsed is None:
        return [], False
    candidates = parsed.get("tool_calls")
    if candidates is None and "name" in parsed:
        candidates = [parsed]
    valid = []
    for call in candidates or []:
        if isinstance(call, dict):
            checked = validate_call(call.get("name", ""), call.get("args") or call.get("arguments") or {})
            if checked:
                valid.append(checked)
    return valid[:MAX_TOOL_CALLS], True


def calls_from_native(tool_calls: list) -> list[dict]:
    """Valid tool calls from a LangChain AIMessage.tool_calls list."""
    valid = []
    for call in tool_calls or []:
        checked = validate_call(call.get("name", ""), call.get("args") or {})
        if checked:
            valid.append(checked)
    return valid[:MAX_TOOL_CALLS]


def find_nct_ids(text: str) -> list[str]:
    """NCT IDs in text, in order, de-duplicated."""
    return list(dict.fromkeys(NCT_PATTERN.findall((text or "").upper())))


def plan_covers_ids(calls: list[dict], nct_ids: list[str]) -> bool:
    """True if the planned calls fetch every NCT ID the question names."""
    covered = set()
    for call in calls:
        args = call["args"]
        if "nct_id" in args:
            covered.add(args["nct_id"].upper())
        covered.update(i.upper() for i in args.get("nct_ids", []))
    return set(nct_ids) <= covered


def nct_fix_call(question: str, nct_ids: list[str]) -> dict:
    """Deterministic call for questions that name NCT IDs."""
    if len(nct_ids) >= 2:
        return {"name": "compare_trials", "args": {"nct_ids": nct_ids[:5]}}
    if re.search(r"eligib|inclusion|exclusion|criteria", question, re.IGNORECASE):
        return {"name": "get_eligibility", "args": {"nct_id": nct_ids[0]}}
    return {"name": "get_trial", "args": {"nct_id": nct_ids[0]}}


def execute_tool(name: str, args: dict, ctgov, page_size: int = 25) -> list[dict]:
    """Run one validated tool call. `ctgov` provides fetch_trials() and get_trial()."""
    if name == "search_trials":
        parsed = SearchTrialsArgs(**args)
        params = build_query_params(
            condition=parsed.condition, intervention=parsed.intervention,
            phase=parsed.phase, status=parsed.status, term=parsed.term,
            page_size=min(page_size, parsed.max_results),
        )
        return ctgov.fetch_trials(params, max_results=parsed.max_results)

    if name in {"get_trial", "get_eligibility"}:
        parsed = GetTrialArgs(**args)
        trial = ctgov.get_trial(parsed.nct_id)
        return [trial] if trial else []

    if name == "compare_trials":
        parsed = CompareTrialsArgs(**args)
        trials = [ctgov.get_trial(nct_id) for nct_id in parsed.nct_ids]
        return [t for t in trials if t]

    raise ValueError(f"Unknown tool: {name}")
