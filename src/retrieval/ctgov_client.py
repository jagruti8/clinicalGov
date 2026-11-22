"""ClinicalTrials.gov v2 API client: live search, pagination, lookup by NCT ID,
retry with backoff, and an optional on-disk snapshot cache for reproducible runs.

urllib is used instead of httpx because the site's WAF blocked httpx.

Snapshot cache (for reproducible evaluations):
  CTGOV_CACHE_MODE = off (default) | readwrite | readonly
  CTGOV_CACHE_DIR  = ./ctgov_cache
In readonly mode a missing snapshot raises, so a run never silently hits the live API.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
NCT_PATTERN = re.compile(r"NCT\d{8}")
RETRYABLE_HTTP = {429, 500, 502, 503, 504}

_FIELDS = "|".join([
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.designModule.phases",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.armsInterventionsModule.interventions",
])

_PHASE_MAP = {"PHASE1": "1", "PHASE2": "2", "PHASE3": "3", "PHASE4": "4"}
_STATUS_MAP = {"RECRUITING": "rec", "COMPLETED": "com", "NOT_YET_RECRUITING": "not-rec"}

_HEADERS = {
    "User-Agent": "clinicalGov-rag-pipeline/1.0 (research use)",
    "Accept": "application/json",
}

_sleep = time.sleep  # patched in tests


class SnapshotMiss(RuntimeError):
    """readonly cache mode and no snapshot exists for this request."""


def _coerce_str(val) -> Optional[str]:
    if isinstance(val, list):
        val = val[0] if val else None
    return str(val).strip() if val else None


def build_query_params(
    condition: str = None,
    intervention: str = None,
    phase: str = None,
    status: str = None,
    page_size: int = 15,
    term: str = None,
    sponsor: str = None,
) -> dict:
    condition = _coerce_str(condition)
    intervention = _coerce_str(intervention)
    phase = _coerce_str(phase)
    status = _coerce_str(status)
    term = _coerce_str(term)
    sponsor = _coerce_str(sponsor)

    params: dict = {"format": "json", "pageSize": page_size, "fields": _FIELDS}
    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention
    if term:
        params["query.term"] = term
    if sponsor:
        params["query.spons"] = sponsor

    agg_filters = []
    if phase and phase in _PHASE_MAP:
        agg_filters.append(f"phase:{_PHASE_MAP[phase]}")
    if status and status in _STATUS_MAP:
        agg_filters.append(f"status:{_STATUS_MAP[status]}")
    if agg_filters:
        params["aggFilters"] = ",".join(agg_filters)
    return params


def _cache_mode() -> str:
    mode = os.environ.get("CTGOV_CACHE_MODE", "off").lower()
    return mode if mode in {"off", "readwrite", "readonly"} else "off"


def _cache_path(url: str) -> Path:
    directory = Path(os.environ.get("CTGOV_CACHE_DIR", "./ctgov_cache"))
    return directory / (hashlib.sha256(url.encode()).hexdigest() + ".json")


def _http_get_json(url: str, retries: int = 3, backoff_s: float = 2.0, timeout: int = 30) -> dict:
    """GET a JSON document, using the snapshot cache and retrying transient failures."""
    mode = _cache_mode()
    path = _cache_path(url)
    if mode in {"readwrite", "readonly"} and path.exists():
        return json.loads(path.read_text())
    if mode == "readonly":
        raise SnapshotMiss(f"No cached snapshot for {url} (CTGOV_CACHE_MODE=readonly)")

    req = urllib.request.Request(url, headers=_HEADERS)
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            if mode == "readwrite":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_HTTP and attempt < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff_s * (2 ** attempt)
                _sleep(wait)
                last_error = exc
                continue
            raise RuntimeError(f"ClinicalTrials.gov API error {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt < retries:
                _sleep(backoff_s * (2 ** attempt))
                continue
            raise RuntimeError(f"ClinicalTrials.gov unreachable: {exc}") from exc
    raise RuntimeError(f"ClinicalTrials.gov request failed: {last_error}")


def parse_study(study: dict) -> Optional[dict]:
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    nct_id = ident.get("nctId", "")
    if not nct_id:
        return None

    arms = proto.get("armsInterventionsModule", {})
    interventions = [i.get("name", "") for i in arms.get("interventions", [])]
    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle", ""),
        "status": proto.get("statusModule", {}).get("overallStatus", ""),
        "phases": proto.get("designModule", {}).get("phases", []),
        "conditions": proto.get("conditionsModule", {}).get("conditions", []),
        "interventions": interventions,
        "summary": proto.get("descriptionModule", {}).get("briefSummary", ""),
        "eligibility": proto.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
    }


def fetch_trials(query_params: dict, max_results: Optional[int] = None) -> list[dict]:
    """Search trials. Without max_results: one page (pageSize). With it: follow
    nextPageToken until max_results trials are collected or results run out."""
    params = dict(query_params)
    trials: list[dict] = []
    seen: set[str] = set()

    while True:
        if max_results is not None:
            params["pageSize"] = max(1, min(int(params.get("pageSize", 25)), max_results - len(trials)))
        data = _http_get_json(CTGOV_BASE_URL + "?" + urllib.parse.urlencode(params))

        for study in data.get("studies", []):
            trial = parse_study(study)
            if trial and trial["nct_id"] not in seen:
                seen.add(trial["nct_id"])
                trials.append(trial)

        token = data.get("nextPageToken")
        if max_results is None or not token or len(trials) >= max_results:
            break
        params["pageToken"] = token

    return trials[:max_results] if max_results is not None else trials


def get_trial(nct_id: str) -> Optional[dict]:
    """Fetch one trial by NCT ID. Returns None if it does not exist."""
    nct_id = nct_id.strip().upper()
    if not NCT_PATTERN.fullmatch(nct_id):
        raise ValueError(f"Not a valid NCT ID: {nct_id!r}")
    url = f"{CTGOV_BASE_URL}/{nct_id}?" + urllib.parse.urlencode({"format": "json", "fields": _FIELDS})
    try:
        data = _http_get_json(url)
    except RuntimeError as exc:
        if "error 404" in str(exc):
            return None
        raise
    return parse_study(data)


def format_trial_text(trial: dict, eligibility_chars: int = 500) -> str:
    phases = ", ".join(trial.get("phases", [])) or "Not specified"
    conditions = ", ".join(trial.get("conditions", [])) or "Not specified"
    interventions = ", ".join(trial.get("interventions", [])) or "Not specified"
    eligibility = (trial.get("eligibility") or "")[:eligibility_chars]

    return (
        f"Trial ID: {trial['nct_id']}\n"
        f"Title: {trial.get('title', '')}\n"
        f"Status: {trial.get('status', '')}\n"
        f"Phase: {phases}\n"
        f"Conditions: {conditions}\n"
        f"Interventions: {interventions}\n"
        f"Summary: {trial.get('summary', '')}\n"
        f"Eligibility: {eligibility or 'Not specified'}"
    )


def trial_id_from_text(text: str) -> Optional[str]:
    """NCT ID of a formatted trial document (see format_trial_text)."""
    match = re.match(r"\s*Trial ID: (NCT\d{8})", text or "")
    return match.group(1) if match else None
