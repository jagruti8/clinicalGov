import json
import urllib.error
import urllib.parse
import urllib.request

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

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


def _coerce_str(val) -> str | None:
    if isinstance(val, list):
        val = val[0] if val else None
    return str(val).strip() if val else None


def build_query_params(
    condition: str = None,
    intervention: str = None,
    phase: str = None,
    status: str = None,
    page_size: int = 15,
) -> dict:
    condition = _coerce_str(condition)
    intervention = _coerce_str(intervention)
    phase = _coerce_str(phase)
    status = _coerce_str(status)

    params: dict = {"format": "json", "pageSize": page_size, "fields": _FIELDS}
    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention

    agg_filters = []
    if phase and phase in _PHASE_MAP:
        agg_filters.append(f"phase:{_PHASE_MAP[phase]}")
    if status and status in _STATUS_MAP:
        agg_filters.append(f"status:{_STATUS_MAP[status]}")
    if agg_filters:
        params["aggFilters"] = ",".join(agg_filters)
    return params


_HEADERS = {
    "User-Agent": "clinicalGov-rag-pipeline/1.0 (research; contact: pushpak.nitrkl@gmail.com)",
    "Accept": "application/json",
}


def fetch_trials(query_params: dict) -> list[dict]:
    url = CTGOV_BASE_URL + "?" + urllib.parse.urlencode(query_params)
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ClinicalTrials.gov API error {exc.code}: {exc.reason}") from exc

    trials = []
    for study in data.get("studies", []):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        nct_id = ident.get("nctId", "")
        if not nct_id:
            continue

        arms = proto.get("armsInterventionsModule", {})
        interventions = [i.get("name", "") for i in arms.get("interventions", [])]

        trials.append({
            "nct_id": nct_id,
            "title": ident.get("briefTitle", ""),
            "status": proto.get("statusModule", {}).get("overallStatus", ""),
            "phases": proto.get("designModule", {}).get("phases", []),
            "conditions": proto.get("conditionsModule", {}).get("conditions", []),
            "interventions": interventions,
            "summary": proto.get("descriptionModule", {}).get("briefSummary", ""),
            "eligibility": proto.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
        })

    return trials


def format_trial_text(trial: dict) -> str:
    phases = ", ".join(trial.get("phases", [])) or "Not specified"
    conditions = ", ".join(trial.get("conditions", [])) or "Not specified"
    interventions = ", ".join(trial.get("interventions", [])) or "Not specified"
    eligibility = (trial.get("eligibility") or "")[:500]

    return (
        f"Trial ID: {trial['nct_id']}\n"
        f"Title: {trial['title']}\n"
        f"Status: {trial['status']}\n"
        f"Phase: {phases}\n"
        f"Conditions: {conditions}\n"
        f"Interventions: {interventions}\n"
        f"Summary: {trial['summary']}\n"
        f"Eligibility: {eligibility or 'Not specified'}"
    )
