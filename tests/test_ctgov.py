from unittest.mock import MagicMock, patch

from src.retrieval.ctgov_client import build_query_params, fetch_trials, format_trial_text

SAMPLE_TRIAL = {
    "nct_id": "NCT12345678",
    "title": "Metformin for Type 2 Diabetes",
    "status": "RECRUITING",
    "phases": ["PHASE3"],
    "conditions": ["Type 2 Diabetes"],
    "interventions": ["Metformin"],
    "summary": "A study of metformin efficacy.",
    "eligibility": "Adults aged 18-65 with T2DM.",
}


def test_build_query_params_minimal():
    params = build_query_params(condition="diabetes")
    assert params["query.cond"] == "diabetes"
    assert params["format"] == "json"
    assert params["pageSize"] == 15
    assert "filter.phase" not in params


def test_build_query_params_full():
    params = build_query_params(
        condition="diabetes", intervention="metformin",
        phase="PHASE3", status="RECRUITING",
    )
    assert params["query.cond"] == "diabetes"
    assert params["query.intr"] == "metformin"
    assert params["filter.phase"] == "PHASE3"
    assert params["filter.overallStatus"] == "RECRUITING"


def test_format_trial_text_contains_key_fields():
    text = format_trial_text(SAMPLE_TRIAL)
    assert "NCT12345678" in text
    assert "PHASE3" in text
    assert "Metformin" in text
    assert "RECRUITING" in text


def test_format_trial_text_truncates_eligibility():
    long_trial = {**SAMPLE_TRIAL, "eligibility": "x" * 1000}
    text = format_trial_text(long_trial)
    assert len(text) < 2000


@patch("src.retrieval.ctgov_client.httpx.Client")
def test_fetch_trials_parses_response(mock_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "studies": [{
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Test"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "descriptionModule": {"briefSummary": "Test summary"},
                "conditionsModule": {"conditions": ["Diabetes"]},
                "designModule": {"phases": ["PHASE3"]},
                "eligibilityModule": {"eligibilityCriteria": "Adults"},
                "armsInterventionsModule": {"interventions": [{"name": "Metformin"}]},
            }
        }]
    }
    mock_client.return_value.__enter__.return_value.get.return_value = mock_response

    trials = fetch_trials({"query.cond": "diabetes"})
    assert len(trials) == 1
    assert trials[0]["nct_id"] == "NCT12345678"
    assert trials[0]["interventions"] == ["Metformin"]


@patch("src.retrieval.ctgov_client.httpx.Client")
def test_fetch_trials_skips_missing_nct_id(mock_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "studies": [{"protocolSection": {"identificationModule": {}}}]
    }
    mock_client.return_value.__enter__.return_value.get.return_value = mock_response

    trials = fetch_trials({"query.cond": "diabetes"})
    assert trials == []
