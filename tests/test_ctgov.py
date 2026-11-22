import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval import ctgov_client
from src.retrieval.ctgov_client import (
    build_query_params, fetch_trials, format_trial_text, get_trial, trial_id_from_text,
)

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
    assert "aggFilters" not in params


def test_build_query_params_full():
    params = build_query_params(
        condition="diabetes", intervention="metformin",
        phase="PHASE3", status="RECRUITING",
    )
    assert params["query.cond"] == "diabetes"
    assert params["query.intr"] == "metformin"
    assert "aggFilters" in params
    assert "phase:3" in params["aggFilters"]
    assert "status:rec" in params["aggFilters"]


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


@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_fetch_trials_parses_response(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
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
    }).encode()
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    trials = fetch_trials({"query.cond": "diabetes"})
    assert len(trials) == 1
    assert trials[0]["nct_id"] == "NCT12345678"
    assert trials[0]["interventions"] == ["Metformin"]


@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_fetch_trials_skips_missing_nct_id(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "studies": [{"protocolSection": {"identificationModule": {}}}]
    }).encode()
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    trials = fetch_trials({"query.cond": "diabetes"})
    assert trials == []


def _study(nct):
    return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": nct}}}


def _resp(payload):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    return ctx


@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_fetch_trials_follows_pagination_until_max_results(mock_urlopen):
    mock_urlopen.side_effect = [
        _resp({"studies": [_study("NCT00000001"), _study("NCT00000002")], "nextPageToken": "p2"}),
        _resp({"studies": [_study("NCT00000003"), _study("NCT00000004")], "nextPageToken": "p3"}),
    ]
    trials = fetch_trials({"query.cond": "x", "pageSize": 2}, max_results=3)
    assert [t["nct_id"] for t in trials] == ["NCT00000001", "NCT00000002", "NCT00000003"]
    assert "pageToken=p2" in mock_urlopen.call_args_list[1][0][0].full_url


@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_fetch_trials_stops_when_no_next_page(mock_urlopen):
    mock_urlopen.side_effect = [_resp({"studies": [_study("NCT00000001")]})]
    assert len(fetch_trials({"query.cond": "x"}, max_results=50)) == 1
    assert mock_urlopen.call_count == 1


@patch("src.retrieval.ctgov_client._sleep")
@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_retries_on_429_then_succeeds(mock_urlopen, mock_sleep):
    err = urllib.error.HTTPError("u", 429, "Too Many", {"Retry-After": "7"}, None)
    mock_urlopen.side_effect = [err, _resp({"studies": [_study("NCT00000001")]})]
    assert len(fetch_trials({"query.cond": "x"})) == 1
    mock_sleep.assert_called_once_with(7.0)


@patch("src.retrieval.ctgov_client._sleep")
@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_404_is_not_retried(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    with pytest.raises(RuntimeError, match="404"):
        fetch_trials({"query.cond": "x"})
    assert mock_urlopen.call_count == 1 and not mock_sleep.called


@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_get_trial_parses_single_study(mock_urlopen):
    mock_urlopen.side_effect = [_resp(_study("NCT00000009"))]
    assert get_trial("nct00000009")["nct_id"] == "NCT00000009"
    assert "/NCT00000009?" in mock_urlopen.call_args[0][0].full_url


@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_get_trial_missing_returns_none(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    assert get_trial("NCT00000009") is None


def test_get_trial_rejects_bad_id():
    with pytest.raises(ValueError):
        get_trial("not-an-id")


@patch("src.retrieval.ctgov_client.urllib.request.urlopen")
def test_snapshot_cache_readwrite_then_readonly(mock_urlopen, tmp_path, monkeypatch):
    monkeypatch.setenv("CTGOV_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CTGOV_CACHE_MODE", "readwrite")
    mock_urlopen.side_effect = [_resp({"studies": [_study("NCT00000001")]})]
    assert len(fetch_trials({"query.cond": "x"})) == 1
    assert len(fetch_trials({"query.cond": "x"})) == 1      # served from disk
    assert mock_urlopen.call_count == 1

    monkeypatch.setenv("CTGOV_CACHE_MODE", "readonly")
    with pytest.raises(ctgov_client.SnapshotMiss):
        fetch_trials({"query.cond": "never-seen"})


def test_query_params_term_and_sponsor():
    params = build_query_params(term="aspirin", sponsor="NIH")
    assert params["query.term"] == "aspirin" and params["query.spons"] == "NIH"


def test_trial_id_from_text_roundtrip():
    trial = {"nct_id": "NCT00000001", "title": "t", "status": "s"}
    assert trial_id_from_text(format_trial_text(trial)) == "NCT00000001"
    assert trial_id_from_text("nothing") is None


def test_eligibility_length_is_configurable():
    trial = {"nct_id": "NCT00000001", "title": "t", "status": "s", "eligibility": "y" * 800}
    assert "y" * 800 in format_trial_text(trial, eligibility_chars=1000)
    assert "y" * 501 not in format_trial_text(trial)
