import json

import pytest

from src.agent.tools import (
    calls_from_json, calls_from_native, execute_tool, find_nct_ids, nct_fix_call,
    parse_json_loose, plan_covers_ids, tool_schemas, validate_call,
)
from tests.conftest import FakeCtgov


def test_parse_json_loose_handles_fences_think_and_prose():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('<think>hmm {"x":2}</think>{"a": 1}') == {"a": 1}
    assert parse_json_loose('Sure! Here you go: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert parse_json_loose("no json here") is None
    assert parse_json_loose("[1,2]") is None


def test_validate_call_accepts_and_normalizes():
    call = validate_call("search_trials", {"condition": "diabetes", "phase": "PHASE3"})
    assert call == {"name": "search_trials",
                    "args": {"condition": "diabetes", "phase": "PHASE3", "max_results": 25}}


@pytest.mark.parametrize("name,args", [
    ("search_trials", {"phase": "PHASE9"}),
    ("get_trial", {"nct_id": "123"}),
    ("compare_trials", {"nct_ids": ["NCT00000001"]}),
    ("delete_everything", {}),
    ("get_trial", "not a dict"),
])
def test_validate_call_rejects_bad_input(name, args):
    assert validate_call(name, args) is None


def test_calls_from_json_keeps_only_valid_calls():
    raw = json.dumps({"tool_calls": [
        {"name": "get_trial", "args": {"nct_id": "NCT00000001"}},
        {"name": "get_trial", "args": {"nct_id": "bad"}},
        {"name": "nope", "args": {}},
    ]})
    calls, ok = calls_from_json(raw)
    assert ok and [c["name"] for c in calls] == ["get_trial"]


def test_calls_from_json_flags_unparseable_output():
    assert calls_from_json("I think you should search") == ([], False)


def test_calls_from_json_caps_number_of_calls():
    raw = json.dumps({"tool_calls": [{"name": "search_trials", "args": {"term": str(i)}} for i in range(6)]})
    assert len(calls_from_json(raw)[0]) == 3


def test_calls_from_json_accepts_single_call_object():
    calls, _ = calls_from_json('{"name": "search_trials", "args": {"term": "x"}}')
    assert calls[0]["name"] == "search_trials"


def test_calls_from_native():
    calls = calls_from_native([{"name": "get_trial", "args": {"nct_id": "NCT00000001"}, "id": "1"},
                               {"name": "bogus", "args": {}}])
    assert calls == [{"name": "get_trial", "args": {"nct_id": "NCT00000001"}}]


def test_find_nct_ids_dedupes_and_uppercases():
    assert find_nct_ids("see nct00000001, NCT00000002 and NCT00000001") == ["NCT00000001", "NCT00000002"]


def test_plan_covers_ids():
    plan = [{"name": "compare_trials", "args": {"nct_ids": ["NCT00000001", "NCT00000002"]}}]
    assert plan_covers_ids(plan, ["NCT00000001", "NCT00000002"])
    assert not plan_covers_ids(plan, ["NCT00000003"])
    assert not plan_covers_ids([], ["NCT00000001"])


def test_nct_fix_call_picks_tool_from_question():
    assert nct_fix_call("Tell me about NCT00000001", ["NCT00000001"])["name"] == "get_trial"
    assert nct_fix_call("What are the eligibility criteria of NCT00000001?", ["NCT00000001"])["name"] == "get_eligibility"
    assert nct_fix_call("Compare", ["NCT00000001", "NCT00000002"])["name"] == "compare_trials"


def test_execute_tools_against_fake_api():
    api = FakeCtgov()
    assert [t["nct_id"] for t in execute_tool("get_trial", {"nct_id": "NCT00000001"}, api)] == ["NCT00000001"]
    assert execute_tool("get_trial", {"nct_id": "NCT00000099"}, api) == []
    both = execute_tool("compare_trials", {"nct_ids": ["NCT00000001", "NCT00000003"]}, api)
    assert len(both) == 2
    found = execute_tool("search_trials", {"condition": "melanoma", "max_results": 10}, api)
    assert [t["nct_id"] for t in found] == ["NCT00000003"]
    params, max_results = api.search_calls[-1]
    assert params["query.cond"] == "melanoma" and max_results == 10


def test_schemas_cover_every_tool():
    names = {s["function"]["name"] for s in tool_schemas()}
    assert names == {"search_trials", "get_trial", "compare_trials", "get_eligibility"}
