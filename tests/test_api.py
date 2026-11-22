import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.pipeline import ClinicalPipeline


@pytest.fixture
def client(make_deps):
    deps = make_deps()
    pipeline = ClinicalPipeline(config=deps.config, deps=deps)
    with TestClient(create_app(pipeline)) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_config_endpoint_has_roles_and_no_secrets(client):
    body = client.get("/v1/config").json()
    assert body["planner"]["name"] == "nemo" and "key" not in str(body).lower().replace("keys", "")


def test_ask(client):
    response = client.post("/v1/ask", json={"question": "type 2 diabetes phase 3 trials"})
    body = response.json()
    assert response.status_code == 200
    assert body["verified_citations"] == ["NCT00000001"] and body["timings"] and body["models_used"]


@pytest.mark.parametrize("payload", [{}, {"question": "ab"}, {"question": "x" * 3000}, {"q": "no"}])
def test_ask_validates_input(client, payload):
    assert client.post("/v1/ask", json=payload).status_code == 422


def test_plan_endpoint(client):
    body = client.post("/v1/plan", json={"question": "Summarize NCT00000001"}).json()
    assert body["tool_calls_llm"][0]["name"] == "get_trial"


def test_tool_endpoint_runs_valid_call(client):
    response = client.post("/v1/tools/get_trial", json={"arguments": {"nct_id": "NCT00000001"}})
    assert response.status_code == 200 and response.json()["n_trials"] == 1


def test_tool_endpoint_rejects_bad_arguments_and_unknown_tools(client):
    assert client.post("/v1/tools/get_trial", json={"arguments": {"nct_id": "nope"}}).status_code == 422
    assert client.post("/v1/tools/compare_trials", json={"arguments": {"nct_ids": ["NCT00000001"]}}).status_code == 422
    assert client.post("/v1/tools/drop_tables", json={"arguments": {}}).status_code == 404


def test_retrieve_endpoint(client):
    body = client.post("/v1/retrieve", json={"question": "metformin diabetes", "condition": "diabetes"}).json()
    assert body["n_fetched"] == 2 and body["results"][0]["nct_id"].startswith("NCT")


def test_retrieve_endpoint_validates_phase(client):
    response = client.post("/v1/retrieve", json={"question": "diabetes trials", "phase": "PHASE9"})
    assert response.status_code == 422


def test_retrieve_falls_back_to_question_as_search_term(client):
    assert client.post("/v1/retrieve", json={"question": "melanoma"}).json()["n_fetched"] == 1


def test_judge_endpoint(client):
    body = client.post("/v1/evaluate/judge", json={
        "question": "q about trials", "answer": "a [NCT00000001]", "contexts": ["ctx"]}).json()
    assert body["faithfulness"] == 0.9 and body["error"] is None
    assert client.post("/v1/evaluate/judge", json={"question": "q about", "answer": "a", "contexts": []}).status_code == 422


def test_openapi_schema_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/v1/ask", "/v1/plan", "/v1/retrieve", "/v1/tools/{tool_name}", "/v1/evaluate/judge", "/healthz"):
        assert path in paths
