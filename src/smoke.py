"""Smoke test: `python main.py --smoke` (add --quick to skip LLM and full-pipeline checks).

Each check prints PASS/WARN/FAIL with a hint, so a first run on a new machine tells you
exactly what to fix. Needs no pytest and no mocks: it exercises the real services.
"""
import json
import time
import urllib.error
import urllib.request

from src.config import ConfigError, load_config

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _http_json(url: str, payload: dict = None, timeout: int = 15) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _report(results: list, status: str, name: str, detail: str = "", hint: str = "") -> None:
    results.append(status)
    line = f"[{status}] {name}" + (f": {detail}" if detail else "")
    print(line)
    if hint and status != PASS:
        print(f"       fix: {hint}")


def _check_ollama(config, results) -> None:
    endpoints = {}
    for spec in config.models.values():
        if spec.provider == "ollama":
            endpoints.setdefault(spec.base_url, []).append(spec)

    for base_url, specs in endpoints.items():
        try:
            tags = _http_json(f"{base_url}/api/tags")
        except Exception as exc:
            _report(results, FAIL, f"Ollama reachable at {base_url}", str(exc)[:120],
                    "start Ollama or set OLLAMA_BASE_URL")
            continue
        installed = {m["name"] for m in tags.get("models", [])}
        installed |= {n.removesuffix(":latest") for n in installed}
        _report(results, PASS, f"Ollama reachable at {base_url}", f"{len(installed)} model tags")

        for spec in specs:
            if spec.model not in installed:
                _report(results, FAIL, f"model '{spec.model}' ({spec.name}) installed", "not found",
                        f"run `ollama pull {spec.model}` or edit the tag in the config")
                continue
            _report(results, PASS, f"model '{spec.model}' ({spec.name}) installed")
            if spec.tool_calling == "native":
                try:
                    caps = _http_json(f"{base_url}/api/show", {"model": spec.model}).get("capabilities", [])
                except Exception:
                    caps = []
                if "tools" in caps:
                    _report(results, PASS, f"'{spec.name}' supports native tool calling")
                else:
                    _report(results, WARN, f"'{spec.name}' native tool calling", "capability not reported",
                            "set tool_calling: json for this model")


def _check_ctgov(results) -> None:
    from src.retrieval import ctgov_client

    try:
        trials = ctgov_client.fetch_trials(ctgov_client.build_query_params(condition="diabetes", page_size=2))
        _report(results, PASS, "ClinicalTrials.gov search", f"{len(trials)} trials")
        if trials:
            trial = ctgov_client.get_trial(trials[0]["nct_id"])
            _report(results, PASS if trial else FAIL, "ClinicalTrials.gov lookup by NCT ID",
                    trials[0]["nct_id"])
    except Exception as exc:
        _report(results, FAIL, "ClinicalTrials.gov", str(exc)[:120], "check network/proxy access to clinicaltrials.gov")


def _check_embeddings_and_store(config, results):
    from src.retrieval.vector_store import get_embeddings, get_vector_store, retrieve_documents, store_trials

    try:
        embeddings = get_embeddings(config.embedding.model, config.embedding.device)
        dim = len(embeddings.embed_documents(["smoke test"])[0])
        _report(results, PASS, "embedding model loads", f"{config.embedding.model} ({dim} dims)")
    except Exception as exc:
        hint = ("pip install -r requirements.txt" if isinstance(exc, ImportError)
                else "check Hugging Face access or the model name")
        _report(results, FAIL, "embedding model", str(exc)[:120], hint)
        return None

    try:
        import tempfile
        store = get_vector_store(embeddings, persist_directory=tempfile.mkdtemp(prefix="clinicalgov_smoke_"))
        trial = {"nct_id": "NCT00000000", "title": "Smoke", "status": "COMPLETED", "phases": ["PHASE1"],
                 "conditions": ["smoke"], "interventions": ["none"], "summary": "smoke test", "eligibility": ""}
        store_trials([trial], store)
        docs = retrieve_documents("smoke test", store, k=1, restrict_ids=["NCT00000000"])
        _report(results, PASS if docs else FAIL, "ChromaDB store and retrieve (temp dir)")
    except Exception as exc:
        _report(results, FAIL, "ChromaDB", str(exc)[:120], "check chromadb install")
    return embeddings


def run_smoke(quick: bool = False) -> bool:
    results: list[str] = []
    print("ClinicalGov smoke test\n")

    try:
        config = load_config()
    except ConfigError as exc:
        _report(results, FAIL, "config loads", str(exc)[:300], "fix the YAML or set MODELS_CONFIG")
        return False
    _report(results, PASS, "config loads")
    for role, info in config.describe().items():
        if role in {"fallback", "pipeline"}:
            continue
        infos = info if isinstance(info, list) else [info]
        print(f"       {role}: " + ", ".join(f"{i['name']}={i['provider']}:{i['model']}" for i in infos))
    print()

    _check_ollama(config, results)
    _check_ctgov(results)
    _check_embeddings_and_store(config, results)

    if not quick and FAIL not in results:
        from src.pipeline import ClinicalPipeline

        pipeline = ClinicalPipeline(config)
        for name in config.models:
            try:
                start = time.perf_counter()
                reply = pipeline.deps.registry.call(name, [("human", "Reply with the single word OK.")], chain=[name])
                _report(results, PASS, f"model '{name}' answers", f"{time.perf_counter() - start:.1f}s: {reply.text[:40]!r}")
            except Exception as exc:
                _report(results, FAIL, f"model '{name}' answers", str(exc)[:150], "check the model tag and endpoint")

        try:
            plan = pipeline.plan("What phase 3 trials are recruiting for type 2 diabetes?")
            ok = plan["plan_json_ok"] and bool(plan["tool_calls_llm"])
            _report(results, PASS if ok else WARN, "planner produces valid tool calls",
                    json.dumps(plan["tool_calls_llm"])[:120],
                    "the planner model may be too small to follow the JSON format; try another or tool_calling: native")
        except Exception as exc:
            _report(results, FAIL, "planner", str(exc)[:150])

        try:
            result = pipeline.ask("What phase 3 trials are recruiting for type 2 diabetes?")
            status = PASS if result["answer"] and not result["errors"] else WARN
            _report(results, status, "full pipeline query", f"{result['total_latency_s']}s, "
                    f"{len(result['verified_citations'])} verified / {len(result['unverified_citations'])} unverified citations",
                    "see `errors` and `warnings` below")
            for key in ("errors", "warnings"):
                for item in result[key]:
                    print(f"       {key[:-1]}: {item[:200]}")
            print("       timings: " + json.dumps(result["timings"]))
        except Exception as exc:
            _report(results, FAIL, "full pipeline query", str(exc)[:200])

    failed = results.count(FAIL)
    print(f"\n{results.count(PASS)} passed, {results.count(WARN)} warnings, {failed} failed")
    return failed == 0
