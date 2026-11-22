"""FastAPI service exposing retrieval, tools, planning, answering, and judging.

Run:  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
Docs: /docs (Swagger UI), /openapi.json
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request

from src.agent.tools import TOOLS, validate_call
from src.api.schemas import (
    AskRequest, AskResponse, JudgeRequest, JudgeResponse, PlanResponse, RetrieveRequest,
    RetrieveResponse, ToolRequest, ToolResponse,
)


def create_app(pipeline=None) -> FastAPI:
    """Build the app. Pass a pipeline for tests; otherwise one is created at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            from src.pipeline import ClinicalPipeline
            app.state.pipeline = ClinicalPipeline()
        yield

    app = FastAPI(title="ClinicalGov API", version="1.0.0", lifespan=lifespan)
    app.state.pipeline = pipeline

    def get_pipeline(request: Request):
        return request.app.state.pipeline

    # Endpoints are sync on purpose: FastAPI runs them in a thread pool, so slow
    # LLM and network calls do not block the event loop.

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/v1/config")
    def config(request: Request):
        return get_pipeline(request).describe()

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        return get_pipeline(request).ask(body.question)

    @app.post("/v1/plan", response_model=PlanResponse)
    def plan(body: AskRequest, request: Request):
        return get_pipeline(request).plan(body.question)

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    def retrieve(body: RetrieveRequest, request: Request):
        args = body.model_dump(exclude={"question"}, exclude_none=True)
        if not any(k in args for k in ("condition", "intervention", "term")):
            args["term"] = body.question
        try:
            return get_pipeline(request).search(args, question=body.question)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/v1/tools/{tool_name}", response_model=ToolResponse)
    def run_tool(tool_name: str, body: ToolRequest, request: Request):
        if tool_name not in TOOLS:
            raise HTTPException(status_code=404, detail=f"Unknown tool. Available: {sorted(TOOLS)}")
        if validate_call(tool_name, body.arguments) is None:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid arguments for {tool_name}. Schema: {TOOLS[tool_name][1].model_json_schema()}",
            )
        try:
            trials = get_pipeline(request).call_tool(tool_name, body.arguments)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return {"tool": tool_name, "n_trials": len(trials), "trials": trials}

    @app.post("/v1/evaluate/judge", response_model=JudgeResponse)
    def judge(body: JudgeRequest, request: Request):
        return get_pipeline(request).judge(body.question, body.answer, body.contexts)

    return app


app = create_app()
