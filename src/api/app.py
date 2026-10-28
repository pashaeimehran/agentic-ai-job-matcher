"""FastAPI application: typed HTTP routes over the shared tool layer and the LangGraph agent.

Routes translate HTTP <-> application calls only. Search, matching and saving
run in src.tools via ToolRegistry.execute(); /chat runs LangGraphAgent.
"""
from __future__ import annotations

import re
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import APIRouter, Depends, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import (ApiServices, build_default_services, get_agent, get_catalog,
                                  get_request_id, get_services)
from src.api.errors import ApiError, ErrorResponse, from_agent_status, log_event
from src.api.schemas import (AnalyzeJobRequest, AnalyzeJobResponse, ChatRequest, ChatResponse, CVSearchRequest,
                             CVSearchResponse, HealthResponse, JobDetailsResponse, JobListItem, JobListResponse,
                             JobSearchResponse, SearchRequest, SearchResponse, ToolCallSummary)
from src.graph import LangGraphAgent
from src.tools.jobs import JobCatalog
from src.tools.schemas import JobDetailsResult

API_VERSION = "0.1.0"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ERROR_RESPONSES = {code: {"model": ErrorResponse} for code in (404, 409, 422, 500, 502, 503)}

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(services: ApiServices = Depends(get_services)) -> HealthResponse:
    return HealthResponse(version=API_VERSION, chat_available=services.agent is not None,
                          tools=services.registry.names(), catalog_jobs=len(services.context.catalog.all()))


@router.get("/jobs", response_model=JobListResponse, responses=ERROR_RESPONSES)
def list_jobs(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
              catalog: JobCatalog = Depends(get_catalog),
              request_id: str = Depends(get_request_id)) -> JobListResponse:
    """Browse the job catalog in stable catalog order. Use POST /search for ranked search."""
    jobs = catalog.all()
    page = [JobListItem(job_id=j.job_id, role=j.posting.role, location=j.posting.location,
                        seniority=j.posting.seniority, skills=j.posting.skills, source=j.source)
            for j in jobs[offset:offset + limit]]
    return JobListResponse(request_id=request_id, total=len(jobs), offset=offset, limit=limit, jobs=page)


@router.get("/jobs/{job_id}", response_model=JobDetailsResponse, responses=ERROR_RESPONSES)
def job_details(job_id: str = Path(min_length=1, max_length=120), services: ApiServices = Depends(get_services),
                request_id: str = Depends(get_request_id)) -> JobDetailsResponse:
    _, data = services.call_tool("get_job_details", {"job_id": job_id}, request_id)
    return JobDetailsResponse(request_id=request_id, job=JobDetailsResult.model_validate(data))


@router.post("/search", response_model=SearchResponse, responses=ERROR_RESPONSES)
def search(body: SearchRequest, services: ApiServices = Depends(get_services),
           request_id: str = Depends(get_request_id)) -> JobSearchResponse | CVSearchResponse:
    """Keyword job search (target=jobs) or CV evidence retrieval (target=cv). Empty is not an error."""
    tool = "search_cv" if isinstance(body, CVSearchRequest) else "search_jobs"
    status, data = services.call_tool(tool, body.model_dump(exclude={"target"}, exclude_none=True), request_id)
    if tool == "search_cv":
        return CVSearchResponse(request_id=request_id, status=status, **data)
    return JobSearchResponse(request_id=request_id, status=status, **data)


@router.post("/analyze-job", response_model=AnalyzeJobResponse, responses=ERROR_RESPONSES)
def analyze_job(body: AnalyzeJobRequest, services: ApiServices = Depends(get_services),
                request_id: str = Depends(get_request_id)) -> AnalyzeJobResponse:
    """Deterministic CV compatibility for one job via calculate_match (no LLM involved)."""
    _, data = services.call_tool("calculate_match", {"job_ids": [body.job_id]}, request_id)
    return AnalyzeJobResponse(request_id=request_id, match=data["matches"][0], scoring_note=data["scoring_note"])


@router.post("/chat", response_model=ChatResponse, responses=ERROR_RESPONSES)
def chat(body: ChatRequest, agent: LangGraphAgent = Depends(get_agent),
         request_id: str = Depends(get_request_id)) -> ChatResponse:
    """Natural-language request through the LangGraph agent; thread_id continues a conversation."""
    # A new conversation always gets a fresh thread; client-chosen request IDs never select one.
    result = agent.run(body.message, thread_id=body.thread_id or uuid.uuid4().hex)
    log_event("chat_run", request_id=request_id, run_id=result.run_id, thread_id=result.thread_id,
              status=result.status, tool_calls=len(result.tool_calls))
    if result.status != "completed":
        raise from_agent_status(result.status, result.error, result.run_id)
    return ChatResponse(
        request_id=request_id, run_id=result.run_id, thread_id=result.thread_id, answer=result.final_answer or "",
        model_turns=result.model_turns, ungrounded_scores=result.ungrounded_scores,
        tool_calls=[ToolCallSummary(step=r.step, tool=r.tool_name, status=r.status, error_code=r.error_code,
                                    attempts=r.attempts) for r in result.tool_calls])


# --- Error handling ----------------------------------------------------------------

def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]


def _error_response(request: Request, error: ApiError) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(error.body(request_id).model_dump(), status_code=error.status_code,
                        headers={"X-Request-ID": request_id})


async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
    log_event("api_error", request_id=_request_id(request), path=request.url.path, code=exc.code,
              status=exc.status_code)
    return _error_response(request, exc)


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Input values are omitted: request bodies may contain personal data.
    details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return _error_response(request, ApiError(422, "validation_error", "Request does not match the API schema.",
                                             details=details))


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code, message = {404: ("not_found", "Resource not found."),
                     405: ("method_not_allowed", "Method not allowed.")}.get(
        exc.status_code, ("internal_error", "Request failed."))
    return _error_response(request, ApiError(exc.status_code, code, message))


async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    log_event("api_unhandled_exception", request_id=_request_id(request), path=request.url.path,
              exception=type(exc).__name__)
    return _error_response(request, ApiError(500, "internal_error", "Internal server error."))


def create_app(services: ApiServices | None = None,
               services_factory: Callable[[], ApiServices] = build_default_services) -> FastAPI:
    """Build the app. Pass services in tests; otherwise they are created at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "services", None) is None:
            app.state.services = services_factory()
        yield

    app = FastAPI(title="Job Match Agent API", version=API_VERSION, lifespan=lifespan,
                  description="Typed REST API over the shared job-matching tools and the LangGraph agent.")
    app.state.services = services

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled_error)
    app.include_router(router)
    return app
