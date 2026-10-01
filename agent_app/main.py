"""
LakeRCM Agent App — Standalone agent exposing /responses as SSE stream.

Architecture:
  POST /responses -> LangGraph ReAct Agent -> sync tools -> SSE stream
  (point lookups on Lakebase; analytics on the Reyden warehouse via system.ai.dbsql MCP)
"""

import json
import logging
import os
import sys
import time
from uuid import uuid4

import psycopg
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s %(levelname)s [%(name)s] "
        "[req=%(req_id)s thread=%(thread_id)s] %(message)s"
    ),
    stream=sys.stdout,
    force=True,
)


class _DefaultExtrasFilter(logging.Filter):
    """Fill `req_id`/`thread_id` defaults on records that don't pass `extra=`.

    Implemented as a Filter (not a LogRecordFactory) so it runs *after*
    `Logger.makeRecord` applies the caller-provided `extra=` dict — a factory
    would pre-populate those attrs and trigger
    "Attempt to overwrite 'req_id' in LogRecord" KeyErrors.
    """

    def filter(self, record):  # noqa: D401
        if not hasattr(record, "req_id"):
            record.req_id = "-"
        if not hasattr(record, "thread_id"):
            record.thread_id = "-"
        return True


for _h in logging.getLogger().handlers:
    _h.addFilter(_DefaultExtrasFilter())

logging.getLogger("uvicorn.access").setLevel(
    os.getenv("UVICORN_ACCESS_LOG_LEVEL", "WARNING")
)
logging.getLogger("psycopg.pool").setLevel("WARNING")
logging.getLogger("mlflow").setLevel("WARNING")
logging.getLogger("httpx").setLevel("WARNING")
logging.getLogger("databricks.sdk").setLevel("WARNING")

logger = logging.getLogger("lakercm.agent")

from agent.graph import create_agent, get_agui_graph  # noqa: E402
from agent.llm import set_llm_session_id  # noqa: E402
from agent.tools import (  # noqa: E402
    kg_status,
    set_active_document_id,
    set_authorized_user_email,
)
from services.checkpointer import (
    delete_thread as delete_checkpointer_thread,
)  # noqa: E402
from services.lakehouse_db import get_db  # noqa: E402
from services.user_mapping import get_user_for_email  # noqa: E402
from services.observability import (  # noqa: E402
    init_tracing,
    set_prompt_context,
    set_session_context,
)

init_tracing()

app = FastAPI(title="LakeRCM Agent", version="1.0.0")
# PHI-safe FastAPI OTel instrumentation. OPT-IN and OFF by default (the agent
# app ships uninstrumented — plain uvicorn, MLflow dual-export is its telemetry
# path); gated behind LAKERCM_AGENT_OTEL_FASTAPI. See services/otel_fastapi.py.
from services.otel_fastapi import instrument_fastapi  # noqa: E402

instrument_fastapi(app)

_cors_origins = [
    o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _bootstrap_async_checkpointer():
    """Open the AsyncConnectionPool and run AsyncPostgresSaver.setup() once,
    inside the live FastAPI event loop. The AG-UI graph was constructed at
    module-import with the saver reference already attached; this step just
    finishes its initialization (opens the pool, creates/verifies the
    `public.checkpoint_*` tables under the group-backed role).

    Tracing is disabled inside this block because `mlflow.langchain.autolog()`
    is already active and would promote the setup DDL into orphan root
    traces in the experiment. The setup runs once per pod boot — observed
    via logs, not MLflow.
    """
    from services.checkpointer import init_async_checkpointer
    from services.tracing import tracing_disabled

    with tracing_disabled():
        await init_async_checkpointer()


@app.on_event("shutdown")
async def _flush_telemetry_on_shutdown():
    """Flush buffered spans/traces so the FINAL batch isn't dropped on pod stop.

    Covers both export paths, best-effort and fully guarded (a missing API or
    disabled export must never turn shutdown into an error):
      1. MLflow async trace-logging queue (MLFLOW_ENABLE_ASYNC_TRACE_LOGGING).
      2. MLflow span processors — native MlflowV3SpanProcessor and, when
         MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT is on, the additive
         OtelSpanProcessor (its BatchSpanProcessor batches OTLP exports).
      3. Any global OTel SDK TracerProvider (no-op ProxyTracerProvider when
         telemetry is off — guarded).
    """
    try:
        import mlflow

        flush = getattr(mlflow, "flush_trace_async_logging", None)
        if callable(flush):
            flush(terminate=True)
            logger.info("MLflow async trace logging flushed on shutdown")
    except Exception as e:
        logger.warning("MLflow async trace flush failed on shutdown: %s", e)

    try:
        from mlflow.tracing.provider import _get_span_processors

        for proc in _get_span_processors() or []:
            force_flush = getattr(proc, "force_flush", None)
            if callable(force_flush):
                force_flush()
        logger.info("MLflow span processors force_flush complete on shutdown")
    except Exception as e:
        logger.warning("MLflow span-processor flush failed on shutdown: %s", e)

    try:
        from opentelemetry import trace as _otel_trace

        provider = _otel_trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as e:
        logger.warning("OTel provider flush/shutdown failed on shutdown: %s", e)


# Boot outcome of complexity-tiered routing, surfaced on /health. Populated by the
# tier-graph block far below; read at request time, so definition order is fine.
# Default is the "never even attempted" state — distinguishable from an explicit
# off and from a failure, because those want different fixes.
_ROUTING_BOOT: dict = {"enabled": False, "reason": "not initialized"}

# Per-turn routing outcomes. Deliberately a plain module dict: it only ever
# accumulates small counters, it is read by /health, and it must never be able to
# raise inside the request path it measures.
_ROUTING_TURNS: dict = {}


@app.get("/health")
async def health():
    """Readiness-aware health with reviewer-style degraded semantics.

    The agent still serves stateless turns when the checkpointer is down, so
    degraded != dead. Read-only — never opens pools or triggers setup()
    (health probes must not mutate state). Pattern absorbed from the
    agent-openai-advanced template review.
    """
    from config import settings
    from services.checkpointer import checkpointer_status

    cp = checkpointer_status()
    return {
        "status": "healthy" if cp["async_setup_done"] else "degraded",
        "service": "lakercm-agent",
        "version": "1.0.0",
        "checkpointer": cp,
        "experiment_configured": bool(
            os.getenv("MLFLOW_EXPERIMENT_ID") or settings.mlflow_experiment_name
        ),
        "lakebase_host": settings.pg_host or None,
        # Whether the tier selector in the UI actually does anything. Without
        # this, "routing is on" could only be inferred from a routing_marker
        # event appearing in an SSE stream.
        "routing": dict(_ROUTING_BOOT),
        # Per-turn outcomes. `total > 0` with `resolved == 0` localises the break
        # to the request path rather than boot; `skipped_no_tier_agents` vs
        # `failed` says which half, and `last_error` carries the exception that a
        # rotated log window would otherwise have eaten.
        "routing_turns": dict(_ROUTING_TURNS),
        # Whether the knowledge-graph traversal tool is actually registered in
        # this pod. #88 shipped the tool but its two env vars were missing, so it
        # was dark with no way to tell from outside; `reason` also catches the
        # half-state where the flag is on but the schema is empty, which
        # registers the tool and then refuses every call.
        "kg": kg_status(),
    }


# --- AG-UI endpoint ---------------------------------------------------------
#
# Mounts a bare AG-UI-speaking endpoint at /copilotkit so the reviewer app's
# React frontend can drive the LangGraph agent. We deliberately do NOT use
# the `copilotkit` Python runtime adapter: with `copilotkit==0.1.90` and
# `ag-ui-langgraph==0.0.35` it ships a broken pair (the SDK calls
# `dict_repr()` and `execute()` on its `LangGraphAGUIAgent` but neither
# method exists — GH issues #2891, #2997, #4995). CopilotKit React v2
# ships `@ag-ui/client.HttpAgent`, which speaks raw AG-UI directly, so we
# point that at this bare endpoint via `selfManagedAgents` on the React
# provider and skip the broken Python layer entirely.
#
# Runs alongside the legacy /responses SSE route. Auth flows through
# Databricks App OBO headers exactly the same way — AG-UI does not
# impose its own auth scheme.
#
# Subclass `LangGraphAgent` to open an explicit MLflow span around each
# run and populate the trace metadata Databricks Scheduled Scorers expect.
# Three concerns rolled together:
#
#   1. `mlflow.langchain.autolog()` does create per-call spans, but they
#      don't propagate as "current active span" reliably inside the
#      LangGraph hooks we use (post_model_hook). The legacy /responses
#      path solves this by opening `mlflow.start_span("agent_turn")`
#      explicitly; the AG-UI path needs the same so post_model_hook can
#      dispatch its trace_marker custom event.
#   2. Inputs/outputs must be set on the root span — Safety, Correctness,
#      RelevanceToQuery, Guidelines scorers all read from
#      trace.info.{request_preview, response_preview} and the span's
#      `messages` payload. Without this, scorers no-op or score the
#      empty string.
#   3. session.thread_id + session.user_email tags make traces
#      filterable in the MLflow UI and groupable per reviewer.
#
# user_email arrives via the OBO header chain. A FastAPI middleware below
# parks it in a contextvar that this subclass reads — the alternative
# (modifying the AG-UI request body) was rejected because it would
# require re-encoding the body in the reviewer-app proxy.
import contextvars as _contextvars  # noqa: E402

_agui_user_email: _contextvars.ContextVar[str] = _contextvars.ContextVar(
    "_agui_user_email", default="unknown"
)


@app.middleware("http")
async def _capture_forwarded_user(request: Request, call_next):
    """Park x-forwarded-email into a contextvar so the AG-UI agent run
    can tag its MLflow trace without needing direct access to the
    incoming FastAPI Request (which ag_ui_langgraph hides from us).

    Critically, do NOT reset the contextvar in a `finally` block — for
    StreamingResponse the body generator iterates AFTER the middleware
    returns, so a reset there would wipe the value before
    `_TracedAGUIAgent.run` ever runs. The contextvar is per-task; it
    cleans up automatically when the request task ends.
    """
    if request.url.path.startswith("/copilotkit"):
        raw_email = (
            request.headers.get("x-forwarded-email")
            or request.headers.get("x-forwarded-preferred-username")
            or request.headers.get("x-forwarded-user")
            or ""
        )
        resolved = raw_email if "@" in raw_email else "unknown"
        _agui_user_email.set(resolved)
        logger.info(
            "agui middleware captured user=%s (raw=%r) for path=%s",
            resolved,
            raw_email[:40],
            request.url.path,
        )
    return await call_next(request)


def _extract_latest_user_text(input_obj) -> str:
    """Pull the freshest user message content out of a RunAgentInput.
    Tolerant of dict-vs-Pydantic message shapes."""
    msgs = getattr(input_obj, "messages", None) or []
    for m in reversed(msgs):
        role = getattr(m, "role", None) or (
            m.get("role") if isinstance(m, dict) else None
        )
        if role != "user":
            continue
        content = getattr(m, "content", None) or (
            m.get("content") if isinstance(m, dict) else None
        )
        if content:
            return content
    return ""


class _TracedAGUIAgent(LangGraphAgent):
    def __init__(self, *args, tier_agents=None, **kwargs):
        # tier_agents: {tier -> LangGraphAgent} — INDEPENDENT per-tier instances
        # we dispatch to per conversation. We never mutate self.graph: `self` is
        # a single instance shared across all concurrent conversations, so
        # swapping its graph would bleed one conversation's tier onto another.
        # None → routing off; run() falls back to self.graph via super().run().
        super().__init__(*args, **kwargs)
        self._tier_agents = tier_agents or None

    def clone(self):
        """Carry the tier agents onto the per-request copy.

        add_langgraph_fastapi_endpoint does `request_agent = agent.clone()` on
        EVERY request, and the base clone() rebuilds via
        `type(self)(name=, graph=, description=, config=)` — it cannot know about
        an extra __init__ parameter, so `tier_agents` defaulted to None and
        `self._tier_agents` was None for every turn the app ever served. The base
        docstring says exactly this: "Subclasses that add required __init__
        parameters must override clone() to pass those parameters through."

        Because the clone is still a _TracedAGUIAgent, every OTHER symptom looked
        healthy, which is what made this hard to see: boot logged 3 tier graphs,
        /health reported routing enabled, the agent_turn_agui span was created,
        and trace_marker dispatched. Only `if self._tier_agents:` silently failed
        — and a skipped branch logs nothing — so the single visible symptom was
        the ABSENCE of a routing_marker event. The tier selector rendered, the
        client sent agent_tier, and the server ignored it 100% of the time.
        """
        cloned = super().clone()
        cloned._tier_agents = self._tier_agents
        return cloned

    async def run(self, input):
        import mlflow as _mlflow

        thread_id = getattr(input, "thread_id", None) or "unknown"
        # Pin this conversation to one gateway destination for prefix-cache
        # reuse + a consistent model across turns. Set here (the async task that
        # calls the graph) so the async httpx auth hook sees it on every LLM
        # call. "unknown" is fine — a stable-per-conversation key is all the
        # gateway's rendezvous hashing needs.
        set_llm_session_id(thread_id)
        # Prefer user_email from the AG-UI body's forwardedProps — the
        # reviewer-app proxy injects it there because Databricks Apps
        # app-to-app calls overwrite `x-forwarded-email` with the
        # calling app's SP UUID, making header-based identity unreliable.
        # Fall back to the header-derived contextvar in case a future
        # client posts to /copilotkit directly with the right header.
        fp = getattr(input, "forwarded_props", None) or {}
        fp_email = fp.get("user_email") if isinstance(fp, dict) else None
        user_email = (
            fp_email
            if isinstance(fp_email, str) and "@" in fp_email
            else _agui_user_email.get("unknown")
        )
        # The in-document reviewer assistant injects the open document id via
        # forwardedProps.document_id. Park it (and the resolved user_email) in
        # the tools' contextvars so the reviewer-action + memory tools are
        # scoped to the right document and reviewer. Setting these BEFORE the
        # graph runs is what lets the sync tools (run in an executor thread by
        # LangChain, which copies the context) read them. The legacy /responses
        # path sets set_authorized_user_email itself; the AG-UI path did not,
        # so this also closes that gap for the memory tools.
        fp_document_id = fp.get("document_id") if isinstance(fp, dict) else None
        set_authorized_user_email(user_email)
        set_active_document_id(fp_document_id)

        user_text = _extract_latest_user_text(input)

        # Complexity-tiered routing: pick the per-tier agent (an INDEPENDENT
        # instance) for this conversation. Off/unresolved → self.graph via
        # super().run(). A resolution failure never breaks the stream.
        routing_tier = None
        routing_source = None
        requested = None
        inner_agent = None
        # Clear first, unconditionally. A contextvar that survived from an earlier
        # turn would badge this one with a tier it never used, and "routing is
        # off" has to look like no badge rather than the last one.
        try:
            from agent.routing import set_routing_decision as _clear_routing

            _clear_routing(None, None)
        except Exception:  # pragma: no cover — cosmetic
            pass

        # Per-turn outcome, surfaced on /health. Boot state alone was not enough:
        # the live app reported routing enabled with 3 tier graphs while NO turn
        # ever emitted a routing_marker, and nothing distinguished "the block was
        # skipped" from "resolve_tier raised" from "it worked and the marker was
        # lost downstream". Counters, not a log line, because app logs here are a
        # short rolling window that had already rotated past every request.
        _ROUTING_TURNS["total"] = _ROUTING_TURNS.get("total", 0) + 1
        if not self._tier_agents:
            _ROUTING_TURNS["skipped_no_tier_agents"] = (
                _ROUTING_TURNS.get("skipped_no_tier_agents", 0) + 1
            )

        if self._tier_agents:
            try:
                from agent.routing import (
                    normalize_requested_tier,
                    resolve_tier,
                    set_routing_decision,
                )

                # The UI's tier selector rides forwardedProps beside document_id
                # and user_email — the same channel, for the same reason: it is
                # the one part of the AG-UI body a client controls end to end.
                # None/"auto" reproduces the pre-selector behavior exactly.
                requested = normalize_requested_tier(
                    fp.get("agent_tier") if isinstance(fp, dict) else None
                )
                routing_tier, routing_source = await resolve_tier(
                    thread_id, user_text, requested=requested
                )
                # Readable inside the graph, where post_model_hook turns it into
                # the routing_marker event the selector needs to show an
                # escalation. Set BEFORE the graph runs.
                set_routing_decision(routing_tier, routing_source)
                inner_agent = self._tier_agents.get(routing_tier)
                _ROUTING_TURNS["resolved"] = _ROUTING_TURNS.get("resolved", 0) + 1
                _ROUTING_TURNS["last_source"] = routing_source
                _ROUTING_TURNS["last_tier"] = routing_tier
            except Exception as e:
                logger.warning("AG-UI tier routing failed; using default graph: %s", e)
                _ROUTING_TURNS["failed"] = _ROUTING_TURNS.get("failed", 0) + 1
                _ROUTING_TURNS["last_error"] = f"{type(e).__name__}: {e}"[:200]

        assistant_text_parts: list[str] = []

        with _mlflow.start_span(name="agent_turn_agui", span_type="CHAIN") as span:
            # session.thread_id / session.user_email tags + matching
            # metadata — exact same shape as the legacy /responses path
            # so MLflow UI filters and Scheduled Scorer queries work
            # identically on AG-UI traces.
            try:
                set_session_context(thread_id=thread_id, user_email=user_email)
            except Exception as e:
                logger.debug("set_session_context (AG-UI) failed: %s", e)

            # Inputs: canonical chat-completion shape. parse_inputs_to_str
            # in MLflow's trace_utils recognizes `messages` and returns a
            # clean string — what the Scheduled Scorers consume.
            # `guidelines_context` is a top-level inputs key for the
            # Databricks Guidelines scorer.
            try:
                span.set_inputs(
                    {
                        "messages": [{"role": "user", "content": user_text}],
                        "guidelines_context": "",
                    }
                )
            except Exception as e:
                logger.debug("span.set_inputs (AG-UI) failed: %s", e)

            # Pre-stream preview so the trace appears in the UI even if
            # the run errors out before producing any assistant text.
            try:
                _mlflow.update_current_trace(request_preview=user_text[:200])
            except Exception:
                pass

            # Tag the routing decision so it is OBSERVABLE (measurable savings,
            # and a router stuck on default_on_error surfaces in traces/alerts).
            if routing_source is not None:
                try:
                    from agent.routing import high_risk_kind as _cue_kind

                    span.set_attributes(
                        {
                            "routing.tier": routing_tier or "",
                            "routing.source": routing_source,
                            # The complexity-vs-stakes axis, kept separately from
                            # `source`. When an explicit selection is escalated the
                            # source becomes user_override_escalated and would
                            # otherwise LOSE which cue drove it — and that split is
                            # the whole reason high_risk_kind distinguishes them.
                            "routing.cue": _cue_kind(user_text) or "",
                            # What the reviewer asked for, so "how often is the
                            # selector overridden" and "what do reviewers pick" are
                            # answerable in SQL rather than by inference.
                            "routing.requested": requested or "",
                        }
                    )
                except Exception:
                    pass

            try:
                # Stream from the per-tier instance when routing resolved one;
                # else the default graph via super().run(). Never mutate
                # self.graph (shared across concurrent conversations).
                _event_source = (
                    inner_agent.run(input)
                    if inner_agent is not None
                    else super().run(input)
                )
                async for event in _event_source:
                    # Accumulate assistant text deltas for set_outputs at
                    # the end. ag_ui_langgraph yields typed Pydantic
                    # events; access by attribute, fall back to dict.
                    try:
                        et = getattr(event, "type", None)
                        if et is None and isinstance(event, dict):
                            et = event.get("type")
                        if et == "TEXT_MESSAGE_CONTENT":
                            delta = getattr(event, "delta", None)
                            if delta is None and isinstance(event, dict):
                                delta = event.get("delta")
                            if delta:
                                assistant_text_parts.append(delta)
                    except Exception:
                        pass
                    yield event

                final_text = "".join(assistant_text_parts).strip()
                if final_text:
                    try:
                        span.set_outputs(
                            {"messages": [{"role": "assistant", "content": final_text}]}
                        )
                    except Exception as e:
                        logger.debug("span.set_outputs (AG-UI) failed: %s", e)
                    try:
                        _mlflow.update_current_trace(response_preview=final_text[:200])
                    except Exception:
                        pass
            except Exception as e:
                try:
                    _mlflow.update_current_trace(
                        tags={"error": "true", "error_type": type(e).__name__}
                    )
                except Exception:
                    pass
                raise


# Build the AG-UI graph at module import. `get_agui_graph()` resolves
# the active prompt via `mlflow.genai.load_prompt(...)` and may emit
# autolog spans during ChatOpenAI construction — both would otherwise
# show up as orphan root traces in the experiment. Suppress tracing
# while the boot-time graph is being assembled; per-turn tracing is
# unaffected (autolog is re-enabled by the context manager on exit).
from services.tracing import tracing_disabled as _boot_tracing_disabled

with _boot_tracing_disabled():
    _agui_graph_at_boot = get_agui_graph()

    # Complexity-tiered routing (AG-UI): pre-compile ONE graph per tier at boot
    # and wrap each in its own PLAIN LangGraphAgent. The dispatcher
    # (_TracedAGUIAgent) picks one per conversation and delegates streaming to
    # it — tracing happens once in the dispatcher, so the inner agents stay
    # plain. Off → single graph, byte-for-byte today.
    _agui_tier_agents = None
    try:
        from config import settings as _boot_settings

        if _boot_settings.routing_enabled:
            from agent.routing import tier_to_endpoint_effort

            _agui_tier_agents = {}
            for _tier in ("low", "med", "high"):
                _ep, _eff = tier_to_endpoint_effort(_tier)
                _g = get_agui_graph(
                    llm_endpoint=_ep,
                    reasoning_effort=_eff,
                    cache_key=f"{_ep or 'blend'}|{_eff}",
                )
                _agui_tier_agents[_tier] = LangGraphAgent(
                    name="lakercm_agent",
                    description=("LakeRCM medical document intelligence assistant."),
                    graph=_g,
                )
            _distinct = len({id(a.graph) for a in _agui_tier_agents.values()})
            logger.info(
                "AG-UI complexity routing enabled: %d distinct tier graph(s)",
                _distinct,
            )
            _ROUTING_BOOT.update(
                enabled=True,
                # Overwrite the "not initialized" default. Leaving it produced a
                # live /health of {"enabled": true, "reason": "not initialized"},
                # which reads as a contradiction and cost real time to discount.
                reason="ok",
                tiers=sorted(_agui_tier_agents),
                distinct_graphs=_distinct,
            )
        else:
            _ROUTING_BOOT.update(enabled=False, reason="LLM_ROUTING_ENABLED is false")
    except Exception as e:
        logger.warning("AG-UI tier-graph setup failed; using single graph: %s", e)
        _agui_tier_agents = None
        # Recorded, not just logged. This except swallows a real outage: the
        # selector keeps rendering and the client keeps sending agent_tier, but
        # the server ignores it, and the ONLY external symptom is the ABSENCE of
        # a routing_marker event — which is invisible unless you diff an SSE
        # stream. A dev chasing a dead tier selector needs one request, not a
        # boot log that has already rotated.
        _ROUTING_BOOT.update(enabled=False, reason=f"tier-graph setup failed: {e}")

add_langgraph_fastapi_endpoint(
    app=app,
    agent=_TracedAGUIAgent(
        name="lakercm_agent",
        description="LakeRCM medical document intelligence assistant.",
        graph=_agui_graph_at_boot,
        tier_agents=_agui_tier_agents,
    ),
    path="/copilotkit",
)
logger.info("Mounted AG-UI endpoint at /copilotkit")


@app.get("/feedback")
async def get_feedback(trace_ids: str, request: Request):
    """Return the latest user_satisfaction Assessment value per trace_id.

    Query string:
        trace_ids = comma-separated list of trace_ids (URL-encoded if they
        contain `:` or `/` — they often do for UC-backed traces).

    Response:
        {"<trace_id>": true | false | null, ...}

    Used by the reviewer app on conversation history load so the thumbs
    selected state survives reload. We pull the assessments from the trace
    rather than maintaining a parallel index — the UC trace tables are the
    source of truth.
    """
    from mlflow import MlflowClient

    _raw_email = (
        request.headers.get("x-forwarded-email")
        or request.headers.get("x-forwarded-preferred-username")
        or request.headers.get("x-forwarded-user")
        or ""
    )
    user_email = _raw_email if "@" in _raw_email else None

    raw_ids = [t for t in (trace_ids or "").split(",") if t.strip()]
    if not raw_ids:
        return {}

    client = MlflowClient()
    out: dict = {}
    for trace_id in raw_ids:
        try:
            trace = client.get_trace(trace_id, display=False)
        except Exception:
            out[trace_id] = None
            continue

        assessments = list(getattr(trace.info, "assessments", None) or [])
        # Filter to user_satisfaction Assessments authored by this user (if known)
        # so a reviewer doesn't see another reviewer's feedback as their own.
        candidates = []
        for a in assessments:
            if getattr(a, "name", None) != "user_satisfaction":
                continue
            source = getattr(a, "source", None)
            source_id = getattr(source, "source_id", None) if source else None
            if user_email and source_id and source_id != user_email:
                continue
            candidates.append(a)

        if not candidates:
            out[trace_id] = None
            continue

        # Latest wins. Use create_time_ms / created_at if present, else last.
        def _ts(a):
            return (
                getattr(a, "create_time_ms", None)
                or getattr(a, "created_at", None)
                or 0
            )

        latest = max(candidates, key=_ts)
        feedback_obj = getattr(latest, "feedback", None)
        value = getattr(feedback_obj, "value", None) if feedback_obj else None
        out[trace_id] = bool(value) if isinstance(value, bool) else None

    return out


@app.post("/feedback")
async def feedback(request: Request):
    """Attach user feedback to an MLflow trace as an Assessment.

    Request body:
        {
            "trace_id": "tr-...",
            "value": true | false,
            "rationale": "optional comment",
            "name": "user_satisfaction" (optional)
        }

    Feedback is bound to the trace produced by the originating /responses call
    (the UI captured the trace_id from the SSE stream).
    """
    import mlflow
    from mlflow.entities import AssessmentSource

    body = await request.json()
    trace_id = body.get("trace_id")
    value = body.get("value")
    if not trace_id or value is None:
        return {"status": "error", "detail": "trace_id and value are required"}

    # Prefer body.user_email — Databricks Apps app-to-app calls overwrite
    # `x-forwarded-email` with the calling app's SP UUID, so the header is
    # unreliable in the reviewer-app → agent-app hop. The reviewer-app
    # feedback proxy now injects user_email into the body for the same
    # reason the AG-UI proxy injects forwardedProps.user_email.
    body_email = body.get("user_email") if isinstance(body, dict) else None
    if isinstance(body_email, str) and "@" in body_email:
        user_email = body_email
    else:
        _raw_email = (
            request.headers.get("x-forwarded-email")
            or request.headers.get("x-forwarded-preferred-username")
            or request.headers.get("x-forwarded-user")
            or ""
        )
        user_email = _raw_email if "@" in _raw_email else "unknown"

    try:
        mlflow.log_feedback(
            trace_id=trace_id,
            name=body.get("name", "user_satisfaction"),
            value=bool(value),
            rationale=body.get("rationale"),
            source=AssessmentSource(source_type="HUMAN", source_id=user_email),
        )
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}


@app.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str, request: Request):
    """Remove all checkpointer state for a conversation (thread_id).

    Called by the reviewer app as part of cascade-delete when a user deletes
    a conversation. Long-term memories in public.store are user-scoped, not
    conversation-scoped, and are intentionally preserved.

    Idempotent: deleting a missing thread returns 200 with status=skipped.

    No MLflow span is opened here: thread deletes are admin/cleanup
    activity, not user agent turns, and a per-delete trace is
    operational noise (it gets sampled by scheduled scorers that have
    no useful request/response to score). The DELETE keeps its log
    lines for observability.
    """
    if not thread_id or len(thread_id) > 64:
        raise HTTPException(status_code=400, detail="invalid thread_id")

    _raw_email = (
        request.headers.get("x-forwarded-email")
        or request.headers.get("x-forwarded-preferred-username")
        or request.headers.get("x-forwarded-user")
        or ""
    )
    user_email = _raw_email if "@" in _raw_email else "unknown"
    logger.info("thread_delete request thread_id=%s user=%s", thread_id, user_email)

    try:
        result = delete_checkpointer_thread(thread_id)
        return {
            "status": "ok",
            "checkpointer": result,
            "memory": "skipped",
            "thread_id": thread_id,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {str(exc)[:300]}",
        )


@app.post("/responses")
async def responses(request: Request):
    """Stream agent response as Server-Sent Events.

    Request body:
        {
            "messages": [{"role": "user", "content": "..."}],
            "user_context": {"user_email": "reviewer@example.com"}
        }

    SSE events emitted:
        {"type": "thinking", "content": "..."}
        {"type": "tool_call", "tool": "...", "input": {...}, "status": "running"}
        {"type": "tool_result", "tool": "...", "output_preview": "...", "status": "completed"}
        {"type": "token", "content": "..."}
        {"type": "done"}
    """
    body = await request.json()

    # Extract messages
    messages = body.get("messages") or body.get("input") or []
    conversation_id = body.get("conversation_id")

    # Server-side user resolution from forwarded headers
    _raw_email = (
        request.headers.get("x-forwarded-email")
        or request.headers.get("x-forwarded-preferred-username")
        or request.headers.get("x-forwarded-user")
        or ""
    )
    email = _raw_email if "@" in _raw_email else ""
    server_user = get_user_for_email(email)

    # Allow client to override user_email (for proxied requests)
    user_context_body = body.get("user_context") or {}
    user_context = {
        "user_email": user_context_body.get("user_email") or server_user["user_email"],
        "display_name": server_user.get("display_name", "Reviewer"),
        "first_name": server_user.get("first_name", "there"),
        "role": server_user.get("role", "Reviewer"),
    }

    # With a PostgresSaver checkpointer, the agent persists message history
    # itself under thread_id=conversation_id. We only need to pass the NEW
    # user turn — LangGraph's add_messages reducer appends it to whatever
    # state the checkpointer already holds. This avoids duplicating the
    # reviewer app's replay of the full transcript into LangGraph state.
    latest_user_content = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_content = msg.get("content", "")

    if not latest_user_content:

        async def empty():
            yield f"data: {json.dumps({'type': 'token', 'content': 'Please send a message to get started.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(empty(), media_type="text/event-stream")

    req_id = str(uuid4())[:8]
    thread_id = conversation_id or str(uuid4())
    log_extra = {"req_id": req_id, "thread_id": thread_id}
    turn_start = time.monotonic()

    logger.info(
        "agent_turn_start user=%s message_chars=%d",
        user_context["user_email"],
        len(latest_user_content),
        extra=log_extra,
    )

    async def event_stream():
        # Emit thinking indicator — land this before any risky setup so the UI
        # always shows activity, even if the subsequent block fails.
        yield f"data: {json.dumps({'type': 'thinking', 'content': 'Searching document records...', 'req_id': req_id})}\n\n"

        try:
            import asyncio
            import mlflow

            # Set thread-local user email for audit + build agent. Moved inside
            # the try block so failures here (missing state schema keys, bad
            # LLM endpoint, etc.) surface as {type:error} events instead of
            # tearing down the chunked HTTP stream mid-flight.
            set_authorized_user_email(user_context["user_email"])

            # Complexity-tiered routing: resolve + pin this conversation's tier,
            # then point the agent at that tier's model service + reasoning
            # effort. Feature-flagged — when OFF, endpoint/effort stay None and
            # create_agent is byte-for-byte today's call. A resolution failure
            # falls through to the default (blend) rather than breaking the turn.
            from config import settings as _settings

            routing_tier = None
            routing_source = None
            _tier_endpoint = None
            _tier_effort = None
            _requested = None
            if _settings.routing_enabled:
                try:
                    from agent.routing import (
                        normalize_requested_tier,
                        resolve_tier,
                        set_routing_decision,
                        tier_to_endpoint_effort,
                    )

                    # Parity with the AG-UI path. This endpoint takes a JSON body
                    # rather than forwardedProps, so the selection arrives in
                    # user_context — no client uses it today, but leaving the two
                    # paths with different routing behavior is how they drift.
                    _requested = normalize_requested_tier(
                        (user_context or {}).get("agent_tier")
                        if isinstance(user_context, dict)
                        else None
                    )
                    routing_tier, routing_source = await resolve_tier(
                        thread_id, latest_user_content, requested=_requested
                    )
                    set_routing_decision(routing_tier, routing_source)
                    _tier_endpoint, _tier_effort = tier_to_endpoint_effort(routing_tier)
                    logger.info(
                        "routing tier=%s source=%s endpoint=%s effort=%s",
                        routing_tier,
                        routing_source,
                        _tier_endpoint or "(blend)",
                        _tier_effort,
                        extra=log_extra,
                    )
                except Exception as e:
                    logger.warning(
                        "tier routing failed; using default endpoint/effort: %s",
                        e,
                        extra=log_extra,
                    )
            agent, prompt_resolution = create_agent(
                user_context,
                thread_id=thread_id,
                llm_endpoint=_tier_endpoint,
                reasoning_effort=_tier_effort,
            )
            input_state = {"messages": [HumanMessage(content=latest_user_content)]}
            # recursion_limit caps the agent loop so a runaway plan (LLM
            # ignores tool result, re-calls same tool) fails fast at ~5
            # full agent-tool round-trips instead of accumulating 80+
            # messages and timing out with an empty error.
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": int(
                    os.getenv("LAKERCM_AGENT_RECURSION_LIMIT", "12")
                ),
            }

            # Token-level streaming via LangChain callbacks.
            # We can't use agent.astream_events here because our checkpointer
            # is sync-only PostgresSaver and astream requires aget_tuple.
            # Instead, run agent.invoke in a background thread with a sync
            # callback handler that publishes LLM tokens + tool events to a
            # thread-safe queue. The SSE generator drains the queue as
            # events arrive.
            #
            # The mlflow.start_span("agent_turn") block lives INSIDE the
            # worker thread (not the main thread) — OpenTelemetry context is
            # thread-local, so autolog's child spans (LangGraph nodes, tools,
            # LLM calls) only nest under our root if the root opens on the
            # same thread agent.invoke runs on. This matches Databricks's
            # canonical Mosaic AI Agent pattern at
            # mlflow/genai/agent_server/server.py:313-330.
            import queue as _queue
            import threading

            from langchain_core.callbacks import BaseCallbackHandler

            _SENTINEL = object()

            class _StreamHandler(BaseCallbackHandler):
                def __init__(self):
                    self.q = _queue.Queue()
                    # Track tool execution wall-clock time. Keyed by
                    # LangChain run_id (UUID per tool invocation, unique
                    # even when the same tool is called twice in a turn).
                    # Server-measured timing is the source of truth — the
                    # frontend's SSE-arrival diff was always near-zero
                    # because the events are queued back-to-back.
                    self.tool_starts: dict = {}

                def on_llm_new_token(self, token, **kwargs):  # noqa: D401
                    if token:
                        self.q.put(("token", token))

                def on_tool_start(self, serialized, input_str, **kwargs):
                    name = (serialized or {}).get("name") or "tool"
                    run_id = str(kwargs.get("run_id") or "")
                    try:
                        tool_input = json.loads(input_str) if input_str else {}
                    except Exception:
                        tool_input = {"raw": str(input_str)[:200]}
                    if run_id:
                        self.tool_starts[run_id] = time.monotonic()
                    self.q.put(("tool_start", name, tool_input, run_id))

                def on_tool_end(self, output, **kwargs):
                    name = kwargs.get("name") or "tool"
                    run_id = str(kwargs.get("run_id") or "")
                    out_str = (
                        output.content
                        if hasattr(output, "content")
                        else str(output) if output is not None else ""
                    )
                    if not isinstance(out_str, str):
                        out_str = str(out_str)
                    if len(out_str) > 500:
                        out_str = out_str[:500] + "..."
                    start = self.tool_starts.pop(run_id, None) if run_id else None
                    duration_ms = (
                        int((time.monotonic() - start) * 1000)
                        if start is not None
                        else None
                    )
                    self.q.put(("tool_end", name, out_str, run_id, duration_ms))

            handler = _StreamHandler()
            cb_config = {
                **config,
                "callbacks": [handler],
            }

            result_holder: dict = {}

            def _run_agent():
                # Pin this conversation to one gateway destination. MUST be set
                # HERE (inside the worker thread) — contextvars do not cross the
                # thread boundary from the request task, so the sync httpx auth
                # hook that runs on THIS thread only sees the session id if it's
                # set on this thread. thread_id is captured from the closure.
                set_llm_session_id(thread_id)
                # Open the agent_turn span IN the worker thread so
                # mlflow.langchain.autolog()'s spans (LangGraph nodes,
                # ToolNode, LLM calls) nest under it as children. If the
                # span opens on a different thread, OTel context isn't
                # shared and autolog produces a separate orphan trace.
                try:
                    with mlflow.start_span(
                        name="agent_turn", span_type="CHAIN"
                    ) as span:
                        # Trace-level metadata. Calls update_current_trace
                        # internally — must run while the span is active.
                        set_session_context(
                            thread_id=thread_id,
                            user_email=user_context["user_email"],
                        )
                        set_prompt_context(
                            name=prompt_resolution.name,
                            version=prompt_resolution.version,
                            alias_used=prompt_resolution.alias_used,
                        )

                        # Tag the routing decision so it is OBSERVABLE — a router
                        # stuck on the default tier (source=default_on_error) must
                        # surface in traces/alerts, never silently downgrade; and
                        # this is what makes the cost-savings measurable.
                        if routing_source is not None:
                            try:
                                from agent.routing import (
                                    high_risk_kind as _cue_kind,
                                )

                                span.set_attributes(
                                    {
                                        "routing.tier": routing_tier or "",
                                        "routing.source": routing_source,
                                        "routing.cue": _cue_kind(latest_user_content)
                                        or "",
                                        "routing.requested": _requested or "",
                                    }
                                )
                            except Exception:
                                pass

                        # Capture trace_id while the span is alive — emit
                        # back to main thread via result_holder.
                        try:
                            tid = (
                                getattr(span, "trace_id", None)
                                or getattr(span, "request_id", None)
                                or (
                                    mlflow.get_current_active_span().trace_id
                                    if mlflow.get_current_active_span()
                                    else None
                                )
                            )
                            if tid:
                                result_holder["trace_id"] = tid
                        except Exception:
                            pass

                        # Run the agent. autolog spans nest under agent_turn.
                        # One-shot retry on transient Lakebase connection errors
                        # — typical when Postgres rotates a pooled conn during
                        # autoscale/maintenance. The pool's check_connection
                        # should evict the bad conn before getconn() returns,
                        # so the second invoke gets a fresh handle.
                        try:
                            result_holder["result"] = agent.invoke(
                                input_state, cb_config
                            )
                        except (
                            psycopg.errors.AdminShutdown,
                            psycopg.OperationalError,
                        ) as conn_err:
                            logger.warning(
                                "transient_lakebase_conn_error retrying_once exc=%s msg=%s",
                                type(conn_err).__name__,
                                str(conn_err)[:200],
                                extra=log_extra,
                            )
                            try:
                                stats = get_db().get_pool().get_stats()
                                logger.warning(
                                    "pool_stats_pre_retry %s",
                                    stats,
                                    extra=log_extra,
                                )
                            except Exception:
                                pass
                            result_holder["result"] = agent.invoke(
                                input_state, cb_config
                            )

                        # Set request/response on the SAME span so the
                        # trace's inputs/outputs columns reflect the user
                        # prompt + the agent's final assistant text. Use
                        # canonical chat-completion shape — MLflow's
                        # parse_inputs_to_str / parse_outputs_to_str
                        # (trace_utils.py:586-631) recognize messages and
                        # return clean strings, which is what Databricks
                        # scheduled scorers (Safety, Correctness,
                        # RelevanceToQuery, Guidelines) consume.
                        # `guidelines_context` is preserved as a top-level
                        # inputs key for the Databricks Guidelines scorer.
                        try:
                            final_response_text = ""
                            result_obj = result_holder.get("result") or {}
                            msgs = (
                                result_obj.get("messages", [])
                                if isinstance(result_obj, dict)
                                else []
                            )
                            for m in reversed(msgs):
                                content = getattr(m, "content", None)
                                if isinstance(content, str) and content.strip():
                                    final_response_text = content
                                    break
                            span.set_inputs(
                                {
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": latest_user_content,
                                        }
                                    ],
                                    "guidelines_context": "",
                                }
                            )
                            if final_response_text:
                                span.set_outputs(
                                    {
                                        "messages": [
                                            {
                                                "role": "assistant",
                                                "content": final_response_text,
                                            }
                                        ],
                                    }
                                )
                                try:
                                    mlflow.update_current_trace(
                                        request_preview=latest_user_content[:200],
                                        response_preview=final_response_text[:200],
                                    )
                                except Exception:
                                    pass
                        except Exception as _trace_io_err:
                            logger.warning(
                                "trace_io_update_skipped exc=%s",
                                _trace_io_err,
                                extra=log_extra,
                            )
                except Exception as e:
                    result_holder["error"] = e
                finally:
                    handler.q.put(_SENTINEL)

            worker = threading.Thread(target=_run_agent, daemon=True)
            worker.start()

            emitted_any_token = False
            loop = asyncio.get_event_loop()
            while True:
                item = await loop.run_in_executor(None, handler.q.get)
                if item is _SENTINEL:
                    break
                kind = item[0]
                if kind == "token":
                    emitted_any_token = True
                    yield (
                        f"data: "
                        f"{json.dumps({'type': 'token', 'content': item[1]})}"
                        f"\n\n"
                    )
                elif kind == "tool_start":
                    # item = ("tool_start", name, input, run_id)
                    yield (
                        f"data: "
                        f"{json.dumps({'type': 'tool_call', 'tool': item[1], 'input': item[2], 'status': 'running', 'run_id': item[3]}, default=str)}"
                        f"\n\n"
                    )
                elif kind == "tool_end":
                    # item = ("tool_end", name, output, run_id, duration_ms)
                    yield (
                        f"data: "
                        f"{json.dumps({'type': 'tool_result', 'tool': item[1], 'output_preview': item[2], 'status': 'completed', 'run_id': item[3], 'duration_ms': item[4]})}"
                        f"\n\n"
                    )

            worker.join(timeout=5)

            # Emit trace event AFTER the worker has finished so we have the
            # trace_id captured from inside the span context. Frontend
            # stores trace_id on the assistant message and renders the
            # "View trace" link only when the tool card is expanded
            # post-stream — late emission is fine.
            trace_id = result_holder.get("trace_id")
            if trace_id:
                workspace_host = ""
                try:
                    from agent.graph import _resolve_host as _rh

                    workspace_host = _rh()
                except Exception:
                    pass
                experiment_id = ""
                try:
                    exp = mlflow.get_experiment_by_name(
                        os.getenv(
                            "MLFLOW_EXPERIMENT_NAME",
                            "/Shared/lakercm/agent-traces",
                        )
                    )
                    if exp is not None:
                        experiment_id = str(exp.experiment_id)
                except Exception:
                    pass
                yield f"data: {json.dumps({'type': 'trace', 'trace_id': trace_id, 'workspace_host': workspace_host, 'experiment_id': experiment_id})}\n\n"

            if "error" in result_holder:
                raise result_holder["error"]
            if not emitted_any_token:
                yield (
                    f"data: "
                    f"{json.dumps({'type': 'token', 'content': 'I was unable to generate a response. Please try again.'})}"
                    f"\n\n"
                )

            latency_ms = int((time.monotonic() - turn_start) * 1000)
            try:
                result_obj = result_holder.get("result") or {}
                msgs = (
                    result_obj.get("messages", [])
                    if isinstance(result_obj, dict)
                    else []
                )
                new_msg_count = len(msgs)
                tool_call_count = sum(1 for m in msgs if getattr(m, "tool_calls", None))
            except Exception:
                new_msg_count = -1
                tool_call_count = -1
            logger.info(
                "agent_turn_end status=ok latency_ms=%d total_msgs=%d tool_calls=%d trace_id=%s",
                latency_ms,
                new_msg_count,
                tool_call_count,
                result_holder.get("trace_id") or "-",
                extra=log_extra,
            )

        except Exception as exc:
            latency_ms = int((time.monotonic() - turn_start) * 1000)
            is_conn_blip = isinstance(
                exc, (psycopg.errors.AdminShutdown, psycopg.OperationalError)
            )
            logger.exception(
                "agent_turn_failed latency_ms=%d exc=%s is_conn_blip=%s",
                latency_ms,
                type(exc).__name__,
                is_conn_blip,
                extra=log_extra,
            )
            if is_conn_blip:
                try:
                    stats = get_db().get_pool().get_stats()
                    logger.warning(
                        "pool_stats_on_error %s",
                        stats,
                        extra=log_extra,
                    )
                except Exception:
                    pass
            try:
                import mlflow as _mlflow

                _mlflow.update_current_trace(
                    tags={"error": "true", "error_type": type(exc).__name__}
                )
            except Exception:
                pass
            # Recognize LangGraph recursion limit specifically so the user
            # sees actionable guidance instead of generic "something went
            # wrong" — this fires when the agent is looping on tool calls.
            if (
                "GraphRecursionError" in type(exc).__name__
                or "recursion" in str(exc).lower()
            ):
                friendly = (
                    "I couldn't complete that request — please rephrase or "
                    "try a narrower query."
                )
            elif is_conn_blip:
                friendly = "Database connection blipped — please retry."
            else:
                friendly = (
                    "Sorry, something went wrong. Try again or start a new "
                    "conversation."
                )
            yield (
                f"data: "
                f"{json.dumps({'type': 'error', 'content': friendly, 'retryable': True, 'req_id': req_id})}"
                f"\n\n"
            )

        yield f"data: {json.dumps({'type': 'done', 'req_id': req_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
