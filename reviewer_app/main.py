"""
LakeRCM FastAPI Application

Medical document extraction review platform with Admin and Physician views.
"""

import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path


def _boot(msg: str) -> None:
    """Print to stderr with flush — survives process exit better than logger."""
    print(f"[main.boot] {msg}", file=sys.stderr, flush=True)


from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from databricks.sdk import WorkspaceClient

import dependencies
from dependencies import resolve_user_identity
from config import settings
from services.lakehouse_db import LakeRCMDatabase
from services.admin_auth import is_admin_for_request
from services.lakebase_monitor import LakebaseMonitor
from services.otel_fastapi import instrument_fastapi

from routes import (
    health,
    documents,
    analytics,
    chat,
    admin,
    transcription,
    proposals,
    kg,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _run_alembic_upgrade():
    """Apply pending Lakebase migrations (schema + GRANTs)."""
    _boot(f"alembic config path: {MIGRATIONS_DIR / 'alembic.ini'}")
    cfg = AlembicConfig(str(MIGRATIONS_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    _boot("calling alembic_command.upgrade(head)")
    alembic_command.upgrade(cfg, "head")
    _boot("alembic_command.upgrade returned")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _boot("lifespan start")
    logger.info("Initializing LakeRCM Backend...")

    try:
        _boot("step 1: init WorkspaceClient")
        dependencies.workspace_client = WorkspaceClient()
        _boot("step 1: WorkspaceClient ready")

        _boot("step 2: discover SQL warehouse")
        settings.discover_warehouse_id(dependencies.workspace_client)
        _boot(f"step 2: sql_warehouse_id={settings.sql_warehouse_id}")

        _boot("step 3: alembic upgrade head")
        _run_alembic_upgrade()
        _boot("step 3: alembic complete")

        _boot("step 4: init LakeRCMDatabase")
        dependencies.lakercm_db = LakeRCMDatabase(
            workspace_client=dependencies.workspace_client
        )
        _boot("step 4: LakeRCMDatabase ready")

        _boot("step 5: start lakebase_monitor")
        dependencies.lakebase_monitor = LakebaseMonitor(
            workspace_client=dependencies.workspace_client,
            endpoint_name=settings.endpoint_name,
        )
        dependencies.lakebase_monitor.start()
        _boot("step 5: lakebase_monitor started")

        # Resolve agent app URL from name. The bundle template injects
        # LAKERCM_AGENT_APP_NAME (a stable identifier); the URL is a
        # runtime field of the apps API and only resolvable at startup.
        # Skip silently if URL was already provided directly.
        if not settings.agent_app_url:
            agent_name = os.getenv("LAKERCM_AGENT_APP_NAME", "").strip()
            if agent_name:
                try:
                    app_obj = dependencies.workspace_client.apps.get(name=agent_name)
                    resolved = (app_obj.url or "").strip()
                    if resolved:
                        settings.agent_app_url = resolved
                        _boot(
                            f"step 6: resolved agent_app_url={resolved} "
                            f"(from name={agent_name})"
                        )
                    else:
                        _boot(f"step 6: agent app {agent_name!r} found but url empty")
                except Exception as e:
                    _boot(
                        f"step 6: agent app URL resolution failed for "
                        f"{agent_name!r}: {type(e).__name__}: {e}"
                    )
            else:
                _boot(
                    "step 6: skipping agent URL resolve "
                    "(neither LAKERCM_AGENT_APP_URL nor _NAME set)"
                )

        # Speech-to-text availability (graceful degrade — never fatal). When
        # the FM endpoint isn't reachable the mic button is disabled in the UI.
        try:
            dependencies.workspace_client.serving_endpoints.get(
                settings.transcription_endpoint
            )
            dependencies.transcription_available = True
            _boot(
                f"step 7: transcription available "
                f"(endpoint={settings.transcription_endpoint})"
            )
        except Exception as e:
            dependencies.transcription_available = False
            _boot(
                f"step 7: transcription UNAVAILABLE "
                f"(endpoint={settings.transcription_endpoint}): "
                f"{type(e).__name__}: {e}"
            )

        _boot(f"backend ready on http://{settings.api_host}:{settings.api_port}")

    except Exception as e:
        _boot(f"FATAL lifespan exception: {type(e).__name__}: {e}")
        _boot(traceback.format_exc())
        logger.error("Failed to initialize backend: %s", e, exc_info=True)
        raise

    yield

    logger.info("Shutting down LakeRCM Backend...")
    try:
        if dependencies.lakebase_monitor:
            await dependencies.lakebase_monitor.stop()
            logger.info("lakebase_monitor stopped")
    except Exception as e:
        logger.error("Error stopping lakebase_monitor: %s", e, exc_info=True)
    try:
        if dependencies.lakercm_db:
            dependencies.lakercm_db.close()
            logger.info("Database connections closed")
    except Exception as e:
        logger.error("Error during shutdown: %s", e, exc_info=True)

    # Flush + shut down OTel span export so the final batch of spans isn't
    # dropped when the pod stops. Under plain uvicorn (telemetry off) the
    # global provider is a no-op proxy without force_flush/shutdown — guard
    # so shutdown never errors.
    try:
        from opentelemetry import trace as _otel_trace

        provider = _otel_trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush()
            logger.info("OTel span exporter force_flush complete")
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
            logger.info("OTel tracer provider shutdown complete")
    except Exception as e:
        logger.warning("OTel flush/shutdown failed: %s", e)


app = FastAPI(
    title="LakeRCM API",
    description="Medical Document Extraction Review Platform",
    version="1.0.0",
    lifespan=lifespan,
)
# PHI-safe FastAPI OTel instrumentation (query-string stripping) — see
# services/otel_fastapi.py. No-op under plain uvicorn (telemetry off).
instrument_fastapi(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(analytics.router, tags=["Analytics"])
# Agent-assisted review of held documents. Its own router (prefix /api) so
# the proposal lifecycle is not buried in the documents CRUD surface.
app.include_router(proposals.router, tags=["Proposals"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
# Router carries its own /api/transcribe prefix — mount bare.
app.include_router(transcription.router, tags=["Transcription"])
app.include_router(kg.router, tags=["Knowledge Graph"])


# --- MLflow trace deep-link metadata -----------------------------------------
# Each tool call in the chat transcript links out to its MLflow trace:
#   <workspace_host>/ml/experiments/<experiment_id>/traces/<trace-id>
# ToolCallCard already builds that URL and renders the anchor, but it HIDES the
# link when workspace_host is empty, and nothing supplied it: the AG-UI stream
# carries only the trace id (the graph's last_trace_id). Host and experiment are
# workspace-stable, so they do not belong on a per-turn event — they are resolved
# here once and cached, because /api/me is called on every page load and must not
# pay a workspace round-trip each time.
_TRACE_LINK_META: dict[str, str] | None = None


def _trace_link_metadata() -> dict[str, str]:
    """Resolve the workspace host + agent experiment id for trace deep-links.

    Best-effort by design: anything that fails here costs only the "View trace"
    link, so it must never raise into /api/me.
    """
    global _TRACE_LINK_META
    if _TRACE_LINK_META is not None:
        return _TRACE_LINK_META

    # Databricks Apps injects DATABRICKS_HOST into the runtime; fall back to the
    # host the SDK resolved so a locally-run app still produces working links.
    host = (settings.databricks_host or "").strip()
    if not host:
        try:
            host = (dependencies.workspace_client.config.host or "").strip()
        except Exception:  # noqa: BLE001 - link-only; never break /api/me
            host = ""
    host = host.rstrip("/")
    if host and not host.startswith("http"):
        host = f"https://{host}"

    # Resolved via the SDK, not mlflow: the reviewer app does not depend on the
    # mlflow package, and this is the only place it would need it. Without an id
    # ToolCallCard falls back to the un-scoped /ml/traces/<id> route.
    experiment_id = ""
    name = os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip()
    if name:
        try:
            found = dependencies.workspace_client.experiments.get_by_name(
                experiment_name=name
            )
            experiment_id = str(getattr(found.experiment, "experiment_id", "") or "")
        except Exception as exc:  # noqa: BLE001 - degrade to the un-scoped route
            logger.info(
                "MLflow experiment %r not resolved for trace links: %s", name, exc
            )

    _TRACE_LINK_META = {"workspace_host": host, "experiment_id": experiment_id}
    return _TRACE_LINK_META


@app.get("/api/me", tags=["User"])
async def get_current_user(request: Request):
    """Return the currently authenticated user's identity."""
    raw = (
        request.headers.get("x-forwarded-email")
        or request.headers.get("x-forwarded-preferred-username")
        or request.headers.get("x-forwarded-user")
        or "demo@example.com"
    )
    identity = resolve_user_identity(raw)
    trace_link = _trace_link_metadata()
    return {
        "email": identity["email"],
        "display_name": identity["display_name"],
        "is_admin": is_admin_for_request(request),
        "auto_verdict_threshold": settings.auto_verdict_threshold,
        "transcription_available": dependencies.transcription_available,
        # Consumed by ChatInterface to build per-tool-call MLflow trace links.
        "workspace_host": trace_link["workspace_host"],
        "experiment_id": trace_link["experiment_id"],
    }


frontend_dist = Path(__file__).parent / "frontend" / "dist"

if frontend_dist.exists():
    logger.info("Serving frontend from %s", frontend_dist)

    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        requested = (frontend_dist / full_path).resolve()
        if requested.is_file() and str(requested).startswith(
            str(frontend_dist.resolve())
        ):
            return FileResponse(requested)
        return FileResponse(frontend_dist / "index.html")

else:
    logger.warning("Frontend not found at %s - serving API only", frontend_dist)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return HTMLResponse(
            "<h1>LakeRCM API</h1>"
            '<p>Backend running. <a href="/docs">API docs</a></p>'
            "<p><em>Build the frontend: <code>cd frontend && npm run build</code></em></p>"
        )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return HTMLResponse(
        content="<h1>404 Not Found</h1><p>The requested resource was not found.</p>",
        status_code=404,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
