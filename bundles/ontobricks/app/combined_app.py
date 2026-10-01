"""One Databricks App serving OntoBricks' main app and its MCP server.

OntoBricks ships its MCP server as a separate app, and also supports a
``mounted`` mode that runs it inside the main process and calls the main app's
REST API on localhost (``src/mcp-server/server/app.py``). Nothing in the main
app mounts it, so this entrypoint does: ``/mcp`` is served by the MCP ASGI app,
ahead of the main app's session / CSRF / permission middleware (``/api/`` and
``/graphql/`` already bypass it — ``src/shared/fastapi/main.py``), and every
other path goes to the main OntoBricks app. Both lifespans run.

LakeRCM copies this file into the fetched OntoBricks tree at deploy time
(``scripts/fetch_ontobricks_source.py``); no OntoBricks source is modified.

Start it the way OntoBricks starts ``run.py`` (the dev group is kept because
``fastmcp`` lives there):

    uv run --frozen --extra lakebase python combined_app.py
"""

from __future__ import annotations

import contextlib
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
# The MCP server package lives under src/mcp-server; the main app under src.
sys.path.insert(0, os.path.join(_ROOT, "src", "mcp-server"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from back.core.logging import setup_logging  # noqa: E402

setup_logging()

from server.app import create_mcp_server  # noqa: E402
from shared.fastapi.main import create_app  # noqa: E402

MCP_PATH = "/mcp"

# The main OntoBricks FastAPI app (UI + REST /api/v1 + GraphQL + builds).
main_app = create_app()

# The MCP server. In the app it runs in "mounted" mode: it calls the main app
# on http://localhost:$DATABRICKS_APP_PORT (loopback, same process). For a
# local run outside Apps, set OB_MCP_MODE=standalone and ONTOBRICKS_URL to this
# process's own URL.
_mcp_mode = os.getenv("OB_MCP_MODE", "mounted")
mcp_app = create_mcp_server(mode=_mcp_mode).http_app(path=MCP_PATH)


@contextlib.asynccontextmanager
async def _lifespans():
    """Run the main app's lifespan (scheduler, sessions) and FastMCP's together."""
    async with main_app.router.lifespan_context(main_app):
        async with mcp_app.lifespan(mcp_app):
            yield


async def _serve_lifespan(receive, send):
    await receive()  # lifespan.startup
    ctx = _lifespans()
    try:
        await ctx.__aenter__()
    except BaseException as exc:  # noqa: BLE001 — report startup failure to the server
        await send({"type": "lifespan.startup.failed", "message": repr(exc)})
        raise
    await send({"type": "lifespan.startup.complete"})
    await receive()  # lifespan.shutdown
    await ctx.__aexit__(None, None, None)
    await send({"type": "lifespan.shutdown.complete"})


async def app(scope, receive, send):
    """ASGI entrypoint: /mcp -> MCP server, everything else -> the main app."""
    if scope["type"] == "lifespan":
        await _serve_lifespan(receive, send)
        return
    path = scope.get("path", "")
    if path == MCP_PATH or path.startswith(MCP_PATH + "/"):
        await mcp_app(scope, receive, send)
    else:
        await main_app(scope, receive, send)


if __name__ == "__main__":
    import uvicorn

    in_app = os.getenv("DATABRICKS_APP_PORT") is not None
    uvicorn.run(
        app,
        host="0.0.0.0" if in_app else "127.0.0.1",
        port=int(
            os.getenv("DATABRICKS_APP_PORT") or os.getenv("OB_LOCAL_PORT", "8000")
        ),
        log_level="info",
        log_config=None,
    )
