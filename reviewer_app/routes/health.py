"""
LakeRCM Health Check Routes
"""

from fastapi import APIRouter, Depends
from datetime import datetime

from schemas import HealthResponse
from dependencies import get_lakercm_db, get_workspace_client
from services.lakehouse_db import LakeRCMDatabase
from databricks.sdk import WorkspaceClient

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: LakeRCMDatabase = Depends(get_lakercm_db),
    workspace: WorkspaceClient = Depends(get_workspace_client),
):
    databricks_connected = False
    try:
        workspace.current_user.me()
        databricks_connected = True
    except Exception:
        pass

    database_connected = db.health_check()

    if databricks_connected and database_connected:
        status = "healthy"
    elif databricks_connected:
        status = "degraded"
    else:
        status = "unhealthy"

    details = {"message": f"System is {status}"}

    # Whether this deploy can read extractions at all. Without the Lakebase copy
    # of gold_extraction_labels the document list, the review queue and the
    # agent's document tools are all empty however well the pipeline ran, and the
    # app has no other way to say so — dev looked like an empty environment for a
    # day. Degraded, not unhealthy: uploads and conversations still work.
    if database_connected:
        gold_sync_ready = db.gold_sync_available()
        details["gold_sync"] = {
            "table": db.GOLD_SYNC,
            "available": gold_sync_ready,
        }
        if not gold_sync_ready:
            status = "degraded"
            details["message"] = (
                f"System is degraded: {db.GOLD_SYNC} is missing, so no extraction "
                "results can be read. Deploy the lakebase_sync bundle (dev) or "
                "check the synced table and its grants."
            )

    if not database_connected:
        details["database_diagnostics"] = {
            "pg_host_discovered": bool(db.pg_host),
            "pg_user_discovered": bool(db.pg_user),
            "pool_initialized": db.pool is not None,
            "hint": "Check if Lakebase instance is deployed and accessible",
        }

    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        databricks_connected=databricks_connected,
        database_connected=database_connected,
        details=details,
    )
