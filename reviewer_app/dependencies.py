"""
LakeRCM Dependency Injection

Global dependencies for services and authentication.
"""

import logging
from typing import Optional, Dict
from databricks.sdk import WorkspaceClient
from services.lakehouse_db import LakeRCMDatabase

logger = logging.getLogger(__name__)

workspace_client: Optional[WorkspaceClient] = None
lakercm_db: Optional[LakeRCMDatabase] = None
lakebase_monitor = None  # type: ignore  # services.lakebase_monitor.LakebaseMonitor

# Set at startup (main.py lifespan): whether the transcription FM endpoint is
# reachable. Surfaced via /api/me so the frontend can disable the mic button
# (rather than hide it) when speech-to-text is unavailable.
transcription_available: bool = False

# Cache: raw user identifier -> {"email": str, "display_name": str | None}
_user_identity_cache: Dict[str, dict] = {}


def get_workspace_client() -> WorkspaceClient:
    if workspace_client is None:
        raise RuntimeError("WorkspaceClient not initialized")
    return workspace_client


def get_lakercm_db() -> LakeRCMDatabase:
    if lakercm_db is None:
        raise RuntimeError("LakeRCMDatabase not initialized")
    return lakercm_db


def resolve_user_identity(raw_id: str) -> dict:
    """Resolve a Databricks user ID or email to {email, display_name}.

    Uses the SCIM Users API with in-memory caching.
    """
    if raw_id in _user_identity_cache:
        return _user_identity_cache[raw_id]

    result = {"email": raw_id, "display_name": None}

    if workspace_client is None:
        _user_identity_cache[raw_id] = result
        return result

    try:
        if raw_id.isdigit():
            user = workspace_client.users.get(raw_id)
            result = {
                "email": user.user_name or raw_id,
                "display_name": user.display_name,
            }
        else:
            safe = raw_id.replace('"', '\\"')
            users = list(
                workspace_client.users.list(filter=f'userName eq "{safe}"', count=1)
            )
            if users:
                result = {
                    "email": users[0].user_name or raw_id,
                    "display_name": users[0].display_name,
                }
    except Exception as e:
        logger.debug("Could not resolve user %s: %s", raw_id, e)

    _user_identity_cache[raw_id] = result
    return result


def get_current_user_email(request) -> str:
    raw = (
        request.headers.get("x-forwarded-email")
        or request.headers.get("x-forwarded-preferred-username")
        or request.headers.get("x-forwarded-user")
        or "demo@example.com"
    )
    # If the header is a numeric ID, resolve to actual email
    if raw.isdigit():
        identity = resolve_user_identity(raw)
        return identity["email"]
    return raw
