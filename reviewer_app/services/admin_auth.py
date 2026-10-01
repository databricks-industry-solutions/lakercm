"""
LakeRCM Admin Auth

Server-side check for workspace `admins` group membership.

Identity comes from the X-Forwarded-Email header Databricks Apps injects.
Group lookup uses the user's own OAuth token (X-Forwarded-Access-Token,
enabled by `user_access_token_authorization` in app.yaml) and calls SCIM
`/Me` — the only endpoint that reliably returns the caller's `groups` list
without requiring the app SP to be a workspace admin.

Falls back to the app SP's `users.list(filter=...)` when no forwarded token
is present (e.g. local dev) — this returns False for most users since the SP
can't see other users' groups, which is the safe default.

Never trust a client-supplied admin claim — every protected endpoint must
depend on `require_admin`, which re-runs the lookup.
"""

import logging
from typing import Optional

from cachetools import TTLCache
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from fastapi import HTTPException, Request

import dependencies
from config import settings

logger = logging.getLogger(__name__)

ADMIN_GROUP_NAME = "admins"

# email -> bool, 5 minute TTL. Removes from `admins` propagate within 5 min.
_admin_cache: TTLCache = TTLCache(maxsize=512, ttl=300)


def _check_via_obo(access_token: str) -> Optional[bool]:
    """Call SCIM /Me as the user; return True if `admins` is in their groups.

    Returns None on error so the caller can fall back to the SP path.
    """
    try:
        host = settings.databricks_host or _host_from_workspace_client()
        if not host:
            logger.debug("OBO admin check: no DATABRICKS_HOST resolved")
            return None
        cfg = Config(host=host, token=access_token, auth_type="pat")
        client = WorkspaceClient(config=cfg)
        me = client.current_user.me()
        groups = me.groups or []
        return any((g.display or "").lower() == ADMIN_GROUP_NAME for g in groups)
    except Exception as e:
        logger.warning("OBO admin check failed: %s", e)
        return None


def _check_via_sp(email: str) -> bool:
    """Fallback: app SP looks up the user's groups via SCIM filter.

    This returns False for most users in production because the SP can't
    typically see other users' group memberships. Kept for local dev paths
    where there is no forwarded token.
    """
    workspace_client = dependencies.workspace_client
    if workspace_client is None:
        return False
    try:
        safe = email.replace('"', '\\"')
        users = list(
            workspace_client.users.list(
                filter=f'userName eq "{safe}"',
                count=1,
                attributes="groups",
            )
        )
        if not users:
            return False
        groups = users[0].groups or []
        return any((g.display or "").lower() == ADMIN_GROUP_NAME for g in groups)
    except Exception as e:
        logger.warning("SP admin check failed for %s: %s", email, e)
        return False


def _host_from_workspace_client() -> Optional[str]:
    wc = dependencies.workspace_client
    if wc is None:
        return None
    try:
        return wc.config.host
    except Exception:
        return None


def is_admin_for_request(request: Request) -> bool:
    """Server-side admin check for a single request.

    Prefers OBO (X-Forwarded-Access-Token) so we get the user's own groups;
    falls back to the SP-based lookup. Caches by email for 5 minutes.
    """
    email = dependencies.get_current_user_email(request)
    if not email:
        return False

    cached = _admin_cache.get(email)
    if cached is not None:
        return cached

    access_token = request.headers.get("x-forwarded-access-token")
    result: Optional[bool] = None
    if access_token:
        result = _check_via_obo(access_token)
    if result is None:
        result = _check_via_sp(email)

    _admin_cache[email] = bool(result)
    return bool(result)


def require_admin(request: Request) -> str:
    """FastAPI dependency: 403 unless the caller is in the workspace `admins`
    group. Returns the verified admin email so handlers can log who acted.
    """
    email = dependencies.get_current_user_email(request)
    if not is_admin_for_request(request):
        raise HTTPException(status_code=403, detail="Admin access required")
    return email


def invalidate_admin_cache(email: Optional[str] = None) -> None:
    if email is None:
        _admin_cache.clear()
    else:
        _admin_cache.pop(email, None)
