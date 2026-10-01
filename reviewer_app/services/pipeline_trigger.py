"""Fire-and-forget pipeline kicks shared across reviewer routes.

A leaf module (imports only config + the SDK type) so both routes/documents and
routes/proposals can trigger a pipeline without importing each other — a
route->route import would drag one router's whole dependency set into the
other's unit tests.
"""

from __future__ import annotations

import logging

from databricks.sdk import WorkspaceClient

from config import settings

logger = logging.getLogger(__name__)


def trigger_analytics_pipeline(workspace_client: WorkspaceClient) -> None:
    """Kick the analytics pipeline after a human review action.

    The analytics pipeline (fact_review / fact_agent_proposal) reads the
    Lakebase CDC mirror of reviews + proposals, which the documents
    file-arrival trigger cannot represent. So the reviewer app kicks it on
    review-submit / proposal-disposition, the same idempotent way it kicks the
    documents pipeline on upload. Together with the analytics task chained off
    lakercm-documents-refresh (the document signal), analytics stays fresh with
    no cron. An "update already active" 409 is the expected idempotent path.
    """
    pipeline_id = settings.analytics_pipeline_id
    if not pipeline_id:
        logger.warning("ANALYTICS_PIPELINE_ID not configured; skipping trigger")
        return
    try:
        workspace_client.pipelines.start_update(
            pipeline_id=pipeline_id, full_refresh=False
        )
        logger.info("Triggered analytics pipeline update: %s", pipeline_id)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "already" in msg or "conflict" in msg or "409" in msg:
            logger.debug("Analytics pipeline %s already updating", pipeline_id)
        else:
            logger.warning("analytics start_update(%s) failed: %s", pipeline_id, e)
