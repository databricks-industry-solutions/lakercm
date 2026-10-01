"""Minimal Statement Execution helper for warehouse reads outside analytics.

Exists so new warehouse readers cannot repeat the failure that made every
analytics endpoint report zeros: ``execute_statement`` left to the warehouse
default returns inline JSON on a PRO warehouse but an ARROW_STREAM
``result.attachment`` on a Lakehouse//RT one, where ``result.data_array`` is
None. Code that reads that as "no rows" turns a populated gold layer into an
all-zeros dashboard, silently.

So the shape is pinned here, and a result that reports rows without an inline
array raises instead of degrading to an empty list.

``routes/analytics.py`` carries its own copy of this logic; the two should be
unified once the fix in flight there has landed, rather than edited in parallel
now.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    Disposition,
    Format,
    StatementParameterListItem,
    StatementState,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 1.0
_POLL_DEADLINE_S = 55.0


class WarehouseReadError(RuntimeError):
    """A warehouse read failed, timed out, or came back in a shape we reject."""


def warehouse_rows(
    workspace_client: WorkspaceClient,
    warehouse_id: str,
    query: str,
    params: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Run one read-only statement and return rows as dicts.

    User-supplied values MUST be bound via ``params`` (named ``:markers``),
    never interpolated into the statement.
    """
    parameters = None
    if params:
        parameters = [
            StatementParameterListItem(name=k, value=v, type="STRING")
            for k, v in params.items()
        ]

    response = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
        parameters=parameters,
        wait_timeout="30s",
        # See the module docstring: not the warehouse default.
        format=Format.JSON_ARRAY,
        disposition=Disposition.INLINE,
    )

    deadline = time.monotonic() + _POLL_DEADLINE_S
    while response.status and response.status.state in (
        StatementState.PENDING,
        StatementState.RUNNING,
    ):
        if time.monotonic() > deadline:
            raise WarehouseReadError("Warehouse query timed out")
        time.sleep(_POLL_INTERVAL_S)
        response = workspace_client.statement_execution.get_statement(
            response.statement_id
        )

    if not response.status or response.status.state != StatementState.SUCCEEDED:
        detail = (
            response.status.error.message
            if response.status and response.status.error
            else f"state={response.status.state if response.status else 'unknown'}"
        )
        raise WarehouseReadError(f"Warehouse query failed: {detail}")

    if not response.result:
        return []

    if response.result.data_array is None:
        row_count = response.result.row_count or 0
        if row_count:
            raise WarehouseReadError(
                f"Warehouse returned {row_count} row(s) with no inline "
                "data_array — unexpected result disposition"
            )
        return []

    columns = [col.name for col in response.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in response.result.data_array]
