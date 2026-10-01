"""
LakeRCM Lakebase Endpoint Monitor

Background asyncio task that polls the Lakebase Autoscaling endpoint state
every 30s and writes transitions to public.lakebase_events. Powers the
admin diagnostics page (cold-start history, spin-up/spin-down).

Endpoint state values (databricks.sdk EndpointStatusState):
    INIT    — endpoint is starting / initializing
    ACTIVE  — endpoint is serving traffic
    IDLE    — endpoint suspended / scaled to zero

A cold start is the IDLE -> INIT -> ACTIVE sequence; cold_start_seconds is
the wall-clock delta between the first INIT-or-later observation after IDLE
and the first ACTIVE observation that follows.
"""

import asyncio
import json
import logging
import time
from typing import Optional

from databricks.sdk import WorkspaceClient

import dependencies

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30


class LakebaseMonitor:
    def __init__(
        self,
        workspace_client: WorkspaceClient,
        endpoint_name: str,
        poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
    ):
        self.workspace_client = workspace_client
        self.endpoint_name = endpoint_name
        self.poll_interval_seconds = poll_interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

        self._last_state: Optional[str] = None
        # Wall-clock (monotonic) when we first OBSERVED the endpoint as IDLE.
        # Used to start the cold-start clock on any IDLE -> non-IDLE transition.
        self._idle_at_mono: Optional[float] = None
        # Set on IDLE -> non-IDLE; cleared once ACTIVE is observed.
        self._cold_start_began_at: Optional[float] = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="lakebase_monitor")
        logger.info(
            "lakebase_monitor started (endpoint=%s, interval=%ds)",
            self.endpoint_name,
            self.poll_interval_seconds,
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        # Seed _last_state from the latest event so a restart doesn't
        # double-record the current steady state.
        self._last_state = self._read_last_state_from_db()
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception as e:
                logger.warning("lakebase_monitor tick failed: %s", e)
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.poll_interval_seconds
                )
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        # SDK calls are blocking — run in the default thread pool so we don't
        # stall the event loop.
        endpoint = await asyncio.to_thread(
            self.workspace_client.postgres.get_endpoint, self.endpoint_name
        )
        state = self._extract_state(endpoint)
        if state is None:
            return

        now_mono = time.monotonic()

        # Mark the moment we first observed IDLE so we can measure how long
        # the wake-up took on the next non-IDLE poll. If we just started up
        # already-IDLE, the next IDLE→non-IDLE transition uses this anchor.
        if state == "IDLE" and self._last_state != "IDLE":
            self._idle_at_mono = now_mono

        if state == self._last_state:
            return

        cold_start_seconds = None
        # Wake-up start: ANY transition out of IDLE counts. Lakebase
        # autoscaling often skips INIT and goes directly IDLE -> ACTIVE,
        # so the original "IDLE -> INIT" trigger never fired in practice.
        if self._last_state == "IDLE" and state != "IDLE":
            self._cold_start_began_at = self._idle_at_mono or now_mono
        # Wake-up complete: clamp to the moment we first see ACTIVE.
        if state == "ACTIVE" and self._cold_start_began_at is not None:
            cold_start_seconds = max(0.0, now_mono - self._cold_start_began_at)
            self._cold_start_began_at = None

        await asyncio.to_thread(
            self._record_transition,
            self._last_state,
            state,
            cold_start_seconds,
            endpoint,
        )
        self._last_state = state

    @staticmethod
    def _extract_state(endpoint) -> Optional[str]:
        status = getattr(endpoint, "status", None)
        if status is None:
            return None
        current = getattr(status, "current_state", None)
        if current is None:
            return None
        # current_state is an enum; .value is the string ("ACTIVE" etc.)
        return getattr(current, "value", str(current))

    def _read_last_state_from_db(self) -> Optional[str]:
        db = dependencies.lakercm_db
        if db is None or db.pool is None:
            return None
        try:
            rows = db._execute_query(
                "SELECT new_state FROM public.lakebase_events "
                "WHERE endpoint_name = %s ORDER BY ts DESC LIMIT 1",
                (self.endpoint_name,),
            )
            if rows:
                return rows[0]["new_state"]
        except Exception as e:
            logger.debug("could not seed last_state from db: %s", e)
        return None

    def _record_transition(
        self,
        prev_state: Optional[str],
        new_state: str,
        cold_start_seconds: Optional[float],
        endpoint,
    ) -> None:
        db = dependencies.lakercm_db
        if db is None or db.pool is None:
            logger.debug("no db pool yet; skipping event record")
            return
        spec = getattr(endpoint, "spec", None)
        status = getattr(endpoint, "status", None)
        meta = {
            "endpoint_type": getattr(
                getattr(spec, "endpoint_type", None), "value", None
            ),
            "min_cu": getattr(status, "autoscaling_limit_min_cu", None),
            "max_cu": getattr(status, "autoscaling_limit_max_cu", None),
            "disabled": getattr(spec, "disabled", None),
        }
        try:
            db._execute_query(
                "INSERT INTO public.lakebase_events "
                "  (endpoint_name, prev_state, new_state, "
                "   cold_start_seconds, metadata) "
                "VALUES (%s, %s, %s, %s, %s::jsonb)",
                (
                    self.endpoint_name,
                    prev_state,
                    new_state,
                    cold_start_seconds,
                    json.dumps(meta),
                ),
                fetch=False,
            )
            logger.info(
                "lakebase transition: %s -> %s (cold_start=%s)",
                prev_state,
                new_state,
                f"{cold_start_seconds:.2f}s" if cold_start_seconds else "n/a",
            )
        except Exception as e:
            logger.warning("failed to record lakebase event: %s", e)
