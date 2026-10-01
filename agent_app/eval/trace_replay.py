"""
Trace-replay for offline eval — serve RECORDED tool outputs instead of live Lakebase.

GEPA/promote eval re-runs the agent to test candidate prompts. The agent's tools
read Lakebase, which couples eval to a live DB + an identity the eval job doesn't
have. But the tool outputs the agent already produced are recorded in the OTel
traces (`<catalog>.<schema>.agent_traces_otel_spans`): each TOOL span carries
`mlflow.spanInputs` (the call args) and `mlflow.spanOutputs` (the returned
ToolMessage, whose `.content` is the tool's JSON string).

So we replay: build a `{(tool_name, args) -> output}` map from recent traces, and
give the eval agent tools with the SAME names/schemas whose bodies return those
recorded outputs. The prompt still drives tool SELECTION, reasoning over the
returned data, formatting, and refusals — which is what prompt optimization tests
— but nothing touches Postgres. Deterministic and reproducible.

Limitation: a candidate prompt that calls a tool with args no recorded trace used
gets a fallback (any recorded output for that tool, else a deterministic stub).
Acceptable for prompt eval; we are not testing novel DB queries.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

TOOL_SPAN_TYPE = "TOOL"
_MISS_OUTPUT = json.dumps(
    {"error": "no recorded data for this tool call (trace-replay miss)"}
)
# Cap recorded tool outputs fed back into the agent. Unbounded payloads (e.g.
# 20-row document lists) overflow the agent's context (trim_messages prunes to
# empty → empty responses → 0 scores) AND GEPA's reflection-model context. A few
# KB preserves realistic structure while staying bounded — and mirrors the
# good-practice that tools shouldn't dump huge blobs into an LLM context.
MAX_REPLAY_OUTPUT_CHARS = 4000


def _cap(text: str) -> str:
    if text is None or len(text) <= MAX_REPLAY_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_REPLAY_OUTPUT_CHARS]
        + f"\n…[truncated {len(text) - MAX_REPLAY_OUTPUT_CHARS} chars for eval]"
    )


__all__ = [
    "extract_tool_io",
    "build_replay_map",
    "build_replay_tools",
    "replay_tools_from_fixtures",
    "records_have_fixtures",
    "load_recent_traces",
]


def records_have_fixtures(records: list) -> bool:
    """True iff EVERY record carries `inputs._tool_fixtures`.

    When true, per-record replay (`replay_tools_from_fixtures`) serves every tool
    call, so the GLOBAL trace-replay map is unnecessary — and skipping it avoids
    `load_recent_traces()`/`mlflow.search_traces`, which cold-starts a STOPPED
    monitoring SQL warehouse (waits up to 1200s) and scans the full OTel corpus.
    Robust to `inputs` arriving as a dict OR a JSON string (the UC dataset
    backing column is `inputs::string`, though `Dataset.to_df()` usually
    deserializes it).
    """
    if not records:
        return False
    for r in records:
        inp = r.get("inputs") if isinstance(r, dict) else getattr(r, "inputs", None)
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except Exception:  # noqa: BLE001
                return False
        # Key PRESENCE (not truthiness): a refusal record carries an empty
        # `_tool_fixtures: {}` and is still self-contained — per-record replay
        # serves it stub tools (no real Lakebase tools), which is exactly what a
        # no-tool/refusal case wants.
        if not (isinstance(inp, dict) and "_tool_fixtures" in inp):
            return False
    return True


def _norm_args(args: Any) -> str:
    """Stable string key for a tool call's args."""
    try:
        return json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return str(args)


def _span_attr(span, key: str) -> Any:
    attrs = getattr(span, "attributes", None) or {}
    return attrs.get(key)


def _content_from_outputs(raw_out: Any) -> str | None:
    """Pull the tool's returned string from a TOOL span's `mlflow.spanOutputs`.

    spanOutputs is the LangChain ToolMessage dict
    ({content, name, tool_call_id, status, type}); `.content` is the tool's
    JSON string. Falls back to the whole object if there's no `content`.
    """
    if raw_out is None:
        return None
    obj = raw_out
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:  # noqa: BLE001
            return obj  # already a plain string output
    if isinstance(obj, dict):
        content = obj.get("content")
        if content is not None:
            return (
                content
                if isinstance(content, str)
                else json.dumps(content, default=str)
            )
        return json.dumps(obj, default=str)
    return str(obj)


def extract_tool_io(trace) -> dict[tuple[str, str], str]:
    """Build `{(tool_name, normalized_args): output_content}` from one trace's TOOL spans."""
    out: dict[tuple[str, str], str] = {}
    spans = getattr(getattr(trace, "data", None), "spans", None) or []
    for span in spans:
        stype = _span_attr(span, "mlflow.spanType") or getattr(span, "span_type", "")
        if str(stype).upper() != TOOL_SPAN_TYPE:
            continue
        name = getattr(span, "name", None)
        if not name:
            continue
        content = _content_from_outputs(_span_attr(span, "mlflow.spanOutputs"))
        if content is None:
            continue
        out[(name, _norm_args(_span_attr(span, "mlflow.spanInputs")))] = _cap(content)
    return out


def build_replay_map(traces) -> dict[tuple[str, str], str]:
    """Merge `extract_tool_io` across many traces into one global replay index."""
    replay: dict[tuple[str, str], str] = {}
    for trace in traces:
        try:
            replay.update(extract_tool_io(trace))
        except Exception as e:  # noqa: BLE001 — one bad trace shouldn't kill the build
            logger.warning("skipping a trace during replay-map build: %s", e)
    logger.info(
        "Built replay map: %d (tool,args) entries from %d traces",
        len(replay),
        len(traces),
    )
    return replay


def build_replay_tools(replay_map: dict, base_tools: list | None = None) -> list:
    """Tools mirroring the real ones (name/description/args_schema) that return
    recorded outputs. On an args miss → any recorded output for that tool name;
    else a deterministic stub."""
    from langchain_core.tools import StructuredTool

    if base_tools is None:
        from agent.tools import get_all_tools

        base_tools = get_all_tools()

    by_name: dict[str, str] = {}
    for (name, _args), output in replay_map.items():
        by_name.setdefault(name, output)  # first recorded output per tool = fallback

    def _make_fn(tool_name: str):
        def _fn(**kwargs) -> str:
            key = (tool_name, _norm_args(kwargs))
            if key in replay_map:
                return replay_map[key]
            return by_name.get(tool_name, _MISS_OUTPUT)

        return _fn

    replay_tools = []
    for t in base_tools:
        name = getattr(t, "name", None)
        if not name:
            continue
        replay_tools.append(
            StructuredTool.from_function(
                func=_make_fn(name),
                name=name,
                description=getattr(t, "description", "") or name,
                args_schema=getattr(t, "args_schema", None),
            )
        )
    return replay_tools


def replay_tools_from_fixtures(fixtures: dict, base_tools: list | None = None) -> list:
    """Per-record replay tools for the fixture-based golden eval (agent_eval_v2).

    Unlike the trace-derived global map (keyed by exact (tool, args)), a record's
    `_tool_fixtures` says "whenever the agent calls <tool>, return THIS output" —
    args-agnostic, because the authored fixture IS the ground truth the record's
    `expected_facts` were derived from. A tool the agent calls that has no fixture
    in this record returns a deterministic empty stub (so an off-target tool call
    still completes, and ToolCallCorrectness penalises the wrong selection).
    """
    from langchain_core.tools import StructuredTool

    if base_tools is None:
        from agent.tools import get_all_tools

        base_tools = get_all_tools()

    served: dict[str, str] = {}
    for name, output in (fixtures or {}).items():
        served[name] = _cap(
            output if isinstance(output, str) else json.dumps(output, default=str)
        )

    def _make_fn(tool_name: str):
        def _fn(**kwargs) -> str:
            return served.get(
                tool_name,
                json.dumps(
                    {"note": "tool not exercised in this eval scenario", "results": []}
                ),
            )

        return _fn

    tools = []
    for t in base_tools:
        name = getattr(t, "name", None)
        if not name:
            continue
        tools.append(
            StructuredTool.from_function(
                func=_make_fn(name),
                name=name,
                description=getattr(t, "description", "") or name,
                args_schema=getattr(t, "args_schema", None),
            )
        )
    return tools


def load_recent_traces(days: int = 30, max_traces: int = 2000) -> list:
    """Fetch recent OK traces in-workspace, reusing the curation recipe
    (sets MLFLOW_TRACING_SQL_WAREHOUSE_ID from the experiment tag)."""
    import time

    import mlflow

    from eval.dataset import _ensure_sql_warehouse_for_traces, _experiment_id

    _ensure_sql_warehouse_for_traces()
    cutoff_ms = int((time.time() - days * 86_400) * 1000)
    filter_string = f"trace.status = 'OK' AND trace.timestamp_ms >= {cutoff_ms}"
    exp = _experiment_id()
    # Prefer `locations=` (`experiment_ids` is deprecated); fall back for older
    # mlflow. Each form also tolerates `return_type` being unsupported.
    for loc_kwargs in ({"locations": [exp]}, {"experiment_ids": [exp]}):
        try:
            return mlflow.search_traces(
                filter_string=filter_string,
                max_results=max_traces,
                return_type="list",
                **loc_kwargs,
            )
        except TypeError:
            try:
                return mlflow.search_traces(
                    filter_string=filter_string,
                    max_results=max_traces,
                    **loc_kwargs,
                )
            except TypeError:
                continue
    return mlflow.search_traces(filter_string=filter_string, max_results=max_traces)
