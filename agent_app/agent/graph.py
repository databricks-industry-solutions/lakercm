"""
LangGraph agent for LakeRCM — medical document intelligence assistant.
Uses Claude Opus 4.6 via Databricks Foundation Model API.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
from langgraph.prebuilt import create_react_agent
from databricks.sdk import WorkspaceClient

from agent.hooks import post_model_hook, pre_model_hook
from agent.llm import build_chat_openai
from agent.state import AgentState
from agent.tools import get_all_tools
from config import settings
from services.checkpointer import get_async_checkpointer, get_checkpointer
from services.store import get_store

logger = logging.getLogger(__name__)

LLM_ENDPOINT = settings.llm_endpoint
# Base path the OpenAI-compatible client posts to (it appends
# `/chat/completions`). Prod points this at the Unity AI Gateway model service
# (`/ai-gateway/mlflow/v1`) so LLM_ENDPOINT is a UC model-service name that fans
# traffic across 5 models with input/output guardrails; default keeps the
# classic FM serving path for local dev / other callers.
LLM_BASE_PATH = settings.llm_base_path
CHAMPION_ALIAS = settings.champion_alias
CANDIDATE_ALIAS = settings.candidate_alias
CANDIDATE_TRAFFIC_PCT = max(0, min(100, settings.candidate_traffic_pct))
# Legacy alias kept for migration — old pods/registry entries used "production".
LEGACY_ALIAS = os.getenv("LAKERCM_PROMPT_ALIAS", "production")  # TODO: remove


@dataclass(frozen=True)
class PromptResolution:
    """Pinned prompt for a single agent turn — drives both rendering and trace tagging."""

    template: str
    name: str
    version: int
    alias_used: str  # "champion" | "candidate" | "explicit_version" | "inline_fallback"


def _prompt_name() -> str:
    """Fully-qualified prompt name. Settings are resolved at import (via the
    config.py model_validator), so this is equivalent to PROMPT_NAME but
    safer if ``settings.catalog`` / ``settings.schema_name`` ever shift at
    runtime."""
    return settings.prompt_name or (
        f"{settings.catalog}.{settings.schema_name}.agent_system_prompt"
    )


# Module-level alias for backward compatibility with eval modules that
# import PROMPT_NAME directly. Computed at import — relies on
# config.py's model_validator having populated settings.catalog from
# DATABRICKS_CATALOG env.
PROMPT_NAME = _prompt_name()


_RESOLVE_HOST_LOGGED = False


def _resolve_host() -> str:
    """Resolve Databricks host at call time. The Apps runtime auto-injects
    ``DATABRICKS_HOST`` into the process env; the SDK normalizes it to a
    fully-qualified URL via ``WorkspaceClient().config.host``. We try the
    SDK first because empirically (per app logs) it always returns a
    well-formed URL, then fall back to env / settings if the SDK can't
    construct a config (e.g. local dev without a profile)."""
    global _RESOLVE_HOST_LOGGED

    host = ""
    sdk_host = ""
    try:
        sdk_host = WorkspaceClient().config.host or ""
    except Exception as e:
        if not _RESOLVE_HOST_LOGGED:
            logger.warning("WorkspaceClient host resolution failed: %s", e)

    env_host = os.getenv("DATABRICKS_HOST", "")
    settings_host = settings.databricks_host or ""

    host = sdk_host or env_host or settings_host

    if not _RESOLVE_HOST_LOGGED:
        logger.info(
            "host_resolution sdk=%r env=%r settings=%r chosen=%r",
            sdk_host,
            env_host,
            settings_host,
            host,
        )
        _RESOLVE_HOST_LOGGED = True

    if not host:
        raise RuntimeError(
            "Could not resolve Databricks host: ensure DATABRICKS_HOST is "
            "set in the app environment or the WorkspaceClient is configured."
        )
    # Defensive: ensure scheme prefix.
    if not (host.startswith("http://") or host.startswith("https://")):
        host = f"https://{host}"
    return host.rstrip("/")


# Canonical system-prompt template lives in a file alongside this module so
# prompt iteration doesn't require Python edits or bundle redeploys. The CLI
# at scripts/register_agent_prompt.py pushes the file to MLflow Prompt
# Registry; running pods pick up new versions within
# settings.prompt_resolution_ttl_seconds via the refresh logic below.
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"

try:
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
except FileNotFoundError as e:
    # Fail fast at import — far easier to debug than a degraded runtime.
    raise RuntimeError(
        f"Required system-prompt file not found: {_SYSTEM_PROMPT_PATH}. "
        "It must ship in the bundle alongside graph.py."
    ) from e


@dataclass
class _CachedResolution:
    """A prompt resolution + the monotonic timestamp it was loaded at.

    Lives at module scope; refreshed lazily by _resolve_champion /
    _resolve_candidate on the next call after TTL expiry.
    """

    resolution: PromptResolution
    fetched_at: float


_champion_cache: _CachedResolution | None = None
_candidate_cache: _CachedResolution | None = None


def _load_alias(alias: str) -> PromptResolution | None:
    """Load a prompt by alias from the registry. Returns None if alias is unset/missing."""
    try:
        prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{alias}")
        return PromptResolution(
            template=prompt.template,
            name=PROMPT_NAME,
            version=int(prompt.version),
            alias_used=alias,
        )
    except Exception as e:
        logger.info("Prompt alias %s@%s not resolvable: %s", PROMPT_NAME, alias, e)
        return None


def _register_default_and_alias(alias: str) -> PromptResolution:
    """Register the inline default and point `alias` at the new version."""
    prompt = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        commit_message="initial registration from graph.py",
    )
    mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias=alias, version=prompt.version)
    logger.info(
        "Registered %s v%s and set alias %s", PROMPT_NAME, prompt.version, alias
    )
    return PromptResolution(
        template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        name=PROMPT_NAME,
        version=int(prompt.version),
        alias_used=alias,
    )


def _bootstrap_champion() -> PromptResolution:
    """Cold-start path when @champion doesn't exist in the registry yet.

    Order:
      1. Legacy alias (default "production") — copy its version to champion so
         a rolling deploy doesn't flap and the old pods keep serving while the
         new pods take over.
      2. Register the inline default (read from system_prompt.md) and seed
         both aliases.
      3. Last resort: inline default with version=0 (registry unreachable).
    """
    if LEGACY_ALIAS and LEGACY_ALIAS != CHAMPION_ALIAS:
        legacy = _load_alias(LEGACY_ALIAS)
        if legacy is not None:
            try:
                mlflow.genai.set_prompt_alias(
                    name=PROMPT_NAME, alias=CHAMPION_ALIAS, version=legacy.version
                )
                logger.info(
                    "Migrated alias: %s → %s (v%s)",
                    LEGACY_ALIAS,
                    CHAMPION_ALIAS,
                    legacy.version,
                )
                return PromptResolution(
                    template=legacy.template,
                    name=legacy.name,
                    version=legacy.version,
                    alias_used=CHAMPION_ALIAS,
                )
            except Exception as e:
                logger.warning("Could not set %s alias: %s", CHAMPION_ALIAS, e)

    try:
        return _register_default_and_alias(CHAMPION_ALIAS)
    except Exception as e:
        logger.warning("Prompt Registry unavailable, using inline default: %s", e)
        return PromptResolution(
            template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
            name=PROMPT_NAME,
            version=0,
            alias_used="inline_fallback",
        )


def _ttl_seconds() -> float:
    """Read TTL at call time so PROMPT_RESOLUTION_TTL_SECONDS env overrides
    apply without a process restart (useful for testing)."""
    return float(settings.prompt_resolution_ttl_seconds)


def _resolve_champion() -> PromptResolution:
    """Resolve the champion prompt with a TTL-refreshing cache.

    Fast path: cache is hot → return cached resolution (no MLflow call).
    Slow path: cache stale or empty → refresh from MLflow, fall back to:
      - serve stale (if registry returned None and we have a prior cache)
      - bootstrap chain (cold start, no cache, no registry hit)
    """
    global _champion_cache
    now = time.monotonic()
    cached = _champion_cache
    if cached is not None and (now - cached.fetched_at) < _ttl_seconds():
        return cached.resolution

    fresh = _load_alias(CHAMPION_ALIAS)
    if fresh is not None:
        _champion_cache = _CachedResolution(fresh, now)
        return fresh

    if cached is not None:
        # Registry temporarily unreachable or @champion missing — serve stale
        # (fail-open: better than breaking chat). Re-stamp fetched_at so we
        # don't hammer MLflow on every subsequent request.
        logger.warning(
            "Champion refresh missed registry; serving stale v%s",
            cached.resolution.version,
        )
        _champion_cache = _CachedResolution(cached.resolution, now)
        return cached.resolution

    # Cold-start path — no cache and registry empty. Run the bootstrap chain.
    bootstrap = _bootstrap_champion()
    _champion_cache = _CachedResolution(bootstrap, now)
    return bootstrap


def _resolve_candidate() -> PromptResolution | None:
    """Resolve the candidate prompt with a TTL-refreshing cache.

    Inverted failure semantics vs champion: when refresh returns None, drop
    the cache (fail-closed). This lets an operator stop an A/B test by
    deleting the alias and have it propagate within the TTL window.
    Champion-fallback handles the dropped-candidate case.
    """
    global _candidate_cache
    now = time.monotonic()
    cached = _candidate_cache
    if cached is not None and (now - cached.fetched_at) < _ttl_seconds():
        return cached.resolution

    fresh = _load_alias(CANDIDATE_ALIAS)
    if fresh is not None:
        _candidate_cache = _CachedResolution(fresh, now)
        return fresh

    # No candidate now — either intentionally unset, never existed, or MLflow
    # flapped. Drop the cache so the candidate actually goes away.
    _candidate_cache = None
    return None


def _bucket(user_email: str | None, thread_id: str | None) -> int:
    """Deterministic 0-99 bucket from (user_email, thread_id) for traffic split."""
    key = f"{user_email or ''}|{thread_id or ''}".encode()
    return int.from_bytes(hashlib.sha1(key).digest()[:4], "big") % 100


def resolve_prompt(
    user_context: dict | None = None,
    thread_id: str | None = None,
) -> PromptResolution:
    """Pick champion or candidate for this request. Per-turn — caches the loaded prompts but routes per call."""
    champion = _resolve_champion()
    candidate = _resolve_candidate() if CANDIDATE_TRAFFIC_PCT > 0 else None
    if candidate is None:
        return champion
    user_email = (user_context or {}).get("user_email")
    if _bucket(user_email, thread_id) < CANDIDATE_TRAFFIC_PCT:
        return candidate
    return champion


def _personalize(template: str, user_context: dict | None) -> str:
    from config import settings
    import re

    first_name = user_context.get("first_name", "there") if user_context else "there"
    role = user_context.get("role", "Reviewer") if user_context else "Reviewer"
    result = template.replace("{{first_name}}", first_name).replace("{{role}}", role)

    # Strip KG content blocks when the feature is disabled
    if not getattr(settings, "kg_enabled", False):
        result = re.sub(
            r"<!-- kg:start -->.*?<!-- kg:end -->\n?", "", result, flags=re.DOTALL
        )

    return result


def build_system_prompt(user_context: dict | None = None) -> str:
    """Backward-compatible helper — resolves champion only and personalizes."""
    return _personalize(_resolve_champion().template, user_context)


def register_default_prompt() -> int:
    """Force-register the bundled system_prompt.md as a new version under the
    champion alias. Mostly a bootstrap helper today — ongoing prompt iteration
    should use scripts/register_agent_prompt.py instead, which decouples
    prompt edits from code redeploys. Returns the new version id."""
    prompt = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        commit_message="refresh from graph.py",
    )
    mlflow.genai.set_prompt_alias(
        name=PROMPT_NAME, alias=CHAMPION_ALIAS, version=prompt.version
    )
    global _champion_cache
    _champion_cache = _CachedResolution(
        PromptResolution(
            template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
            name=PROMPT_NAME,
            version=int(prompt.version),
            alias_used=CHAMPION_ALIAS,
        ),
        time.monotonic(),
    )
    logger.info(
        "Registered %s v%s as alias %s",
        PROMPT_NAME,
        prompt.version,
        CHAMPION_ALIAS,
    )
    return int(prompt.version)


def load_prompt_version(version: int) -> PromptResolution:
    """Load a specific prompt version (for offline evaluation)."""
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{version}")
    return PromptResolution(
        template=prompt.template,
        name=PROMPT_NAME,
        version=int(version),
        alias_used="explicit_version",
    )


def create_agent(
    user_context: dict | None = None,
    thread_id: str | None = None,
    prompt_resolution: PromptResolution | None = None,
    use_v2_prebuilt: bool = False,
    tools: list | None = None,
    stateless: bool = False,
    use_responses_api: bool | None = None,
    reasoning_effort: str | None = None,
    llm_endpoint: str | None = None,
) -> tuple[object, PromptResolution]:
    """Create the personalized LangGraph ReAct agent.

    Returns (agent, resolution) so callers can tag the trace with which
    prompt version actually rendered. If `prompt_resolution` is provided
    (offline eval path), it bypasses champion/candidate routing.

    `use_v2_prebuilt` switches `create_react_agent` to its v2 internal
    graph — required for `post_model_hook` to actually run. The legacy
    /responses path leaves this off (False) to preserve bit-for-bit
    behavior; the AG-UI path (`get_agui_graph`) sets it True so the
    trace_id post-hook fires and surfaces via STATE_DELTA.
    """
    # LLM client comes from the shared factory (agent/llm.py) so its auth token
    # refreshes per request — critical for the AG-UI graph, which is compiled
    # once at boot: a baked api_key would expire and every async model call
    # would 401, silently aborting the SSE stream.
    # Reasoning/API routing: explicit args (the eval path passes these) win,
    # else fall back to the deployed settings. reasoning_effort is only sent on
    # the /responses path — on /chat/completions the gateway forwards it
    # unchanged and Claude 400s ("reasoning_effort: Extra inputs are not
    # permitted"). Eval keeps the baseline (use_responses_api=False, no effort)
    # for run-to-run comparability.
    _use_responses = (
        settings.agent_llm_use_responses_api
        if use_responses_api is None
        else use_responses_api
    )
    _effort = (
        settings.agent_llm_effort if reasoning_effort is None else reasoning_effort
    )
    _llm_kwargs: dict = dict(
        # Databricks-hosted Claude Opus does not accept `temperature` — the
        # FMAPI returns 400 BAD_REQUEST. Use the model's built-in default.
        max_tokens=2000,
        # Bound every model call: a slow/stuck endpoint must fail fast, not hang
        # the request (or a whole GEPA eval run, which makes dozens of calls).
        timeout=90,
        max_retries=2,
        # Required so per-token callbacks (on_llm_new_token) fire — used by
        # main.py's _StreamHandler to deliver real token streaming over SSE.
        streaming=True,
    )
    # Complexity-tiered routing (agent/routing.py) passes a per-tier model
    # service here; None keeps build_chat_openai's default (settings.llm_endpoint)
    # so the non-routed path is byte-for-byte unchanged.
    if llm_endpoint:
        _llm_kwargs["model"] = llm_endpoint
    if _use_responses:
        # /responses lets the gateway normalize ONE reasoning.effort across all
        # destinations AND lets reasoning-model destinations tool-call while
        # reasoning (which the /chat/completions path cannot do uniformly).
        # NOTE: do NOT set output_version="responses/v1" here — it changes the
        # streamed on_llm_new_token deltas from text to structured content
        # blocks (dicts), which breaks the SSE 'token' text stream the frontend
        # renders. The langchain default streams text correctly.
        _llm_kwargs["use_responses_api"] = True
        _llm_kwargs["reasoning_effort"] = _effort
    llm = build_chat_openai(**_llm_kwargs)

    resolution = prompt_resolution or resolve_prompt(user_context, thread_id)
    system_prompt = _personalize(resolution.template, user_context)
    # `tools` override is the seam for trace-replay eval (recorded tool outputs
    # instead of live Lakebase). Default None → production behavior unchanged.
    tools = tools if tools is not None else get_all_tools()

    # `stateless` (offline eval / trace-replay) skips the Postgres store +
    # checkpointer entirely — eval runs with thread_id=None and never persists,
    # so there's no reason to open a Lakebase pool (which would otherwise time
    # out per call when no PG creds are present).
    if stateless:
        store = None
        checkpointer = None
    else:
        try:
            store = get_store()
        except Exception as e:
            logger.warning(
                "Long-term memory store unavailable, continuing without it: %s", e
            )
            store = None

        # Pick the checkpointer flavor that matches the call-style of the graph
        # invoker: AG-UI awaits aget_state/aget_tuple, so we hand it an
        # AsyncPostgresSaver. The legacy /responses path runs agent.invoke()
        # synchronously (in a worker thread) and needs the sync PostgresSaver.
        # Both share the same `public.checkpoint_*` tables under the group role.
        try:
            if use_v2_prebuilt:
                checkpointer = get_async_checkpointer()
            else:
                checkpointer = get_checkpointer()
        except Exception as e:
            logger.warning("Checkpointer unavailable, agent will run stateless: %s", e)
            checkpointer = None

    create_kwargs = dict(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        state_schema=AgentState,
        pre_model_hook=pre_model_hook,
        checkpointer=checkpointer,
        store=store,
    )
    if use_v2_prebuilt:
        # post_model_hook only runs on the v2 prebuilt graph; gated to the
        # AG-UI path so the legacy /responses behavior is unchanged.
        create_kwargs["post_model_hook"] = post_model_hook
        create_kwargs["version"] = "v2"
    agent = create_react_agent(**create_kwargs)  # TODO: fix

    return agent, resolution


# --- AG-UI module-level graph -------------------------------------------------
#
# `add_langgraph_fastapi_endpoint` needs a single long-lived compiled graph.
# The legacy /responses path rebuilds the agent per turn (so it can inject a
# personalized system prompt). For the AG-UI pilot, build once at startup
# with default personalization placeholders ("there"/"Reviewer") — the
# trade-off is documented in the plan and revisited when we migrate fully.
_agui_graph_cache: dict = {}


def get_agui_graph(
    llm_endpoint: str | None = None,
    reasoning_effort: str | None = None,
    cache_key: str = "default",
) -> object:
    """Compiled graph used by the AG-UI endpoint. Built lazily, cached per
    ``cache_key`` so complexity-tiered routing can hold one graph per tier
    (endpoint + effort). Called with no args → the single default graph,
    behavior unchanged."""
    global _agui_graph_cache
    if cache_key in _agui_graph_cache:
        return _agui_graph_cache[cache_key]
    default_user_context = {
        "user_email": "agui@lakercm",
        "display_name": "Reviewer",
        "first_name": "there",
        "role": "Reviewer",
    }
    # Per-request personalization on /chat-v2 is handled by useCopilotReadable
    # on the frontend pushing reviewer identity into the agent's context; the
    # system-prompt substitution stays generic. use_v2_prebuilt=True so the
    # post_model_hook fires and feeds last_trace_id into STATE_DELTA.
    agent, _ = create_agent(
        user_context=default_user_context,
        thread_id=None,
        use_v2_prebuilt=True,
        llm_endpoint=llm_endpoint,
        reasoning_effort=reasoning_effort,
    )
    # AG-UI's LangGraphAgent calls graph.aget_state(...) on every run; if the
    # graph was compiled without a checkpointer (because Lakebase's
    # PostgresSaver.setup() failed on a fresh deploy) the call raises
    # "No checkpointer set" and the SSE stream silently aborts mid-flight.
    # Fall back to an in-process MemorySaver so the AG-UI path is at least
    # usable within a single conversation while we sort out the Lakebase
    # grants. State is lost on pod restart — acceptable for the pilot.
    needs_memory_fallback = False
    try:
        from langgraph.pregel.types import CheckpointerProtocol  # type: ignore  # noqa: F401
    except Exception:
        pass
    try:
        # Some graph builders expose `.checkpointer` (None when missing).
        if getattr(agent, "checkpointer", None) is None:
            needs_memory_fallback = True
    except Exception:
        needs_memory_fallback = True

    if needs_memory_fallback:
        try:
            from langgraph.checkpoint.memory import MemorySaver

            logger.warning(
                "AG-UI graph rebuilt with in-memory checkpointer "
                "(Lakebase checkpointer unavailable at startup)."
            )
            _fallback_kwargs: dict = dict(
                max_tokens=2000,
                timeout=90,
                max_retries=2,
                streaming=True,
            )
            _fb_effort = (
                reasoning_effort
                if reasoning_effort is not None
                else settings.agent_llm_effort
            )
            if settings.agent_llm_use_responses_api:
                _fallback_kwargs["use_responses_api"] = True
                _fallback_kwargs["reasoning_effort"] = _fb_effort
            if llm_endpoint:
                _fallback_kwargs["model"] = llm_endpoint
            llm = build_chat_openai(**_fallback_kwargs)
            resolution = resolve_prompt(default_user_context, None)
            system_prompt = _personalize(resolution.template, default_user_context)
            try:
                store = get_store()
            except Exception:
                store = None
            agent = create_react_agent(
                model=llm,
                tools=get_all_tools(),
                prompt=system_prompt,
                state_schema=AgentState,
                pre_model_hook=pre_model_hook,
                post_model_hook=post_model_hook,
                version="v2",
                checkpointer=MemorySaver(),
                store=store,
            )
        except Exception as e:
            logger.exception(
                "AG-UI MemorySaver fallback failed; AG-UI endpoint may 500: %s", e
            )
    _agui_graph_cache[cache_key] = agent
    return agent
