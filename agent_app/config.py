"""
LakeRCM Agent App — Configuration.

Pydantic Settings for Lakebase endpoint and LLM endpoint. Point lookups read
Lakebase Postgres (public.* reviewer tables + the lakercm.* synced gold
tables). The analytics tools read the Reyden (Lakehouse//RT) warehouse through
the Unity AI Gateway MCP service system.ai.dbsql (dbsql_mcp_warehouse_id).
"""

import logging
import os

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator

logger = logging.getLogger(__name__)

# Per-tier model-service endpoints, which read "none" as unset.
_TIER_ENDPOINT_FIELDS = frozenset(
    {"llm_endpoint_low", "llm_endpoint_med", "llm_endpoint_high"}
)


class Settings(BaseSettings):
    """Agent app configuration loaded from environment variables."""

    databricks_host: str = Field(default="", env="DATABRICKS_HOST")
    catalog: str = Field(default="main", env="DATABRICKS_CATALOG")
    schema_name: str = Field(default="lakercm", env="LAKERCM_SCHEMA")

    endpoint_name: str = Field(
        default="projects/lakercm/branches/prod/endpoints/primary",
        env="ENDPOINT_NAME",
    )
    pg_database: str = Field(default="databricks_postgres", env="PGDATABASE")
    pg_host: str = Field(default="", env="PGHOST")
    pg_port: int = Field(default=5432, env="PGPORT")
    pg_user: str = Field(default="", env="PGUSER")
    pg_sslmode: str = Field(default="require", env="PGSSLMODE")
    pg_appname: str = Field(default="lakercm-agent", env="PGAPPNAME")

    llm_endpoint: str = Field(default="databricks-claude-opus-4-6", env="LLM_ENDPOINT")
    # Path segment the agent's OpenAI-compatible client posts to (it appends
    # `/chat/completions`). Default is the classic FM serving path; the agent app
    # overrides it to the Unity AI Gateway model-service path
    # (`/ai-gateway/mlflow/v1`) via env, so `llm_endpoint` can be a UC model
    # service name (`catalog.schema.name`) fanned across models + guardrails.
    llm_base_path: str = Field(default="/serving-endpoints", env="LLM_BASE_PATH")

    # Reasoning-effort control for the agent LLM. Sent as the OpenAI Responses
    # API `reasoning.effort` (langchain maps `reasoning_effort` → responses when
    # use_responses_api is on). The Unity AI Gateway normalizes ONE uniform
    # effort per destination — so a single knob drives reasoning across the whole
    # (heterogeneous) traffic split, which the bare /chat/completions path cannot
    # do because the destinations' reasoning params are mutually incompatible.
    agent_llm_effort: str = Field(default="high", env="LLM_EFFORT")
    # Route agent LLM calls through the gateway's /responses API (True) vs the
    # legacy /chat/completions (False). /responses is what makes uniform reasoning
    # work across the blend and lets reasoning-model destinations tool-call while
    # reasoning behind the shared endpoint.
    agent_llm_use_responses_api: bool = Field(default=True, env="LLM_USE_RESPONSES_API")

    # Payer-policy Vector Search index backing the search_payer_policy retriever
    # tool. Empty (the default) means the index name is derived as
    # {catalog}.{schema_name}.payer_policy_index — matching what
    # scripts/create_vector_index.py provisions. Set explicitly only to point the
    # agent at a differently-named index.
    vector_search_index: str = Field(default="", env="LAKERCM_VS_INDEX")
    policy_retrieval_num_results: int = Field(
        default=4, env="LAKERCM_POLICY_NUM_RESULTS"
    )

    # The Reyden (Lakehouse//RT) warehouse the analytics tools pin every
    # system.ai.dbsql MCP call to (`_meta.warehouse_id`). The bundle injects it
    # from the agent's rt-warehouse resource. Empty -> those tools decline.
    dbsql_mcp_warehouse_id: str = Field(default="", env="DBSQL_MCP_WAREHOUSE_ID")

    # --- Knowledge graph traversal (feature-flagged; default OFF) ------
    # Enable the traverse_claims_graph tool and related KG features. The OntoBricks
    # bundle is dev-only, so this is on for dev and off for prod by default.
    kg_enabled: bool = Field(default=False, env="LAKERCM_KG_ENABLED")
    # Schema holding the OntoBricks triplestore view. Empty (the default) means
    # the schema is derived from the bundle variables, but the agent app doesn't
    # have access to DAB runtime values directly, so it reads this via env.
    kg_schema: str = Field(default="", env="LAKERCM_KG_SCHEMA")

    # --- Complexity-tiered model routing (feature-flagged; default OFF) ------
    # Route each conversation to a model+effort tier chosen once at conversation
    # start and pinned. OFF → today's single-endpoint behavior, byte-for-byte.
    # See agent/routing.py + docs/complexity-tiered-model-routing.md.
    routing_enabled: bool = Field(default=False, env="LLM_ROUTING_ENABLED")
    # Tier used when the classifier errors/times out or the pin store is
    # unavailable. Emitted with a routing.source trace tag + alerted — a router
    # stuck on the default must be observable, never silent.
    routing_default_tier: str = Field(default="med", env="LLM_ROUTING_DEFAULT_TIER")
    # Classifier model — a plain FM serving endpoint, NOT the gateway blend.
    # GPT-5.6 Luna is "the fast, cost-efficient model in OpenAI's GPT-5.6
    # family". It MUST be queried on /serving-endpoints, hence
    # classifier_base_path: in the deployed app llm_base_path is the gateway path
    # (/ai-gateway/mlflow/v1), so without an explicit base_path the classifier
    # call would hit the gateway, not the FM.
    classifier_endpoint: str = Field(
        default="databricks-gpt-5-6-luna", env="LLM_CLASSIFIER_ENDPOINT"
    )
    # Reasoning effort for the classifier call. Luna accepts none|low|medium|
    # high|xhigh (NOT minimal); anything else is coerced to medium. Empty omits
    # the parameter entirely — required if the classifier is ever pointed at a
    # Claude endpoint, which 400s on reasoning_effort over /chat/completions.
    classifier_reasoning_effort: str = Field(
        default="medium", env="LLM_CLASSIFIER_EFFORT"
    )
    classifier_base_path: str = Field(
        default="/serving-endpoints", env="LLM_CLASSIFIER_BASE_PATH"
    )
    # The classifier call sits on the pre-first-token path of a new
    # conversation. Fail fast: the default tier is cheap and observable, so a
    # stalled classifier must not hold up the reviewer. ~3s x 2 attempts.
    classifier_timeout_seconds: float = Field(
        default=3.0, env="LLM_CLASSIFIER_TIMEOUT_SECONDS"
    )
    # Shelf life of a thread -> tier pin. Upward-only still holds within the
    # window; past it the next turn re-decides, so one escalation cannot fix a
    # long-lived thread at the dearest tier forever. 0 disables expiry.
    routing_pin_ttl_seconds: float = Field(
        default=1800.0, env="LLM_ROUTING_PIN_TTL_SECONDS"
    )
    # Per-tier model-service endpoints (Phase 2). Empty → use llm_endpoint (the
    # existing blend) and vary ONLY reasoning_effort (Phase 1, no new infra).
    # LOW is only a distinct tier once llm_endpoint_low is set; until then it
    # resolves exactly like MED (see routing.tier_to_endpoint_effort).
    llm_endpoint_low: str = Field(default="", env="LLM_ENDPOINT_LOW")
    llm_endpoint_med: str = Field(default="", env="LLM_ENDPOINT_MED")
    llm_endpoint_high: str = Field(default="", env="LLM_ENDPOINT_HIGH")
    # Per-tier reasoning effort. The blend never receives none/low (Opus can
    # emit tool calls as plain text) — routing clamps it to medium in code, so
    # effort_low only ever reaches a dedicated low-tier service.
    effort_low: str = Field(default="low", env="LLM_EFFORT_LOW")
    effort_med: str = Field(default="medium", env="LLM_EFFORT_MED")
    effort_high: str = Field(default="high", env="LLM_EFFORT_HIGH")

    prompt_name: str = Field(default="", env="LAKERCM_PROMPT_NAME")
    champion_alias: str = Field(default="champion", env="LAKERCM_CHAMPION_ALIAS")
    candidate_alias: str = Field(default="candidate", env="LAKERCM_CANDIDATE_ALIAS")
    candidate_traffic_pct: int = Field(default=0, env="LAKERCM_CANDIDATE_TRAFFIC_PCT")
    eval_dataset_name: str = Field(default="", env="LAKERCM_EVAL_DATASET")
    gepa_reflection_model: str = Field(
        default="databricks-claude-sonnet-4-5",
        env="LAKERCM_GEPA_REFLECTION_MODEL",
    )

    # --- Eval rigor / promotion-gate tunables -------------------------------
    # Max metric calls GEPA may issue in a single optimize run. Caps cost AND
    # wall-clock: each call is a real agent rollout (an LLM round-trip). 150 is
    # ample for the ~40-row train split (GEPA skips perfect subsamples and
    # converges well under it); raise only for much larger datasets. 0 disables.
    gepa_max_metric_calls: int = Field(default=150, env="LAKERCM_GEPA_MAX_METRIC_CALLS")
    # The promotion gate refuses to decide on fewer than this many paired
    # eval examples — point estimates on tiny sets are noise.
    min_eval_samples: int = Field(default=20, env="LAKERCM_MIN_EVAL_SAMPLES")
    # The gate fails CLOSED if the candidate's rollouts error/empty on more than
    # this fraction of held-out rows — a prompt that broke the agent must not be
    # averaged into a "slightly worse" composite and slip through. Counts
    # predict_fn exceptions (the empty degraded row). 0.10 = block above 10%.
    max_degraded_rate: float = Field(default=0.10, env="LAKERCM_MAX_DEGRADED_RATE")
    # Train/holdout split percent for GEPA. GEPA reflects on the train split;
    # the post-GEPA eval AND the promotion gate score ONLY the holdout. 60 keeps
    # the holdout (~40%) comfortably above min_eval_samples on the ~70-row golden
    # set (a 70/30 split landed a 15-row holdout in-workspace once UC
    # serialization shifted the content-hash buckets).
    eval_train_pct: int = Field(default=60, env="LAKERCM_EVAL_TRAIN_PCT")
    # How many times run_eval repeats mlflow.genai.evaluate and averages.
    # Reduces judge/retrieval variance; 1 preserves historical behavior.
    eval_num_runs: int = Field(default=1, env="LAKERCM_EVAL_NUM_RUNS")
    # GEPA reflection minibatch size — examples per reflection prompt. Small
    # keeps the reflection prompt under the reflection model's context window.
    gepa_reflection_minibatch: int = Field(
        default=2, env="LAKERCM_GEPA_REFLECTION_MINIBATCH"
    )
    # Per-call timeout (seconds) for GEPA's INTERNAL reflection-model call.
    # GEPA wraps a string reflection_lm in gepa.lm.LM, which calls
    # litellm.completion with NO timeout (litellm default 600s) and num_retries=3
    # → a slow/stuck reflection stalls ~4×600s≈40min until the job timeout. We
    # route this through gepa_kwargs={"reflection_lm_kwargs": {...}} (the only
    # key MLflow's GepaPromptOptimizer doesn't override) to bound it.
    gepa_reflection_timeout: int = Field(
        default=120, env="LAKERCM_GEPA_REFLECTION_TIMEOUT"
    )
    # Wall-clock ceiling (seconds) for the WHOLE GEPA optimization, enforced via
    # gepa's TimeoutStopCondition stop_callback (checked at iteration boundaries).
    # Guarantees gepa.optimize RETURNS the best candidate (so the candidate gets
    # registered + the loop completes) even if reflections keep timing out —
    # because reflection failures don't increment max_metric_calls. Kept under
    # the job timeout_seconds (2700) so GEPA exits gracefully before the job is
    # hard-killed.
    gepa_wall_clock_seconds: int = Field(
        default=1500, env="LAKERCM_GEPA_WALL_CLOCK_SECONDS"
    )

    # --- Automated canary analysis (ACA) thresholds -------------------------
    # Kayenta-style trichotomy: score >= pass → auto-promote eligible;
    # marginal <= score < pass → hold for human; score < marginal → rollback.
    aca_pass_threshold: float = Field(default=95.0, env="LAKERCM_ACA_PASS")
    aca_marginal_threshold: float = Field(default=75.0, env="LAKERCM_ACA_MARGINAL")
    # Minimum minutes a candidate must bake on the canary before ACA scores it.
    canary_bake_minutes: int = Field(default=1440, env="LAKERCM_CANARY_BAKE_MINUTES")
    # Autonomy rung: "recommend" (post recommendation only),
    # "auto_promote" (promote on clean PASS, human rollback),
    # "staged" (auto-promote + auto-rollback on fast guardrails).
    auto_promote_mode: str = Field(default="recommend", env="LAKERCM_AUTO_PROMOTE_MODE")

    pipeline_id: str = Field(default="", env="PIPELINE_ID")

    # Confidence threshold above which an extraction is auto-verified
    # (no human review needed). Must mirror reviewer_app's threshold —
    # they share the same gold_extraction_labels_sync predicate.
    auto_verdict_threshold: float = Field(default=0.92, env="AUTO_VERDICT_THRESHOLD")

    # How long a pod caches the resolved system prompt before re-checking
    # MLflow Prompt Registry. Updates to @champion / @candidate aliases
    # propagate to running pods within this window — without it, prompt
    # changes require a pod bounce.
    prompt_resolution_ttl_seconds: float = Field(
        default=60.0, env="PROMPT_RESOLUTION_TTL_SECONDS"
    )

    # MLflow tracing + scorer-registration config. The Apps runtime sets
    # these via app.yaml; resolving them through Settings keeps every
    # consumer (observability.init_tracing, eval.scorers.register_scorers)
    # off os.getenv literals. Defaults match the historical hardcoded
    # values so unset envs preserve previous behavior.
    mlflow_tracking_uri: str = Field(default="databricks", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="/Shared/lakercm/agent-traces", env="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_experiment_id: str = Field(default="", env="MLFLOW_EXPERIMENT_ID")
    mlflow_logged_model_name: str = Field(default="", env="LAKERCM_LOGGED_MODEL_NAME")
    mlflow_enable_async_trace_logging: str = Field(
        default="true", env="MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"
    )
    mlflow_trace_enable_otlp_dual_export: str = Field(
        default="", env="MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT"
    )
    otel_exporter_otlp_traces_endpoint: str = Field(
        default="", env="OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    otel_service_name: str = Field(default="", env="OTEL_SERVICE_NAME")
    otel_resource_attributes: str = Field(default="", env="OTEL_RESOURCE_ATTRIBUTES")
    skip_scorer_registration: bool = Field(
        default=False, env="LAKERCM_SKIP_SCORER_REGISTRATION"
    )

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, data):
        # pydantic-settings V2's Field(env="…") doesn't reliably pick up
        # env vars in this app's configuration. Bridge the values we need
        # via explicit os.getenv calls so the deployed app sees the env
        # the Apps runtime injects (DATABRICKS_HOST, DATABRICKS_CLIENT_ID)
        # and the values app.yaml sets (DATABRICKS_CATALOG, etc.).
        if not isinstance(data, dict):
            return data

        def env_or(field_name, env_name):
            if data.get(field_name):
                return data[field_name]
            return os.getenv(env_name) or data.get(field_name)

        for field, env in (
            ("databricks_host", "DATABRICKS_HOST"),
            ("catalog", "DATABRICKS_CATALOG"),
            ("schema_name", "LAKERCM_SCHEMA"),
            ("endpoint_name", "ENDPOINT_NAME"),
            ("pg_database", "PGDATABASE"),
            ("pg_host", "PGHOST"),
            ("pg_sslmode", "PGSSLMODE"),
            ("pg_appname", "PGAPPNAME"),
            ("llm_endpoint", "LLM_ENDPOINT"),
            ("llm_base_path", "LLM_BASE_PATH"),
            ("agent_llm_effort", "LLM_EFFORT"),
            ("agent_llm_use_responses_api", "LLM_USE_RESPONSES_API"),
            ("vector_search_index", "LAKERCM_VS_INDEX"),
            ("policy_retrieval_num_results", "LAKERCM_POLICY_NUM_RESULTS"),
            ("dbsql_mcp_warehouse_id", "DBSQL_MCP_WAREHOUSE_ID"),
            ("kg_enabled", "LAKERCM_KG_ENABLED"),
            ("kg_schema", "LAKERCM_KG_SCHEMA"),
            ("routing_enabled", "LLM_ROUTING_ENABLED"),
            ("routing_default_tier", "LLM_ROUTING_DEFAULT_TIER"),
            ("classifier_endpoint", "LLM_CLASSIFIER_ENDPOINT"),
            ("classifier_reasoning_effort", "LLM_CLASSIFIER_EFFORT"),
            ("classifier_base_path", "LLM_CLASSIFIER_BASE_PATH"),
            ("classifier_timeout_seconds", "LLM_CLASSIFIER_TIMEOUT_SECONDS"),
            ("routing_pin_ttl_seconds", "LLM_ROUTING_PIN_TTL_SECONDS"),
            ("llm_endpoint_low", "LLM_ENDPOINT_LOW"),
            ("llm_endpoint_med", "LLM_ENDPOINT_MED"),
            ("llm_endpoint_high", "LLM_ENDPOINT_HIGH"),
            ("effort_low", "LLM_EFFORT_LOW"),
            ("effort_med", "LLM_EFFORT_MED"),
            ("effort_high", "LLM_EFFORT_HIGH"),
            ("prompt_name", "LAKERCM_PROMPT_NAME"),
            ("champion_alias", "LAKERCM_CHAMPION_ALIAS"),
            ("candidate_alias", "LAKERCM_CANDIDATE_ALIAS"),
            ("eval_dataset_name", "LAKERCM_EVAL_DATASET"),
            ("gepa_reflection_model", "LAKERCM_GEPA_REFLECTION_MODEL"),
            ("gepa_max_metric_calls", "LAKERCM_GEPA_MAX_METRIC_CALLS"),
            ("min_eval_samples", "LAKERCM_MIN_EVAL_SAMPLES"),
            ("eval_train_pct", "LAKERCM_EVAL_TRAIN_PCT"),
            ("eval_num_runs", "LAKERCM_EVAL_NUM_RUNS"),
            ("gepa_reflection_minibatch", "LAKERCM_GEPA_REFLECTION_MINIBATCH"),
            ("gepa_reflection_timeout", "LAKERCM_GEPA_REFLECTION_TIMEOUT"),
            ("gepa_wall_clock_seconds", "LAKERCM_GEPA_WALL_CLOCK_SECONDS"),
            ("aca_pass_threshold", "LAKERCM_ACA_PASS"),
            ("aca_marginal_threshold", "LAKERCM_ACA_MARGINAL"),
            ("canary_bake_minutes", "LAKERCM_CANARY_BAKE_MINUTES"),
            ("auto_promote_mode", "LAKERCM_AUTO_PROMOTE_MODE"),
            ("pipeline_id", "PIPELINE_ID"),
            ("auto_verdict_threshold", "AUTO_VERDICT_THRESHOLD"),
            ("prompt_resolution_ttl_seconds", "PROMPT_RESOLUTION_TTL_SECONDS"),
            ("mlflow_tracking_uri", "MLFLOW_TRACKING_URI"),
            ("mlflow_experiment_name", "MLFLOW_EXPERIMENT_NAME"),
            ("mlflow_experiment_id", "MLFLOW_EXPERIMENT_ID"),
            ("mlflow_logged_model_name", "LAKERCM_LOGGED_MODEL_NAME"),
            (
                "mlflow_enable_async_trace_logging",
                "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING",
            ),
            (
                "mlflow_trace_enable_otlp_dual_export",
                "MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT",
            ),
            (
                "otel_exporter_otlp_traces_endpoint",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            ),
            ("otel_service_name", "OTEL_SERVICE_NAME"),
            ("otel_resource_attributes", "OTEL_RESOURCE_ATTRIBUTES"),
            ("skip_scorer_registration", "LAKERCM_SKIP_SCORER_REGISTRATION"),
        ):
            value = env_or(field, env)
            # An app env var cannot be empty (the Apps deployment API rejects an
            # entry with no value), so the bundle passes "none" for a tier with
            # no dedicated model service.
            if field in _TIER_ENDPOINT_FIELDS and str(value).strip().lower() == "none":
                data.pop(field, None)
                continue
            if value:
                data[field] = value

        # PGUSER precedence (high → low):
        #   1. Explicit PGUSER env var rendered from app.yaml — typically
        #      the group-backed Lakebase role name (e.g. "lakercm-
        #      checkpoint-runtime"). Session-as the group role so the
        #      LangGraph PostgresSaver owns its checkpoint_* tables under
        #      a stable identity that outlives any specific SP.
        #   2. DATABRICKS_CLIENT_ID auto-injected by the Apps runtime =
        #      the running SP's client_id. Fallback for older targets
        #      that haven't migrated to the group-role pattern yet.
        if not data.get("pg_user"):
            data["pg_user"] = os.getenv("PGUSER") or os.getenv(
                "DATABRICKS_CLIENT_ID", ""
            )
        return data

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
