"""
LakeRCM Configuration

Pydantic Settings for Lakebase Autoscaling, UC Volume, and SQL Warehouse.
"""

import os
import logging
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    databricks_host: str = Field(default="", env="DATABRICKS_HOST")
    databricks_token: str = Field(default="", env="DATABRICKS_TOKEN")

    # Lakebase Autoscaling
    endpoint_name: str = Field(
        default="projects/lakercm/branches/prod/endpoints/primary",
        env="ENDPOINT_NAME",
    )
    pg_database: str = Field(default="databricks_postgres", env="PGDATABASE")
    pg_host: str = Field(default="", env="PGHOST")
    pg_port: int = Field(default=5432, env="PGPORT")
    pg_user: str = Field(default="", env="PGUSER")
    pg_sslmode: str = Field(default="require", env="PGSSLMODE")
    pg_appname: str = Field(default="lakercm", env="PGAPPNAME")

    # The analytics routes read the Reyden (Lakehouse//RT) warehouse. The bundle
    # injects its id as SQL_WAREHOUSE_ID; the name is the discovery fallback.
    sql_warehouse_id: str = Field(default="", env="SQL_WAREHOUSE_ID")
    warehouse_name: str = Field(default="lakercm_rt_warehouse", env="WAREHOUSE_NAME")

    catalog: str = Field(default="main", env="DATABRICKS_CATALOG")
    lakercm_schema: str = Field(default="lakercm", env="LAKERCM_SCHEMA")
    # Knowledge-graph panel. Both are required at runtime: without the schema the
    # triplestore view cannot be qualified, so the panel stays off rather than
    # issuing a query against a half-known name. The agent app learned this the
    # hard way in #88 -- its two env vars were missing and the KG tool shipped
    # dark, with nothing to say so.
    kg_enabled: bool = Field(default=False, env="LAKERCM_KG_ENABLED")
    kg_schema: str = Field(default="", env="LAKERCM_KG_SCHEMA")

    pipeline_id: str = Field(default="", env="PIPELINE_ID")
    analytics_pipeline_id: str = Field(default="", env="ANALYTICS_PIPELINE_ID")

    @model_validator(mode="before")
    @classmethod
    def load_from_bundle_vars(cls, data: dict) -> dict:
        logger.info("Loading LakeRCM configuration from environment...")

        def get_env_value(
            field_name: str, bundle_var_name: str, default: str = ""
        ) -> str:
            value = os.getenv(field_name.upper())
            if value:
                return value
            value = os.getenv(f"BUNDLE_VAR_{bundle_var_name}")
            if value:
                return value
            return default

        data["sql_warehouse_id"] = get_env_value(
            "sql_warehouse_id", "sql_warehouse_id", ""
        )
        data["warehouse_name"] = get_env_value(
            "warehouse_name", "rt_warehouse_name", "lakercm_rt_warehouse"
        )
        data["catalog"] = get_env_value("databricks_catalog", "catalog", "main")
        data["lakercm_schema"] = get_env_value(
            "lakercm_schema", "lakercm_schema", "lakercm"
        )
        # Explicit bridge, like every other field here: pydantic's `env=` alone
        # does not populate these through this validator.
        data["kg_enabled"] = get_env_value("lakercm_kg_enabled", "kg_enabled", "false")
        data["kg_schema"] = get_env_value(
            "lakercm_kg_schema", "ontobricks_registry_schema", ""
        )
        data["endpoint_name"] = get_env_value(
            "endpoint_name",
            "endpoint_name",
            "projects/lakercm/branches/prod/endpoints/primary",
        )
        data["transcription_endpoint"] = get_env_value(
            "transcription_endpoint",
            "transcription_endpoint",
            "databricks-gemini-3-5-flash",
        )
        data["transcription_engine"] = get_env_value(
            "transcription_engine", "transcription_engine", "fm"
        )
        data["agent_app_url"] = get_env_value(
            "lakercm_agent_app_url",
            "agent_app_url",
            "",
        )
        # Apps runtime auto-injects DATABRICKS_CLIENT_ID = the running SP's
        # client_id, which IS the Postgres role name for Lakebase OAuth.
        # Fall back to it if PGUSER isn't set explicitly.
        if not data.get("pg_user"):
            data["pg_user"] = os.getenv("PGUSER") or os.getenv(
                "DATABRICKS_CLIENT_ID", ""
            )
        return data

    # Agent app URL (standalone agent service)
    agent_app_url: str = Field(default="", env="LAKERCM_AGENT_APP_URL")

    llm_endpoint: str = Field(default="databricks-claude-opus-4-6", env="LLM_ENDPOINT")

    # Speech-to-text: a managed multimodal FM serving endpoint that accepts an
    # OpenAI-compatible `audio_url` content block (see services/transcription.py).
    # `transcription_engine` switches to a self-hosted Whisper path when != "fm".
    transcription_endpoint: str = Field(
        default="databricks-gemini-3-5-flash", env="TRANSCRIPTION_ENDPOINT"
    )
    transcription_engine: str = Field(default="fm", env="TRANSCRIPTION_ENGINE")
    transcription_rate_per_min: int = Field(
        default=60, env="TRANSCRIPTION_RATE_PER_MIN"
    )

    api_port: int = Field(default=8000)
    api_host: str = Field(default="0.0.0.0")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    cors_origins: List[str] = Field(default=["*"])

    db_pool_min_connections: int = Field(default=1)
    db_pool_max_connections: int = Field(default=10)

    # Auto-verdict threshold, for display only (the confidence marker in the
    # review UI). Routing is the pipeline's: the app reads
    # gold_extraction_labels.is_automated, which also sends a confident document
    # to review for a code or member-ID problem. The bundle sets this from the
    # same `auto_verdict_threshold` variable the pipeline uses.
    auto_verdict_threshold: float = Field(default=0.92, env="AUTO_VERDICT_THRESHOLD")
    automated_reviewer_email: str = Field(
        default="<automated>", env="AUTOMATED_REVIEWER_EMAIL"
    )

    def discover_warehouse_id(self, workspace_client) -> str:
        if self.sql_warehouse_id:
            return self.sql_warehouse_id

        try:
            logger.info(
                "Auto-discovering warehouse ID for name: %s", self.warehouse_name
            )
            warehouses = list(workspace_client.warehouses.list())

            for wh in warehouses:
                if wh.name == self.warehouse_name:
                    self.sql_warehouse_id = wh.id
                    logger.info(
                        "Discovered warehouse ID: %s (exact match: %s)", wh.id, wh.name
                    )
                    return self.sql_warehouse_id

            for wh in warehouses:
                name = wh.name or ""
                # Personal dev warehouses (Databricks auto-prefixes them with
                # `[dev <user>]`) are never the right answer for an app SP —
                # the SP doesn't have CAN_USE. Skip so we don't silently lock
                # onto one when the bundle-managed warehouse is missing.
                if name.startswith("[dev "):
                    continue
                if name.endswith(self.warehouse_name):
                    self.sql_warehouse_id = wh.id
                    logger.info(
                        "Discovered warehouse ID: %s (suffix match: '%s')",
                        wh.id,
                        wh.name,
                    )
                    return self.sql_warehouse_id

            available = [wh.name for wh in warehouses]
            logger.warning(
                "No warehouse found matching name '%s'. Available: %s",
                self.warehouse_name,
                available,
            )
        except Exception as e:
            logger.error("Failed to auto-discover warehouse ID: %s", e)

        return self.sql_warehouse_id

    def get_warehouse_id(self) -> str:
        if not self.sql_warehouse_id:
            raise ValueError(
                "SQL Warehouse ID is not configured. "
                "Set SQL_WAREHOUSE_ID or ensure WAREHOUSE_NAME is valid for auto-discovery."
            )
        return self.sql_warehouse_id

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "allow",
    }


settings = Settings()
