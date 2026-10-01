"""
Alembic Environment Configuration for LakeRCM Lakebase

Connects to Lakebase Autoscaling using postgres.generate_database_credential().
PG* env vars must be set (from app.yaml or manually for local dev).
"""

import os
import sys
import logging
import traceback
from logging.config import fileConfig

from sqlalchemy import pool, create_engine
from alembic import context


def _boot(msg: str) -> None:
    """Print to stderr with flush — survives process exit better than logger."""
    print(f"[alembic.boot] {msg}", file=sys.stderr, flush=True)


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from databricks.sdk import WorkspaceClient

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

target_metadata = None


def get_lakebase_connection_url() -> str:
    """
    Build PostgreSQL connection URL for Lakebase Autoscaling.

    Uses postgres.generate_database_credential() for password.
    PG* env vars must be set (PGHOST, PGUSER, PGDATABASE, PGPORT).
    """
    logger.info("Resolving Lakebase connection for migrations...")

    workspace_client = WorkspaceClient()

    pg_host = os.environ.get("PGHOST", "")
    # PGUSER falls back to DATABRICKS_CLIENT_ID (auto-injected by the Apps
    # runtime) — the running SP's client_id IS the Postgres role name for
    # Lakebase OAuth.
    pg_user = os.environ.get("PGUSER") or os.environ.get("DATABRICKS_CLIENT_ID", "")
    pg_database = os.environ.get("PGDATABASE", "databricks_postgres")
    pg_port = os.environ.get("PGPORT", "5432")
    pg_sslmode = os.environ.get("PGSSLMODE", "require")

    if not pg_user:
        logger.info("PGUSER not set, discovering service principal...")
        me = workspace_client.current_user.me()
        pg_user = getattr(me, "user_name", None) or str(getattr(me, "id", ""))

    if not pg_host:
        raise RuntimeError(
            "PGHOST not set. For Lakebase Autoscaling, set PGHOST to the "
            "endpoint hostname from the Lakebase project."
        )

    _boot(
        f"pg_host={pg_host} pg_user={pg_user} pg_database={pg_database} pg_port={pg_port}"
    )

    # Generate credential via Lakebase Autoscaling API
    endpoint_name = os.environ.get(
        "ENDPOINT_NAME",
        "projects/lakercm/branches/prod/endpoints/primary",
    )
    _boot(f"requesting credential for endpoint={endpoint_name}")
    try:
        credential = workspace_client.postgres.generate_database_credential(
            endpoint=endpoint_name
        )
        pg_password = credential.token
        _boot(f"credential acquired (len={len(pg_password) if pg_password else 0})")
    except Exception as e:
        _boot(f"autoscaling credential failed: {type(e).__name__}: {e}")
        _boot("falling back to workspace OAuth token")
        pg_password = workspace_client.config.oauth_token().access_token
        _boot(f"OAuth token acquired (len={len(pg_password) if pg_password else 0})")

    if not pg_password:
        raise RuntimeError("Failed to get credential for Lakebase")

    connection_url = (
        f"postgresql+psycopg://{pg_user}:{pg_password}"
        f"@{pg_host}:{pg_port}/{pg_database}"
        f"?sslmode={pg_sslmode}"
    )

    _boot("connection URL built")
    return connection_url


def run_migrations_offline() -> None:
    url = get_lakebase_connection_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    _boot("run_migrations_online: start")
    try:
        connection_url = get_lakebase_connection_url()
    except Exception as e:
        _boot(f"FATAL get_lakebase_connection_url raised: {type(e).__name__}: {e}")
        _boot(traceback.format_exc())
        raise

    _boot("creating SQLAlchemy engine")
    connectable = create_engine(connection_url, poolclass=pool.NullPool)

    _boot("opening connection to Lakebase (psycopg connect)")
    try:
        with connectable.connect() as connection:
            _boot("connection opened; configuring alembic context")
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
            )

            with context.begin_transaction():
                _boot("running migrations (context.run_migrations)")
                context.run_migrations()
                _boot("migrations complete")
    except Exception as e:
        _boot(f"FATAL during connect/migrate: {type(e).__name__}: {e}")
        _boot(traceback.format_exc())
        raise


if context.is_offline_mode():
    _boot("entering offline migration mode")
    run_migrations_offline()
else:
    _boot("entering online migration mode")
    run_migrations_online()
