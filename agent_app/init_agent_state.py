"""
One-shot deploy step: create LangGraph checkpointer + store tables.

Run this once after a deploy, NOT on every worker boot. It is idempotent:
PostgresSaver.setup() / PostgresStore.setup() only apply pending migrations.

Usage:
    python init_agent_state.py

Env required:
    DATABRICKS_HOST, PGHOST, PGDATABASE, PGUSER, PGPORT, PGSSLMODE, ENDPOINT_NAME
    (same as the running app)
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("init_agent_state")


def main() -> int:
    from services.lakehouse_db import get_db
    from langgraph.checkpoint.postgres import PostgresSaver

    pool = get_db().get_pool()

    logger.info("Running PostgresSaver.setup()...")
    PostgresSaver(pool).setup()
    logger.info("Checkpointer schema ready")

    # Store setup is wired in Step 5 (requires the vector extension).
    try:
        from langgraph.store.postgres import PostgresStore  # noqa: F401
        from services.store import get_store

        logger.info("Running PostgresStore.setup()...")
        get_store().setup()
        logger.info("Store schema ready")
    except ImportError:
        logger.info("PostgresStore not installed yet — skipping store setup")
    except Exception as e:
        logger.warning(
            "Store setup failed (likely missing 'vector' extension — "
            "apply the alembic migration that enables it first): %s",
            e,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
