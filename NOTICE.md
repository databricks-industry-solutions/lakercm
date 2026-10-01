Copyright (2026) Databricks, Inc.

This Software includes software developed at Databricks (https://www.databricks.com/) and its use is subject to the included LICENSE file.

## Third-party software

This repository redistributes no third-party source code, binaries, fonts or
images. The packages below are installed from PyPI and npm when the apps are
built or deployed, and each is subject to its own license.

### Python (apps, jobs, deploy scripts and the synthetic data generator)

| License | Packages |
|---|---|
| Apache-2.0 | databricks-sdk, mlflow, python-multipart, opentelemetry-distro, opentelemetry-exporter-otlp-proto-grpc, opentelemetry-instrumentation-fastapi, opentelemetry-instrumentation-psycopg |
| Apache-2.0 or BSD-3-Clause | python-dateutil |
| BSD-3-Clause | uvicorn, httpx, sse-starlette, psutil |
| BSD | reportlab (synthetic data generator) |
| MIT | fastapi, pydantic, pydantic-settings, alembic, SQLAlchemy, pytz, cachetools, langchain, langchain-core, langchain-community, langchain-openai, langgraph, langgraph-prebuilt, langgraph-checkpoint-postgres, ag-ui-langgraph, gepa, faker (synthetic data generator) |
| LGPL-3.0-only | psycopg, psycopg-pool (installed from PyPI; not modified or redistributed) |
| Databricks License | databricks-agents |

### npm (reviewer app frontend)

| License | Packages |
|---|---|
| MIT | react, react-dom, react-router, react-markdown, remark-gfm, axios, date-fns, @ag-ui/client |
| MIT (build and lint only) | vite, @vitejs/plugin-react, eslint, eslint-plugin-react, eslint-plugin-react-hooks, eslint-plugin-react-refresh, @types/react, @types/react-dom |

`ag-ui-langgraph` and `@ag-ui/client` publish no license field in their package
metadata; both come from the MIT-licensed
[ag-ui-protocol/ag-ui](https://github.com/ag-ui-protocol/ag-ui) repository.
