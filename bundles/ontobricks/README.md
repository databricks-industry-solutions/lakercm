# `ontobricks` bundle — LakeRCM knowledge graph (dev-only)

One Databricks App that runs [OntoBricks](https://github.com/databrickslabs/ontobricks)
(Databricks Labs) **unmodified**: its UI + REST `/api/v1` + GraphQL **and** its
MCP server, mounted at `/mcp` in the same process. LakeRCM's agent pulls the
graph over `/mcp`; the reviewer app renders it in a Knowledge Graph tab over
`/api/kg`.

## Why the source is not committed

OntoBricks is under the **Databricks License**, so its source is never checked
into this repo. Instead it is fetched at a **pinned commit** and only our thin
entrypoint is added on top:

- **`ontobricks.lock.json`** — pins `{repo, ref, sha}`. Re-pin by SHA, one PR
  per bump.
- **`scripts/fetch_ontobricks_source.py`** — shallow-fetches that exact SHA into
  the **gitignored** `build/ontobricks/`, verifies `HEAD == sha`, and copies
  `app/combined_app.py` in. It is idempotent (`--check` verifies without
  fetching) and runs as:
  - the `ontobricks_source` **pre-step** in `scripts/deploy_all.py` (before
    `bundle validate`/`summary`, because the app's `source_code_path` points
    into that fetched tree), and
  - a **CI step** before `bundle validate` (`.github/workflows/ci.yml`),
    mirroring the Genie render step.

`build/` is in `.gitignore`; `git ls-files build/` must stay empty.

## The one app

`app/combined_app.py` (ours; the only file added to the fetched tree) builds
OntoBricks' `create_app()` and mounts `create_mcp_server(mode="mounted")` at
`/mcp` in the same ASGI app, running both lifespans. `mounted` mode calls the
main app on `http://localhost:$DATABRICKS_APP_PORT` (loopback, no auth headers),
and `/api/`, `/graphql/`, `/mcp` all sit ahead of the session/CSRF/permission
middleware.

**Command** (`resources/ontobricks_app.yml`):
`uv run --frozen --extra lakebase python combined_app.py` — the dev dependency
group is kept **on purpose** (`fastmcp` lives there; `combined_app.py` imports
it to mount `/mcp`).

**Warehouse:** the **standard** SQL warehouse (`app_warehouse_name`), never the
read-only Reyden (Lakehouse//RT) one — OntoBricks builds run DDL/COPY/
materialization (AGENTS.md forbids DDL/DML/GRANT on RT).

## Deploy

Dev-only for now (`scripts/bundle_registry.py`, `targets=("dev",)`), placed last
so an OntoBricks failure blocks nothing else. Needs `foundation`, `lakebase`,
`pipelines`, `apps`. Requires Databricks CLI **>= 1.15.0** (the app carries
`resources.apps.*.config`, and direct-engine app updates 400 before 1.15.0).
Deploy through CD's deployer SP — interactive user deploys currently fail in the
dev workspace.

Post-steps (see the `BundleSpec.post` comment):

1. **`ontobricks_start`** — roll + start the app, the way the apps bundle does.
   The bootstrap that follows talks HTTP, so the app has to be running.
2. **`ontobricks_bootstrap`** (`scripts/bootstrap_ontobricks.py`) — the four
   things no DAB resource can express, all idempotent:
   * **CAN_MANAGE for the app on itself.** OntoBricks decides who is an admin by
     reading the app's own ACL, which it cannot do unless its SP holds
     CAN_MANAGE on the app — so until this lands, *nobody* is an admin and the
     app answers `PermissionDenied: ... apps.ruleSets/get`. DAB cannot declare
     it: `service_principal_client_id` is a read-only API output with no
     bundle-schema field, and the SP does not exist until the app does. Applied
     with PATCH, never PUT, which would drop the bundle-managed ACL entries.
     Note this gates only the UI and `/domain/*`; `/api/`, `/graphql/` and
     `/mcp` sit ahead of that middleware, so the agent never needed it.
   * **Narrow registry privileges + an assertion.** OntoBricks' own helper
     (`back/core/databricks/lakebase/grants.py`) asks for `ALL_PRIVILEGES` on
     the whole catalog, best-effort. It fails here (no MANAGE), so the app SP is
     granted only what it needs on the registry schema, and the step then
     *asserts* it holds nothing wider than `USE_CATALOG`. An assertion, not a
     "defensive revoke": the revoke is a no-op today and would quietly stop
     protecting anything the day the SP does get MANAGE.
   * **`POST /settings/registry/initialize`** — creates the registry's Postgres
     tables. Until it runs `/api/v1/domains` answers 502 and the agent's MCP
     tools return an empty graph with no actionable error.
   * **The LakeRCM domain** — import, publish and build from
     `scripts/lakercm_domain_content.py` (6 classes, 7 object properties, 6
     entity and 7 relationship mappings over the gold layer; no raw PHI).

   `--check` reports without changing anything; `--rebuild` re-imports a domain
   that is already published (the step is otherwise a no-op on a built graph).

The registry's UC schema and volume are ordinary bundle resources
(`resources/registry.yml`), not artefacts of the initialise call, so
`bundle deploy` guarantees they exist before the bootstrap runs. Creating them
the other way round makes initialise report "could not create binary volume"
and leaves the registry half-configured.

**Still to come:** `kg_layer` — the `kg_finding` secure view and
`claims_rule_metrics`.

## Known upstream caveat: prefer id-addressed reads over `describe_entity`

Two of OntoBricks' MCP `describe_entity` output sections are unreliable, so
**anything that needs a specific entity should address it by id/URI**, not read
it out of a `describe_entity` rendering. Verified live on dev against the pinned
SHA:

- the node listed under `── Matching Entities ──` is often **not** the entity
  that matched the query, and
- some nodes render as `(Unknown type)` even though their `rdf:type` is present
  in the graph.

**Root cause (upstream, not ours).** The formatter picks its "matching" seeds
with a heuristic — it walks the triples grouped by subject and takes the first
`seed_count` subjects that happen to carry a non-type literal
(`src/mcp-server/server/formatting.py`). It cannot do better: the API it renders
(`src/api/routers/digitaltwin.py`) returns only `seed_count: int`, **not the seed
URIs**, so which entities actually matched is information the formatter never
receives. `(Unknown type)` is the same shape of problem — a label resolved from
the triples it happens to have grouped rather than from the type edge.

**Do not patch this.** The fetched tree is Databricks-licensed and gitignored;
`app/combined_app.py` is the only file we add (see above). A pin bump does not
help either — `origin/0.9.0` carries the identical heuristic. Fixing it properly
needs an upstream API change (return the seed URIs), so it is an upstream issue
to file, not a local edit.

**The reliable path** is GraphQL, where the document id is the key:

```graphql
{ document(id: "synthetic-1790440247-0011-denial.pdf") {
    documentType isAutomated reviewReasons confidenceScore
    billedTo { name } deniedFor { code description category }
    hasDiagnosis { code billable } hasProcedure { code } } }
```

That returned exactly the row under test — `denial_management`,
`isAutomated: false`, `reviewReasons: [missing_member_id]`, confidence
`0.999785`, payer Silver Harbor, CARC 119 (`frequency_benefit_limit`),
diagnoses `R07.9`/`I48.91`/`I25.10` all billable, procedure `93000` — with no
ambiguity about which entity was being described. `documents(limit/offset/
search)` is the list form, and the typed edges (`billedTo`, `hasDiagnosis`,
`hasProcedure`, `deniedFor`) traverse from there. Use `describe_entity` for
orientation and prose only; never parse it for a fact a demo or an assertion
depends on.
