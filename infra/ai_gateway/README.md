# Agent LLM — Unity AI Gateway model service (Terraform)

Infrastructure-as-Code for the agent's Unity AI Gateway **model service**
(`<catalog>.<schema>.lakercm-agent-llm`) — the governed endpoint that fans
the agent's LLM traffic across several foundation models with per-conversation
session affinity, and which the agent app queries over the `/responses` API with
a uniform reasoning effort.

## Why this ONE service is still a standalone module

> **Updated (CLI v1.17.0).** The original reason — "DABs do not expose a
> model-service resource type (CLI v1.14.x), on the roadmap targeting 2026" — no
> longer holds. DABs now has `model_services` (plus `model_provider_services` and
> `mcp_services`), on the ordinary terraform engine. The per-tier services moved
> out of here into the apps bundle (see below). **This blend service did not**, for
> a specific reason:
>
> `databricks_ai_gateway_model_service.agent_llm` was adopted by `terraform
> import` and relies on `lifecycle { ignore_changes = [parent, model_service_id] }`
> — the read/import API returns neither field (only the derived `name`), and
> `parent` is immutable. DABs `lifecycle` supports **only `prevent_destroy`**;
> `ignore_changes` does not exist in the v1.17.0 bundle schema. So binding the
> live service into a bundle would see null immutable identity and plan a
> **REPLACE** — destroying the service and its payload table — or, with
> prevent_destroy, fail permanently. Neither is acceptable, so the already-live
> blend stays here until DABs gains `ignore_changes` (or a delete + recreate
> window is acceptable).
>
> The rule of thumb this leaves: **greenfield gateway services go in DABs;
> already-imported ones stay in Terraform.**

The Terraform provider resource is `databricks_ai_gateway_model_service` (Public
Beta, provider `>= 1.128.0` — the same provider the bundles already use).

The agent **app** (bundle-managed) points at this service by name via
`var.agent_llm_endpoint` in `bundles/_shared/variables.yml`; this module owns the
service's routing/destinations/inference-table. They are decoupled — deploy the
app with the bundle, manage the endpoint here.

## First-time adoption (import the already-live service — NO recreate)

The live service was originally created via the `databricks ai-gateway` CLI.
Import it into Terraform state instead of recreating it (a recreate would drop
the inference table + cause downtime):

```bash
cd infra/ai_gateway
export TF_VAR_databricks_profile=<profile> TF_VAR_catalog=<catalog>
terraform init
terraform import databricks_ai_gateway_model_service.agent_llm \
  model-services/<catalog>.<schema>.lakercm-agent-llm
terraform plan   # MUST report "No changes" — confirms this config matches live
```

`terraform import` on this Beta resource only populates the derived `name`, not
the create-time identity inputs (`parent`, `model_service_id`) — those are
immutable, so `main.tf` marks them `ignore_changes` to keep the import a clean
no-op instead of a destroy+recreate.

## Changing the blend (weights, models, effort routing)

Edit `main.tf` (`local.destination_models` / `local.traffic_percentage` /
`local.fallback_model`), then:

```bash
terraform plan    # review
terraform apply
```

Notes:
- Weights must sum to 100 across `destinations`.
- Only add destinations that tool-call **and** accept a uniform `reasoning.effort`
  over the gateway `/responses` API. (Empirically, the Bedrock-backed grok
  destination 400s on `/responses` — "Invalid task type" — so it is excluded;
  re-confirmed for Grok 4.6 on 2026-09-24.) For the tier services,
  `scripts/probe_tier_destinations.py` checks exactly this rule.
- The reasoning effort itself is sent by the **app**, not set here
  (`var.agent_llm_effort` in the bundle).

## Complexity-tiered routing — per-tier services moved to `bundles/ai_gateway`

`tiers.tf` is **gone**. The per-tier services, `…-agent-llm-low`, `…-agent-llm-med`
and `…-agent-llm-high`, each an 80/20 split with a fallback from a third
provider and no model shared between tiers, now live as DABs `model_services`
in:

    bundles/ai_gateway/resources/model_services.yml

(deployed for the dev target only, together with dev's own copy of the blend in
`agent_llm.yml`; this Terraform module still owns prod's blend)

They were never applied from Terraform, so they are greenfield: no import, no
`ignore_changes`, no adoption hazard — which is exactly why they could move and
the blend could not. They live in their own `engine: direct` bundle because
`model_services` is **direct-engine only** and `bundles/apps` pins terraform (see
`bundles/ai_gateway/databricks.yml`). They are still **deferred** (inert, excluded
from that bundle's `resources/*.yml`) because Phase 2 remains gated on the on-domain eval,
the same staging `tiers.tf` had. That file holds the activation runbook.

Moving them also removes a real risk: two IaC systems owning the same
resources. The blend (`…-agent-llm`) stays the ultimate fallback either way, and
`terraform plan` here must still read **No changes** — it is untouched by this.

Verified read-only against the workspace (CLI v1.17.0): the definition validates
and plans **3 to add, 0 to change, 0 to delete**. Every destination must also
pass `scripts/probe_tier_destinations.py --floor` before activation.

## State backend (caveat)

This module uses **local state** (`terraform.tfstate`, git-ignored). That means
the state lives only on the operator's machine — fine for single-operator
management, but for team/CI use, migrate to a shared backend and have others
re-`import` or pull the shared state. Because the resource is identified by its
UC name and `ignore_changes` covers the identity fields, re-importing on another
machine is safe and idempotent.
