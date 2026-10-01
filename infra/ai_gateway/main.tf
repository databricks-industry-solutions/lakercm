# Unity AI Gateway model service for the LakeRCM agent LLM.
#
# This is the IaC replacement for the imperative `databricks ai-gateway
# create-model-service` CLI flow. It is a STANDALONE Terraform module (not a
# Databricks Asset Bundle) because DABs do not yet expose a model-service
# resource type (CLI v1.14.1) — native DABs/SDK/CLI CRUD is on the roadmap
# (targeting 2026). The `databricks_ai_gateway_model_service` resource is
# Public Beta in the databricks provider (>= 1.128.0 — the same provider the
# bundle already uses).
#
# The agent app queries this service via the gateway /responses API with a
# uniform reasoning effort and a per-conversation session-affinity header; see
# bundles/_shared/variables.yml and agent_app/agent/llm.py. The app itself is
# bundle-managed and points at this service by name (var.agent_llm_endpoint).
#
# FIRST-TIME ADOPTION (no recreate — import the already-live service):
#   terraform import databricks_ai_gateway_model_service.agent_llm \
#     model-services/<catalog>.<schema>.lakercm-agent-llm
#   terraform plan   # MUST show "No changes" — proves this config matches live
# Thereafter: edit here + `terraform apply` (weights, models, effort routing).

locals {
  # Even split across the reasoning-capable destinations that tool-call via the
  # gateway /responses API. Edit weights/roster here — this is the source of
  # truth once the live service is imported.
  destination_models = [
    "databricks-claude-opus-5",
    "databricks-gpt-5-6-sol",
    "databricks-glm-5-3",
    # NOT v4-pro-0813: that endpoint is DEPRECATED and the gateway 400s it
    # ("This endpoint ... is deprecated"), so a fifth of every agent turn failed.
    # Verified by direct invocation on 2026-09-30. Its serving-endpoints record
    # still reads state.ready=READY, so only a real call surfaces this.
    "databricks-deepseek-v4-1-flash",
    "databricks-kimi-k3",
  ]
  traffic_percentage = 20 # 5 destinations × 20 = 100
  fallback_model     = "databricks-claude-opus-5"
  schema_ref         = "schemas/${var.catalog}.${var.schema}"
}

resource "databricks_ai_gateway_model_service" "agent_llm" {
  model_service_id = var.model_service_id
  parent           = local.schema_ref

  # `parent` and `model_service_id` are create-time identity inputs that the
  # read/import API does not return (it only returns the derived `name`), so a
  # freshly-imported resource shows them as null and `parent` (immutable) would
  # otherwise force a replace. Both are immutable identity — ignore post-import
  # drift on them so adoption is a clean no-op instead of a destroy+recreate.
  lifecycle {
    ignore_changes = [parent, model_service_id]
  }

  config = {
    routing = {
      destinations = [
        for m in local.destination_models : {
          destination_type   = "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL"
          name               = "system.ai.${m}"
          traffic_percentage = local.traffic_percentage
          pay_per_token_config = {
            model = "models/system.ai.${m}"
          }
        }
      ]
      fallback = {
        destinations = [
          {
            destination_type   = "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL"
            name               = "system.ai.${local.fallback_model}"
            traffic_percentage = 0
            pay_per_token_config = {
              model = "models/system.ai.${local.fallback_model}"
            }
          }
        ]
      }
    }

    # Payload logging table (created + owned by the service).
    inference_table = {
      disabled          = false
      parent            = local.schema_ref
      table_name_prefix = var.inference_table_prefix
    }
  }
}
