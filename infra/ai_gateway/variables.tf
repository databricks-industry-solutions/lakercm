variable "databricks_profile" {
  type        = string
  description = "Databricks CLI profile for the workspace hosting the model service (TF_VAR_databricks_profile)."
}

variable "catalog" {
  type        = string
  description = "Unity Catalog catalog that holds the model service + its inference table (TF_VAR_catalog)."
}

variable "schema" {
  type        = string
  default     = "lakercm"
  description = "Schema under the catalog for the model service + inference table."
}

variable "model_service_id" {
  type        = string
  default     = "lakercm-agent-llm"
  description = "Leaf name of the model service (must match var.agent_llm_endpoint in the bundle)."
}

variable "inference_table_prefix" {
  type        = string
  default     = "lakercm_agent_llm"
  description = "Prefix for the AI-Gateway payload/inference table (produces <prefix>_payload)."
}
