terraform {
  required_version = ">= 1.5"
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = ">= 1.128.0"
    }
  }
}

provider "databricks" {
  profile = var.databricks_profile
}
