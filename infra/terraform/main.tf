# infra/terraform/main.tf
terraform {
  required_version = ">= 1.5"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }

  backend "azurerm" {
    resource_group_name  = "rg-terraform-state"
    storage_account_name = "stterraformstate"
    container_name       = "tfstate"
    key                  = "financial-analyzer.tfstate"
  }
}

provider "azurerm" {
  features {}
}

locals {
  project     = "financial-analyzer"
  environment = var.environment
  location    = var.location
  tags = {
    Project     = local.project
    Environment = local.environment
    ManagedBy   = "terraform"
  }
}

# --- Resource Group ---
resource "azurerm_resource_group" "main" {
  name     = "rg-${local.project}-${local.environment}"
  location = local.location
  tags     = local.tags
}

# --- Azure AI Search (Vector Store) ---
resource "azurerm_search_service" "main" {
  name                = "srch-${local.project}-${local.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "standard"
  replica_count       = 1
  partition_count     = 1
  tags                = local.tags
}

# --- Azure Storage (Document Uploads) ---
resource "azurerm_storage_account" "main" {
  name                     = "st${local.project}${local.environment}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"
  tags                     = local.tags
}

resource "azurerm_storage_container" "statements" {
  name                  = "bank-statements"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# --- Azure Cosmos DB (Metadata & Transaction Store) ---
resource "azurerm_cosmosdb_account" "main" {
  name                = "cosmos-${local.project}-${local.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = local.location
    failover_priority = 0
  }

  tags = local.tags
}

resource "azurerm_cosmosdb_sql_database" "main" {
  name                = "financial-db"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
}

resource "azurerm_cosmosdb_sql_container" "transactions" {
  name                = "transactions"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  partition_key_path  = "/account_number"
  throughput          = 400
}

# --- Azure Container Registry ---
resource "azurerm_container_registry" "main" {
  name                = "acr${local.project}${local.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.tags
}

# --- Azure Kubernetes Service ---
resource "azurerm_kubernetes_cluster" "main" {
  name                = "aks-${local.project}-${local.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  dns_prefix          = "${local.project}-${local.environment}"

  default_node_pool {
    name                = "default"
    node_count          = var.aks_node_count
    vm_size             = var.aks_vm_size
    enable_auto_scaling = true
    min_count           = 1
    max_count           = 5
  }

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

# --- Azure OpenAI ---
resource "azurerm_cognitive_account" "openai" {
  name                = "cog-${local.project}-${local.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = local.location
  kind                = "OpenAI"
  sku_name            = "S0"
  tags                = local.tags
}

resource "azurerm_cognitive_deployment" "gpt4o" {
  name                 = "gpt-4o"
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "gpt-4o"
    version = "2024-08-06"
  }

  scale {
    type     = "Standard"
    capacity = 20
  }
}

resource "azurerm_cognitive_deployment" "embeddings" {
  name                 = "text-embedding-3-large"
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "text-embedding-3-large"
    version = "1"
  }

  scale {
    type     = "Standard"
    capacity = 20
  }
}

# --- Key Vault (Secrets) ---
resource "azurerm_key_vault" "main" {
  name                = "kv-${local.project}-${local.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  purge_protection_enabled = true

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge",
    ]
  }

  tags = local.tags
}

data "azurerm_client_config" "current" {}