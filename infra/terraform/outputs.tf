# infra/terraform/outputs.tf
output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "acr_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "search_service_endpoint" {
  value = "https://${azurerm_search_service.main.name}.search.windows.net"
}

output "cosmos_endpoint" {
  value = azurerm_cosmosdb_account.main.endpoint
}

output "openai_endpoint" {
  value = azurerm_cognitive_account.openai.endpoint
}

output "storage_account_name" {
  value = azurerm_storage_account.main.name
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}