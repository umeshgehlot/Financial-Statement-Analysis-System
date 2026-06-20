# infra/terraform/variables.tf
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "staging"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus2"
}

variable "aks_node_count" {
  description = "Number of AKS nodes"
  type        = number
  default     = 2
}

variable "aks_vm_size" {
  description = "AKS VM size"
  type        = string
  default     = "Standard_D4s_v3"
}