variable "kubeconfig_path" {
  description = "Path to the kubeconfig used to reach the cluster."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "kubeconfig context to use. Empty means whatever is current."
  type        = string
  default     = ""
}

variable "namespace" {
  description = "Namespace to deploy into."
  type        = string
  default     = "amazingscanner"
}

variable "create_namespace" {
  description = "Create the namespace, or expect it to exist already."
  type        = bool
  default     = true
}

variable "release_name" {
  description = "Name for the deployment, service and autoscaler."
  type        = string
  default     = "amazingscanner"
}

variable "image" {
  description = "Container image to run."
  type        = string
  default     = "ghcr.io/dwin-gharibi/amazingscanner:latest"
}

variable "image_pull_policy" {
  type    = string
  default = "IfNotPresent"

  validation {
    condition     = contains(["Always", "IfNotPresent", "Never"], var.image_pull_policy)
    error_message = "image_pull_policy must be Always, IfNotPresent or Never."
  }
}

variable "replicas" {
  description = "Baseline replica count, and the autoscaler's floor."
  type        = number
  default     = 2
}

variable "threads" {
  description = <<-EOT
    Torch/OpenMP thread count per pod. Keep this at or below the CPU limit:
    Torch otherwise spawns one thread per host core and then contends with the
    cgroup quota, which costs far more latency than the extra threads win.
  EOT
  type        = number
  default     = 4
}

variable "cpu_request" {
  type    = string
  default = "1"
}

variable "cpu_limit" {
  type    = string
  default = "4"
}

variable "memory_request" {
  description = "Torch plus the fp16 weights sit around 900 MB idle."
  type        = string
  default     = "1Gi"
}

variable "memory_limit" {
  description = "Peak measured on a 12 MP photo end to end was 1.16 GB."
  type        = string
  default     = "3Gi"
}

variable "service_type" {
  type    = string
  default = "ClusterIP"

  validation {
    condition     = contains(["ClusterIP", "NodePort", "LoadBalancer"], var.service_type)
    error_message = "service_type must be ClusterIP, NodePort or LoadBalancer."
  }
}

variable "service_port" {
  type    = number
  default = 80
}

variable "enable_autoscaling" {
  type    = bool
  default = true
}

variable "max_replicas" {
  type    = number
  default = 8
}

variable "target_cpu_utilization" {
  description = "Scale out above this CPU utilisation. Inference is CPU-bound."
  type        = number
  default     = 70
}
