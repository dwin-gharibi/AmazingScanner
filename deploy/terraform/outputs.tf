output "namespace" {
  description = "Namespace the workload landed in."
  value       = var.namespace
}

output "service_name" {
  description = "In-cluster DNS name of the service."
  value       = "${kubernetes_service.this.metadata[0].name}.${var.namespace}.svc.cluster.local"
}

output "port_forward" {
  description = "Ready-made command to reach the app from a laptop."
  value       = "kubectl -n ${var.namespace} port-forward svc/${kubernetes_service.this.metadata[0].name} 7860:${var.service_port}"
}

output "replicas" {
  description = "Replica floor, and the ceiling if autoscaling is on."
  value = {
    min = var.replicas
    max = var.enable_autoscaling ? var.max_replicas : var.replicas
  }
}
