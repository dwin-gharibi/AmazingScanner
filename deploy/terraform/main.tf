terraform {
  required_version = ">= 1.5.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 3.2"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context != "" ? var.kube_context : null
}

locals {
  labels = {
    "app.kubernetes.io/name"       = "amazingscanner"
    "app.kubernetes.io/instance"   = var.release_name
    "app.kubernetes.io/component"  = "web"
    "app.kubernetes.io/managed-by" = "terraform"
  }
}

resource "kubernetes_namespace" "this" {
  count = var.create_namespace ? 1 : 0

  metadata {
    name   = var.namespace
    labels = local.labels
  }
}

resource "kubernetes_service_account" "this" {
  metadata {
    name      = var.release_name
    namespace = var.namespace
    labels    = local.labels
  }

  automount_service_account_token = false

  depends_on = [kubernetes_namespace.this]
}

resource "kubernetes_deployment" "this" {
  metadata {
    name      = var.release_name
    namespace = var.namespace
    labels    = local.labels
  }

  depends_on = [kubernetes_namespace.this]

  spec {
    replicas = var.replicas

    selector {
      match_labels = {
        "app.kubernetes.io/name"     = local.labels["app.kubernetes.io/name"]
        "app.kubernetes.io/instance" = var.release_name
      }
    }

    strategy {
      type = "RollingUpdate"
      rolling_update {
        max_surge       = 1
        max_unavailable = 0
      }
    }

    template {
      metadata {
        labels = local.labels
      }

      spec {
        service_account_name = kubernetes_service_account.this.metadata[0].name

        security_context {
          run_as_non_root = true
          run_as_user     = 10001
          fs_group        = 10001
        }

        container {
          name              = "app"
          image             = var.image
          image_pull_policy = var.image_pull_policy

          port {
            name           = "http"
            container_port = 7860
          }

          env {
            name  = "OMP_NUM_THREADS"
            value = tostring(var.threads)
          }
          env {
            name  = "GRADIO_SERVER_NAME"
            value = "0.0.0.0"
          }

          resources {
            requests = {
              cpu    = var.cpu_request
              memory = var.memory_request
            }
            limits = {
              cpu    = var.cpu_limit
              memory = var.memory_limit
            }
          }

          startup_probe {
            http_get {
              path = "/"
              port = "http"
            }
            failure_threshold = 30
            period_seconds    = 5
          }

          liveness_probe {
            http_get {
              path = "/"
              port = "http"
            }
            period_seconds        = 20
            timeout_seconds       = 5
            failure_threshold     = 3
            initial_delay_seconds = 10
          }

          readiness_probe {
            http_get {
              path = "/"
              port = "http"
            }
            period_seconds  = 10
            timeout_seconds = 3
          }

          security_context {
            allow_privilege_escalation = false
            read_only_root_filesystem = false
            capabilities {
              drop = ["ALL"]
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "this" {
  metadata {
    name      = var.release_name
    namespace = var.namespace
    labels    = local.labels
  }

  depends_on = [kubernetes_namespace.this]

  spec {
    type = var.service_type

    selector = {
      "app.kubernetes.io/name"     = local.labels["app.kubernetes.io/name"]
      "app.kubernetes.io/instance" = var.release_name
    }

    port {
      name        = "http"
      port        = var.service_port
      target_port = "http"
    }
  }
}

resource "kubernetes_horizontal_pod_autoscaler_v2" "this" {
  count = var.enable_autoscaling ? 1 : 0

  metadata {
    name      = var.release_name
    namespace = var.namespace
    labels    = local.labels
  }

  spec {
    min_replicas = var.replicas
    max_replicas = var.max_replicas

    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.this.metadata[0].name
    }

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = var.target_cpu_utilization
        }
      }
    }
  }
}
