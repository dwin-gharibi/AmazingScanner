# Deploying AmazingScanner

Six ways to run the same container. The top-level `README.md` says which one to
pick; this file records what they all have to agree on, because six definitions
of "deployed" is six chances for one to drift.

```
deploy/
├── k8s/         raw manifests           kubectl apply -k deploy/k8s
├── helm/        the same, as a chart    helm upgrade --install amazingscanner deploy/helm
├── terraform/   the same, declared      cd deploy/terraform && tofu apply
├── ansible/     podman + systemd        ansible-playbook -i inventory.ini playbook.yml
└── vagrant/     a VM that runs ansible  cd deploy/vagrant && vagrant up
```

`scripts/validate_manifests.py` (run by CI, and by `task manifests`) parses
every file here, checks the Kubernetes documents have the keys they need, and
verifies container `command`/`args` are tokenised argv rather than one string
with spaces in it. Where the tool is installed it also runs `kubectl
--dry-run`, `helm lint`, `tofu validate` and `ansible-playbook
--syntax-check`. A missing tool is skipped, not failed, so the check is equally
useful on a laptop with none of them.

## The contract every path implements

Change one of these in one place and the others are wrong. They are listed here
so that is easy to notice.

| | Value | Why it is not arbitrary |
|---|---|---|
| Port | `7860` | Gradio's default; the Service, Ingress, systemd unit and Vagrant forward all name it |
| Models | `/app/models`, read-only | `DOCSCANNER_MODELS`; the exported weights, not `runs/` |
| User | uid/gid `10001`, non-root | with `readOnlyRootFilesystem`, so `/tmp` and `/app/outputs` must be writable mounts |
| Threads | `OMP_NUM_THREADS` = CPU limit | more threads than cores makes inference *slower*; the two must move together |
| Health | `GET /` | the app answers before the models load, so readiness gates the process, not a scan |
| API access | none | the ServiceAccount mounts no token, and the NetworkPolicy allows DNS egress only |

Two of those are load-bearing in a way that is easy to undo:

**`serviceAccountName: docscanner`** must be named in the pod spec. The account
sets `automountServiceAccountToken: false`, but an account nobody references
does nothing — the pod falls back to `default` and gets a token mounted.

**Thread count and CPU limit** are set in two different places (`env` and
`resources.limits`). Raising the limit without raising the threads wastes the
cores; raising the threads without the limit oversubscribes a throttled
container, which is slower than doing nothing.

## What is deliberately not in the serving stack

`k8s/job-train.yaml` holds the training Job and the nightly evaluation CronJob.
`kustomization.yaml` leaves it out on purpose: applying the serving stack should
not start a twelve-hour training job. Apply it by hand when you mean it.

```bash
kubectl apply -f deploy/k8s/job-train.yaml
kubectl logs -f job/docscanner-train
```

## Images

Everything here points at `ghcr.io/dwin-gharibi/amazingscanner:latest`, built
from the repository root `Dockerfile` (CPU) or `Dockerfile.gpu` (CUDA, for the
training Job). With kustomize the tag has one home:

```bash
kubectl kustomize deploy/k8s | grep image:          # what you would apply
cd deploy/k8s && kustomize edit set image \
  ghcr.io/dwin-gharibi/amazingscanner=:v1.2.0       # pin a release
```
