# Security

## Reporting

Open a [private security advisory][advisory] rather than a public issue. If you
would rather not use GitHub for it, anything actionable is better than nothing —
say what you found and how you found it.

[advisory]: https://github.com/dwin-gharibi/AmazingScanner/security/advisories/new

## What this project is, in security terms

AmazingScanner takes an image from an untrusted source and runs it through
OpenCV, PyTorch and — if OCR is enabled — Tesseract. That is the whole threat
surface, and it is mostly *other people's* parsers. A malformed JPEG that
crashes `cv2.imread` is an OpenCV bug; report it upstream. A malformed JPEG that
gets past `cv2.imread` and causes this code to write outside its output
directory is our bug, and worth an advisory.

Specifically in scope:

- Path traversal through an image filename, an output path, or a Roboflow
  export's contents.
- Anything in the Gradio interfaces that reads or writes outside the paths it
  was pointed at.
- `torch.load` reachability from untrusted input. Checkpoint loading is a
  well-known code-execution vector; the loaders here are meant to be reached
  only with local paths the operator chose.
- A container that runs as root, mounts more than it needs, or ships a
  vulnerable base layer. The images run as UID 10001 with a read-only root
  filesystem, and the release workflow scans them — a regression in any of that
  is in scope.

Not in scope: a model producing a bad scan. That is an accuracy bug, and
[an issue][issues] is the right home for it.

[issues]: https://github.com/dwin-gharibi/AmazingScanner/issues/new/choose

## Running it where other people can reach it

The Gradio app has no authentication and is not built to have any. It binds
`0.0.0.0` inside a container because that is the only way a container can serve
anything — that is not the same as being safe to expose.

If you put it on a network:

- Terminate TLS and authenticate in front of it. The Helm chart's ingress is
  where that belongs.
- Keep the request body limit realistic. The chart sets 32 MB, which fits a
  phone photograph; unbounded uploads are a denial-of-service vector against a
  process that decodes and resizes whatever it is given.
- Give it a CPU limit. Inference on a large image is bounded work, but a queue
  of them is not.
- Do not mount the training data or the runs directory into a serving pod. It
  needs `models/` read-only and nothing else.

## Weights

`models/*.pt` are built by this repository's own training runs and exported by
`amazingscanner export`. Nothing here downloads pre-trained weights — the brief
forbids it, and it happens to remove the supply-chain question that comes with
loading a pickle from the internet.

If you train your own and share them, the same caution applies to whoever loads
yours: a PyTorch checkpoint is a pickle, and unpickling is code execution.

## Supported versions

The `main` branch. This is a university project, not a maintained product;
there is no backport lane.
