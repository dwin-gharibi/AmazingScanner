# Contributing

Thank you for looking. This is a university project with a written brief, and
that brief constrains what a change is allowed to be — a patch can be correct,
well-tested, and still not belong here. The rules are listed first for that
reason.

## The rules the brief imposes

These are not style preferences. A change that breaks one of them cannot be
merged, however much it improves a number.

**Geometric transforms use OpenCV, and nothing else.** No `kornia`, no
`albumentations`, no `imgaug`, no `torchvision.transforms` doing a warp. The
perspective transforms, the rotations, the resizes and every degradation are
written against `cv2`. The one deliberate exception is the *differentiable*
warp used by the bonus §7 fine-tune, which is built from a hand-written DLT
plus `torch.nn.functional.grid_sample` — still not a third-party transform
library, and it has to be differentiable, which `cv2` is not.

**No pre-designed architectures, no pre-trained weights.** Every network in
`src/docscanner/models/` is assembled from primitive `torch.nn` modules. Not a
ResNet, not a U-Net imported from anywhere, no `timm`, no checkpoint downloaded
from anywhere. If you want a better encoder, write it.

**Dropout belongs to Section 6 only.** The §3 enhancement network and the §5
corner detectors are trained at `dropout=0.0` by design, because Section 6 is
the experiment that asks what dropout does. Adding it to a baseline "because it
helps" deletes the experiment.

**Real photographs are test data.** The Roboflow set of real photos is never
trained on — not for a few epochs, not for fine-tuning, not "just the
backgrounds". The 50 course scans are the training ground truth. This is
checked: the splits are built by *source scan* and the test suite asserts zero
overlap.

**Two tasks, two datasets.** The enhancement network and the corner detectors
get their own generated datasets with their own augmentation policies
(`DegradationConfig.for_enhancement()` and `.for_corners()`). They are not the
same data with a different head on top.

## Getting set up

```bash
task setup          # virtualenv, CPU torch, the package, the dev tools
task setup:hooks    # pre-commit
task train:smoke    # ~10 minutes: proves every step runs before you spend hours
```

No `task` binary? `make setup` does the same thing, and every task maps to a
plain `amazingscanner` command — see [docs/CLI.md](docs/CLI.md).

## Before you open a pull request

```bash
task verify         # lint + manifests + tests, which is what CI runs
```

If your change touches the data pipeline, the networks, the losses or the
evaluation code, it moves the published numbers, and the pull request needs the
before/after:

```bash
amazingscanner eval --all
```

Then paste the two tables. "It looks better" is not evidence; the whole point of
the evaluation harness is that nobody has to take that on trust.

## Comparisons must be at matched budgets

Every comparison in the report — Approach A against Approach B, each model
against its dropout variant, each loss against the others — is run at an
identical epoch count, step count and batch size. A comparison at unmatched
budgets measures the budget.

`amazingscanner all` preserves this automatically; the presets in
`src/docscanner/all_in_one.py` set both arms from the same numbers. Overriding
one arm by hand is how it gets broken quietly.

## If the numbers moved, regenerate everything

Every table, chart, figure, demo GIF and README number is *derived* from the
checkpoints. One command rebuilds all of it in the right order:

```bash
amazingscanner refresh
```

Without it the repository ends up describing a model it no longer ships, which
is worse than describing no model at all.

## Layout

| Where | What lives there |
|---|---|
| `src/docscanner/data/` | corpora, degradations, the synthetic generator, splits |
| `src/docscanner/models/` | the networks, the losses, the differentiable warp |
| `src/docscanner/engine/` | training loops, checkpointing, export |
| `src/docscanner/pipeline/` | inference: corners → rectify → enhance → output |
| `src/docscanner/eval/` | every measurement, chart, figure and report |
| `src/docscanner/app/` | the two Gradio interfaces |
| `tests/` | the suite; `test_end_to_end.py` is the integration one |
| `deploy/` | k8s, Helm, Terraform/OpenTofu, Ansible, Vagrant |
| `scripts/` | repository tooling that is not part of the package |

`model.py`, `train.py` and `evaluate.py` at the root are thin re-export shims.
They exist because the brief names those files; the implementations are in the
package and that is where changes go.

## Code style

`ruff check` is enforced; `ruff format` deliberately is not. This codebase pairs
statements that are one thought (`h, w = img.shape[:2]; cx, cy = w / 2, h / 2`)
and aligns the tables the report generators emit — a formatter would undo both
for no reader's benefit. Match the file you are editing.

Comments should say why, not what. The interesting comments in this repository
explain a decision that was measured — why the highlight knee sits at 0.95 and
not 0.86, why the TTA takes a median rather than a mean — because those are the
lines that stop someone re-litigating a settled question six months later.
