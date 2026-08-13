# Command-line reference

Everything the project does is reachable from one entry point:

```bash
amazingscanner                       # list every command
amazingscanner <command> --help      # the real options for that command
```

## The short version: train on a GPU, bring it home

If you only read one section, read this one.

```bash
git clone -b main https://github.com/dwin-gharibi/AmazingScanner
cd AmazingScanner
pip install -r requirements.txt && pip install -e .

amazingscanner data --all                                   # datasets + frozen splits
amazingscanner train-corners --approach heatmap --device cuda --amp \
    --epochs 60 --steps 300 --batch 32 --minutes 75          # ~15 min on a T4
amazingscanner train-enhance --device cuda --amp \
    --epochs 60 --steps 400 --batch 24 --minutes 90
amazingscanner refresh                                       # everything else
```

**`refresh` is the whole point.** Training writes checkpoints; every number,
table, chart, figure, demo GIF and report in this repository is *derived* from
them. One command regenerates all of it, in the right order, so nothing can be
left describing a model you no longer ship:

```
export → evaluate → charts → figures → stages → report → tests
```

| Flag | What it does |
|---|---|
| `--quick` | Skip the GIFs and the test suite. Good for a fast look. |
| `--only export charts report` | Run just those steps. |
| `--skip tests` | Everything except the suite. |
| `--limit 8` | Cap images per evaluation set — a smoke run, not a result. |

A failing step does not abort the rest — a missing GIF encoder should not cost
you the tables — but it does make the run exit non-zero, so this is safe in CI.

---

The CLI is a **router**, not a second implementation: each command forwards its
remaining arguments to the module that already owns that job, so `--help` always
shows that module's actual flags and the two can never drift apart. Every
command is equally reachable as `python -m docscanner.<module>` if you prefer.

---

## Demo runbook

For a live demonstration on a machine that has the repository and the committed
weights. Nothing here needs the corpora downloaded or a model trained: the
weights are in `models/`, the real photographs and the 18-photo unseen pack are
in `data/`. Every command below was run end to end on a clean checkout before
this was written.

```bash
# 0. one-time, ~2 min
pip install -e ".[all]" && sudo apt-get install -y tesseract-ocr

# 1. the headline: a real photograph becomes a scan          (~2 s)
amazingscanner scan data/real/own/photos/img1_*.jpg -o /tmp/demo
#    -> /tmp/demo/<name>_scan.png, a _steps.jpg strip, and scan_report.json

# 2. hand it something it has never seen, and a whole folder at once
amazingscanner scan data/test_pack/photos -o /tmp/demo_pack --pdf /tmp/scans.pdf

# 3. the two mandatory pipelines on their own
amazingscanner corners data/real/own/photos/img3_*.jpg -o /tmp/demo   # raw photo in
amazingscanner enhance data/test_pack/ground_truth/targets/pack_00_clean.png -o /tmp/demo

# 4. the interactive interface — this is the one to present from
amazingscanner app                      # http://localhost:7860
```

If a grader hands over a **new photograph**, step 1 is the whole answer; drop
the file in and read the timings off the app's Auto Scan tab. If they ask for
something the models have provably never seen, mint it on the spot — the seed
space above 7000 is reserved for exactly this:

```bash
python scripts/make_test_pack.py --out /tmp/fresh --n 12 --seed 7300
amazingscanner scan /tmp/fresh/photos -o /tmp/fresh_scans
python scripts/score_test_pack.py --pack /tmp/fresh          # scored against its own truth
```

Two things worth knowing before demonstrating rather than during:

* **First run of any command loads the weights** (a second or two). Run one
  scan before the audience arrives so the demo itself is warm.
* **Page size drives cost, not page difficulty.** A 12 MP photograph takes a
  few seconds end to end; `--max-side 1200` or `--no-tta` trades a little
  accuracy for a visibly faster response if the room is watching a progress bar.

---

## Scanning

The three pipelines the brief asks for. Each writes the result, a step-by-step
comparison figure, and a JSON record of timings and coordinates.

```bash
# Bonus §7 — raw photo in, finished scan out, no human input
amazingscanner scan photo.jpg -o outputs/
amazingscanner scan photos/ -o outputs/ --pdf scans.pdf      # a folder into one PDF
amazingscanner scan photo.jpg --mode whiteboard              # colour|gray|bw|whiteboard|raw
amazingscanner scan photo.jpg --no-tta                       # ~3x faster detection
amazingscanner scan photo.jpg --preserve-tone 1              # keep a cover's colour

# §3.4 — the enhancement pipeline; input is an ALREADY RECTIFIED page
amazingscanner enhance page.jpg -o outputs/

# §5.1 — the corner pipeline; input is a RAW PHOTO
amazingscanner corners photo.jpg -o outputs/
```

| Flag | Applies to | What it does |
|---|---|---|
| `--mode` | scan, enhance | Output style. `raw` is the network's own output, unstyled. |
| `--max-side` | scan, enhance | Caps the working resolution. A cost/quality dial — the network is fully convolutional, so this is not a correctness setting. |
| `--no-tta` | scan, corners | One view instead of four quarter-turns. Roughly 3× faster detection; costs about 5% of mean accuracy on real photographs (2.01% → 2.12% of the diagonal) while very slightly *reducing* gross misses. Worth it when latency matters. |
| `--refine` | scan, corners | Turns **on** classical sub-pixel edge refinement, which is off by default. It sharpens clean synthetic borders but costs ~0.3% of the diagonal on real photographs, where the page edge competes with bindings, desk rims and creases. Opt in for scans of loose sheets on plain backgrounds; leave off otherwise. (`--no-refine` is still accepted, and is a no-op.) |
| `--preserve-tone` | scan, enhance | `0..1`, default `0`. How much of a non-paper page's own colour and tone survives enhancement. At `0` every page is whitened, book covers included — which is what the network was trained to do, since all 50 targets average 249/255 luma at 3.3 saturation. At `1` a document page is still whitened (it already looks like paper) while a cover keeps its colour. Highlight compression runs either way: clipping is not a look, it is 10–14.5% of the page destroyed. |
| `--pdf` | scan | Collects every page into one PDF. |

---

## Interfaces

```bash
amazingscanner app                     # http://localhost:7860
amazingscanner label                   # http://localhost:7861 — corner annotation
```

---

## Data

```bash
amazingscanner fetch                   # download the source corpora (~1.8 GB)
amazingscanner fetch --check           # what is present? downloads nothing
amazingscanner fetch --no-midv         # skip the 795 MB evaluation-only set

amazingscanner data --all              # fetch, build splits, freeze eval sets
amazingscanner data --all --no-fetch   # assume data/raw is already populated
amazingscanner data --own              # convert the Roboflow export of your own photos
amazingscanner check-labels            # validate annotations + contact sheet

# Keep the real test set in step with its Roboflow project
export ROBOFLOW_API_KEY=...
amazingscanner roboflow --check        # fail if the checked-in labels have drifted
amazingscanner roboflow --sync         # re-download and convert in place
```

`roboflow --check` writes nothing: it downloads to a scratch directory, converts
there, and compares against the manifest in the repository. That is the form CI
wants — it should fail on drift, not silently rewrite annotations mid-build.

### Where the data comes from

| Corpus | Size | How it arrives | Role |
|---|---:|---|---|
| **course scans** (50) | 19 MB | **in the repository** | **trained on** — the graded distribution |
| DocLayNet-small (804) | 381 MB | `fetch` — one zip | trained on, auxiliary — layout and typeface variety |
| DTD textures (5,640) | 625 MB | `fetch` — one tarball | backgrounds — the surfaces a page lands on |
| MIDV-500 | 795 MB | `fetch` — 3 shards | **evaluation only** |
| your Roboflow photos | in repo | `data --own` | **evaluation only** |

Each corpus is fetched as **one archive**, not file by file. That distinction is
the difference between minutes and hours: DTD's Hugging Face mirror stores it as
5,640 loose JPEGs, and pulling those one at a time ran at about 0.6 files per
second — over two hours. The upstream tarball supports range requests, so it is
fetched in parallel chunks instead: **1m42s** including extraction. Raise
`--jobs` for more parallelism; archives are deleted once unpacked.

The course scans are committed because they are irreplaceable: they are the
ground truth every training pair is built from, and unlike the others there is
nowhere to download them from. Everything else is public and fetched on demand,
which is why `data/raw/` is otherwise git-ignored.

**Real photographs are never trained on.** Not MIDV-500, not the annotated
Roboflow set. The splits are built by *source scan* and the test suite asserts
zero overlap between them.

A missing corpus stops the run where it is discovered and names the command
that fixes it. It used to report `course train=0 val=0 test=0` and then fail
several modules later with `ValueError: high <= 0`, which said nothing about
the actual problem.

---

## Training

```bash
# The enhancement network (§3.2)
amazingscanner train-enhance --epochs 22 --steps 170 --lr 2.5e-3 --minutes 120

# Either corner detector (§5) — the approach is the experiment
amazingscanner train-corners --approach heatmap    --epochs 20 --steps 130
amazingscanner train-corners --approach regression --epochs 20 --steps 130

# Section 6: the same models with dropout, at a matched budget
amazingscanner train-corners --approach regression --dropout 0.3  --name reg_drop
amazingscanner train-enhance --dropout 0.15 --name enhance_dropout

# Bonus §7 option: fine-tune the whole chain through the differentiable warp
amazingscanner finetune --minutes 20

# Pick the best run per model and write fp16 deployment weights
amazingscanner export --out models
```

**Pinning a model.** `export` picks the best run automatically, but a score
cannot always answer the question. When the frozen benchmark moves between two
runs their numbers stop being comparable, and "which output does a person prefer
on a real photograph" is not a number at all. Writing `models/<name>.pt.pin`
next to a checkpoint holds it: export reports `[pin]` and moves on, whatever the
scores say. The file's first line is printed as the reason, so a pin has to
justify itself. `--force` overrides it; deleting the pin releases it.
`models/enhance.pt.pin` is one — the 24-epoch enhancement network was judged to
read better on the real photographs than a later, higher-scoring retrain.

| Flag | What it does |
|---|---|
| `--epochs`, `--steps` | The schedule. `steps` is samples-per-epoch, which is a free parameter here because the generator is infinite. |
| `--minutes` | Wall-clock budget. Training stops cleanly at the limit and keeps the best checkpoint. |
| `--device` | `auto` (default), `cpu`, `cuda`, `cuda:1`, `mps`. An explicit `cuda` with no CUDA present **fails loudly** rather than silently spending hours on CPU. |
| `--threads` | Torch thread count. Keep it at or below your CPU allocation. |
| `--dropout` | Section 6 only. The §3 and §5 baselines are trained at `0.0` by design. |
| `--loss` | `combined` (Charbonnier + MS-SSIM + Sobel), `l1`, `mse`. |
| `--resume` | Continue from a checkpoint. |
| `--seg-weight` | Heatmap detector only: weight on the page-interior mask head (default `0.5`; `0` removes the head). The head is what lets corners be decoded as a set that agrees on one page rather than four independent argmaxes. Its validation IoU is recorded as `val_mask_iou`, and the pipeline **uses the mask only above 0.80** — below that it decodes exactly as it would without one. |
| `--no-page-aug` | The control arm for the eleven page, boundary and corner augmentations (see `PAGE_AUGMENTATION` in `engine/train_corners.py`). Everything else is identical, so the comparison measures the augmentation rather than the budget. |

Two things about the corner detector that are easy to miss:

* Changing `--seg-weight` or `--no-page-aug` changes the *training* distribution,
  and the frozen corner sets are built with the same policy — so rebuild them
  (`amazingscanner data --splits --frozen`) or validation scores a distribution
  the model no longer trains on. Corner numbers then stop being comparable with
  earlier runs; enhancement numbers stay comparable, and a test enforces that.
* A run whose `val_mask_iou` lands under 0.80 is not broken. The mask is simply
  not used, and the corner numbers will look like a run without the head —
  which is the intended behaviour, because a mask that weak measured *worse*
  than having none at all.

Matched budgets matter: every comparison in the report (A vs B, control vs
dropout, one loss against another) is run at an identical step count, so the
difference measured is the change under test and nothing else.

**On a GPU:**

```bash
docker build -f Dockerfile.gpu -t amazingscanner:gpu .
docker run --gpus all -v "$PWD/runs:/app/runs" amazingscanner:gpu \
    amazingscanner train-corners --approach heatmap --device cuda --epochs 40
```

---

## Testing and measurement

```bash
pytest tests/ -q                       # the full suite
pytest tests/ -q -k corner             # one area
pytest tests/test_end_to_end.py -q     # the integration suite

amazingscanner eval --all              # every benchmark table -> outputs/report/
amazingscanner eval --corners          # just Section 5
amazingscanner eval --end-to-end       # just the bonus chain

amazingscanner stages                  # per-stage inputs, outputs and accuracy
amazingscanner stages --image photo.jpg --limit 1
amazingscanner charts                  # one chart per measured question
amazingscanner figures --all           # figures, galleries and demo GIFs
amazingscanner report                  # regenerate docs/REPORT.md
```

`stages` is the one to reach for when a result looks wrong but the aggregate
numbers do not say why: it writes **every stage of every photograph as its own
image**, with the metric that stage is responsible for, so a bad page can be
attributed to detection, rectification or enhancement rather than guessed at.

```
outputs/stages/<photo>/1_input.jpg      2_detected.jpg   3_rectified.jpg
                      4_enhanced.jpg    5_final.jpg      6_reference.jpg
                      strip.jpg         stages.json
outputs/stages/STAGES.md                aggregate over the whole set
docs/assets/stages/<photo>.jpg          the strip, published where a clone sees it
docs/assets/stages/README.md            the index, worst corner error first
```

---

## The order things run in

```bash
make setup                             # virtualenv + CPU PyTorch
amazingscanner data --all              # corpora, splits, frozen eval sets
make train                             # every model (~6 h on one H100; far longer on CPU)
amazingscanner export                  # fp16 weights into models/
amazingscanner eval --all              # measure
amazingscanner charts && amazingscanner figures --all && amazingscanner report
pytest tests/ -q
```

`amazingscanner refresh` runs everything after training in that order, which is
the sequence the documentation is regenerated from.

---

## Repository checks

Three things in this repository are not reachable from `amazingscanner`,
because they check the repository rather than the scanner.

```bash
python scripts/validate_manifests.py     # every deployment file
python scripts/check_regression.py       # has accuracy dropped?
ruff check src tests scripts *.py        # lint
```

**`validate_manifests.py`** parses every YAML file, confirms each Kubernetes
document has the keys it needs, and checks that container `command`/`args`
entries are lists of *individual* argv tokens. That last one exists for a
specific reason: `command: ["bash", "amazingscanner all"]` is valid YAML, passes
a schema check, and cannot execute — `exec` does not split on spaces, so the
kernel looks for a binary whose filename contains one. Nothing else in CI ever
runs these files. With `kubectl`, `helm`, `tofu` or `ansible-playbook`
installed it also renders and validates properly; missing tools are skipped
rather than failed.

**`check_regression.py`** is the check the test suite cannot be. `pytest` asks
whether the code runs and the geometry is right, and stays green through
changes that leave the scanner much worse at finding a page. This re-reads
`outputs/report/*.json` and fails when a number has moved past its ceiling.

```bash
python scripts/check_regression.py                    # against the bounds
python scripts/check_regression.py --markdown out.md  # a table to paste
python scripts/check_regression.py --update           # re-baseline, on purpose
```

The bounds in `scripts/regression_bounds.json` are ceilings, not targets, with
a 15% tolerance — wide enough that a BLAS point release reordering
floating-point work does not cry wolf, narrow enough that a real regression
cannot hide. An improvement never fails; it prints the headroom and `--update`
banks it. Every bound names the set it applies to, because the same detector
measures around a pixel on synthetic pages and tens of pixels on photographs of
open books, and quoting one as the other is how these projects mislead.

---

## Make and Task

Two runners, the same commands underneath. `make` is already installed
everywhere; `task` ([taskfile.dev](https://taskfile.dev)) understands
dependencies, guards and pass-through arguments, which `make` was not designed
for. Neither is a second implementation of anything.

```bash
task                 # every task, with a description
task train           # the one command
task verify          # lint + manifests + tests, which is what CI runs
task train PRESET=smoke DEVICE=cuda
task train -- --skip-ablations      # anything after -- reaches the CLI
```

| Task | Make | Equivalent command |
|---|---|---|
| `task setup` | `make setup` | virtualenv, CPU PyTorch, the package |
| `task data` | `make data` | `amazingscanner data --all` |
| `task train` | `make train` | `amazingscanner all` |
| `task train:smoke` | — | the same at a toy budget, ~10 min |
| `task train:plan` | — | print the plan without running it |
| `task eval` | `make eval` | `amazingscanner eval --all` |
| `task regression` | — | `python scripts/check_regression.py` |
| `task manifests` | — | `python scripts/validate_manifests.py` |
| `task refresh` | — | `amazingscanner refresh` |
| `task figures` | `make figures` | `amazingscanner figures --all` |
| `task export` | `make export-models` | `amazingscanner export --out models` |
| `task app` / `task label` | `make app` / `make label` | the two interfaces |
| `task scan -- photo.jpg` | `make scan IMG=photo.jpg` | `amazingscanner scan photo.jpg` |
| `task test` | `make test` | `pytest tests/ -q` |
| `task lint` | — | `ruff check` |
| `task docker` / `task docker:gpu` | `make docker` | build the containers |
| `task helm` / `task k8s` / `task tofu` | `make k8s` | apply the deployments |
| `task bundle` | — | zip the weights and artefacts for download |
