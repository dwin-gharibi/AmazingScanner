from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_validate_manifests_passes() -> None:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_manifests.py")],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_validator_catches_untokenised_argv(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from validate_manifests import _argv_tokens_are_split
    finally:
        sys.path.pop(0)

    broken = {"name": "train", "command": ["bash", "amazingscanner all", "/data/runs"]}
    assert _argv_tokens_are_split(broken, "job.yaml"), "should have flagged the bundled token"

    fixed = {"name": "train", "command": ["amazingscanner", "all", "--out", "/data/runs"]}
    assert not _argv_tokens_are_split(fixed, "job.yaml")

    shell = {"name": "x", "command": ["bash", "-lc", "set -e && python -m docscanner.eval"]}
    assert not _argv_tokens_are_split(shell, "job.yaml")


def test_every_kubernetes_container_command_is_a_real_cli_command() -> None:
    from docscanner.cli import COMMANDS

    invoked: list[tuple[str, str]] = []
    for path in (ROOT / "deploy" / "k8s").glob("*.yaml"):
        for doc in yaml.safe_load_all(path.read_text()):
            for tok in re.findall(r'"amazingscanner",\s*"([a-z-]+)"', json.dumps(doc or {})):
                invoked.append((path.name, tok))

    assert invoked, "expected at least one amazingscanner invocation in deploy/k8s"
    unknown = [f"{f}: {c}" for f, c in invoked if c not in COMMANDS]
    assert not unknown, f"manifests invoke commands that do not exist: {unknown}"


def test_taskfile_only_invokes_real_modules() -> None:
    import importlib.util

    spec = yaml.safe_load((ROOT / "Taskfile.yml").read_text())
    text = json.dumps(spec["tasks"])
    modules = set(re.findall(r"-m (docscanner[.\w]+)", text))

    assert modules, "expected the Taskfile to invoke docscanner modules"
    unimportable = [m for m in modules if importlib.util.find_spec(m) is None]
    assert not unimportable, f"Taskfile invokes modules that do not exist: {unimportable}"


def _write_report(directory: Path, mce: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "corners.json").write_text(json.dumps({"rows": [{
        "Model": "B: heatmap", "Set": "real photos (own)",
        "edge refine": "no", "TTA": "yes", "n": 24,
        "MCE (px)": mce, "median (px)": mce * 0.7, "quad IoU": 0.75,
    }]}))

def test_regression_bounds_file_is_current() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_regression as cr
    finally:
        sys.path.pop(0)

    if not cr.BOUNDS_FILE.exists():
        pytest.skip("no baselines recorded yet")
    report = ROOT / "outputs" / "report"
    if not (report / "corners.json").exists():
        pytest.skip("no evaluation output to compare against")

    baseline = json.loads(cr.BOUNDS_FILE.read_text())["baseline"]
    measured = cr.collect(report)
    unwatched = sorted(set(measured) - set(baseline))
    assert not unwatched, (
        "these measurements have no recorded bound, so a regression in them "
        f"would go unnoticed: {unwatched}. Run `task regression:update`.")


def test_course_scans_are_committed() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "data/raw/course_scans"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert len(tracked) >= 50, (
        f"only {len(tracked)} course scans are tracked by git — a fresh clone "
        f"cannot train. Check the .gitignore rules for data/raw/.")


def test_gitignore_still_excludes_the_downloadable_corpora() -> None:
    for path in ("data/raw/dtd/images/x.jpg",
                 "data/raw/small_dataset/train/images/x.png",
                 "data/raw/midv500/data/x.parquet"):
        proc = subprocess.run(["git", "check-ignore", "-q", path],
                              cwd=ROOT, capture_output=True)
        assert proc.returncode == 0, f"{path} is NOT ignored — it should be"


def test_empty_corpus_raises_something_actionable(tmp_path: Path,
                                                  monkeypatch) -> None:
    from docscanner.data import prepare
    from docscanner.data.fetch import CorpusMissingError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw" / "course_scans").mkdir(parents=True)
    monkeypatch.setattr(prepare, "COURSE_DIR", tmp_path / "data/raw/course_scans")
    monkeypatch.setattr(prepare, "DOCLAYNET_DIR", tmp_path / "data/raw/small_dataset")

    with pytest.raises(CorpusMissingError) as excinfo:
        prepare.build_splits()
    message = str(excinfo.value)
    assert "amazingscanner" in message, "the error must name the command that fixes it"
    assert "course scans" in message.lower()


def test_fetch_declares_every_corpus_prepare_reads() -> None:
    from docscanner.data import prepare
    from docscanner.data.fetch import CORPORA, COURSE_DIR

    fetchable = {str(spec["verify"]) for spec in CORPORA.values()}
    fetchable.add(str(COURSE_DIR))

    needed = {str(prepare.COURSE_DIR), str(prepare.DOCLAYNET_DIR),
              str(prepare.DTD_DIR), str(prepare.MIDV_DIR)}
    unsuppliable = sorted(needed - fetchable)
    assert not unsuppliable, (
        f"prepare reads these but fetch cannot supply them: {unsuppliable}")


def test_every_pick_best_call_passes_a_name() -> None:
    import ast

    offenders: list[str] = []
    for path in (ROOT / "src" / "docscanner").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "pick_best":
                continue
            has_name = (any(k.arg == "name" for k in node.keywords)
                        or len(node.args) >= 2)
            if not has_name:
                offenders.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} calls pick_best "
                    f"without name=")

    assert not offenders, (
        "these will select a different checkpoint than `export` ships:\n  "
        + "\n  ".join(offenders))



def test_export_blocks_a_smoke_run_but_ships_a_real_retrain(tmp_path: Path) -> None:
    import json

    import torch

    from docscanner.engine.export_models import _would_regress

    def run(name: str, mce: float, epochs: int) -> Path:
        d = tmp_path / name
        d.mkdir()
        torch.save({"model": {}, "extra": {"val_mce_px": mce, "epoch": epochs}},
                   d / "best.pt")
        (d / "history.json").write_text(json.dumps([{"val_mce_px": mce}] * epochs))
        return d / "best.pt"

    shipped = tmp_path / "exported.pt"
    torch.save({"model": {}, "extra": {"val_mce_px": 10.2, "epoch": 9}}, shipped)
    (tmp_path / "history.json").write_text(json.dumps([{"val_mce_px": 10.2}] * 9))

    assert _would_regress(run("smoke", 63.4, 2), shipped), (
        "a two-epoch run scoring 63 px must not overwrite a nine-epoch 10 px model")
    assert _would_regress(run("retrain", 15.0, 40), shipped) is None, (
        "a 40-epoch retrain must ship even when the frozen-set score is worse — "
        "that set cannot measure what the retrain improves")
    assert _would_regress(run("better", 8.0, 40), shipped) is None
    assert _would_regress(run("first", 8.0, 40), tmp_path / "nothing.pt") is None


def test_smoke_preset_does_not_export() -> None:
    import contextlib
    import io

    from docscanner.all_in_one import main as all_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = all_main(["--preset", "smoke", "--dry-run"])
    assert rc == 0
    assert "models/ left alone" in buf.getvalue(), (
        "the smoke preset should announce that it skips export")


def test_ablation_run_names_match_between_training_and_evaluation() -> None:
    import re

    trained = set(re.findall(r'"--name", "([a-z_0-9]+)"',
                             (ROOT / "src/docscanner/all_in_one.py").read_text()))
    wanted = set(re.findall(r'runs / "([a-z_0-9]+)"',
                            (ROOT / "src/docscanner/eval/evaluate.py").read_text()))

    assert trained, "expected all_in_one to name its runs"
    assert wanted, "expected evaluate to read named runs"

    unfulfilled = sorted(wanted - trained)
    assert not unfulfilled, (
        "evaluate.py reads these runs, but `amazingscanner all` never trains "
        f"them, so their tables come out empty: {unfulfilled}")


def test_every_dropout_arm_has_a_matched_control() -> None:
    import re

    text = (ROOT / "src/docscanner/all_in_one.py").read_text()
    entries = re.findall(r'\[([^\[\]]*?"--name", "(abl_[a-z_0-9]+)"[^\[\]]*?)\]', text,
                         re.S)
    by_name = {name: argv for argv, name in entries}

    for drop, ctrl in (("abl_heat_drop", "abl_heat_ctrl"),
                       ("abl_reg_drop", "abl_reg_ctrl"),
                       ("abl_enhance_drop", "abl_enhance_ctrl")):
        assert drop in by_name, f"{drop} is not trained"
        assert ctrl in by_name, f"{ctrl} (the matched control for {drop}) is not trained"
        def budget(argv: str) -> dict[str, str]:
            pairs = re.findall(r'"(--epochs|--steps|--batch|--minutes)", "?([^",]+)"?',
                               argv)
            return dict(pairs)
        assert budget(by_name[drop]) == budget(by_name[ctrl]), (
            f"{drop} and {ctrl} are trained at different budgets, so the "
            f"comparison would measure the budget rather than dropout:\n"
            f"  {drop}: {budget(by_name[drop])}\n  {ctrl}: {budget(by_name[ctrl])}")


def test_losses_carry_device_bound_state() -> None:
    import torch

    from docscanner.models.losses import CornerCoordLoss, build_enhancement_loss

    combined = build_enhancement_loss("combined")
    buffers = list(combined.buffers())
    assert buffers, ("the combined loss is expected to hold buffers (Sobel "
                     "kernels); if it no longer does, this test is obsolete")
    assert all(b.device.type == "cpu" for b in buffers)

    moved = combined.to("meta")
    assert all(b.device.type == "meta" for b in moved.buffers()), (
        "`.to()` must relocate the loss buffers — otherwise moving the "
        "criterion would be a no-op and the CUDA fix would be illusory")

    CornerCoordLoss("wing").to("meta")
    assert isinstance(torch.zeros(1), torch.Tensor)


def test_every_trainer_moves_its_criterion_and_validation_loader() -> None:
    import ast

    trainers = {
        "train_enhance.py": ["criterion"],
        "train_corners.py": ["coord_loss", "heat_loss"],
        "finetune_e2e.py": ["enh_loss", "coord_loss"],
    }
    problems: list[str] = []
    for filename, losses in trainers.items():
        path = ROOT / "src" / "docscanner" / "engine" / filename
        text = path.read_text()
        tree = ast.parse(text)

        if "resolve_device(args.device)" not in text:
            problems.append(f"{filename}: never resolves a device of its own")
        if "DeviceLoader(" not in text:
            problems.append(f"{filename}: validation loader is not wrapped in "
                            f"DeviceLoader, so it yields CPU tensors")

        moved: dict[str, bool] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            hit = next((n for n in names if n in losses), None)
            if hit is None:
                continue
            value = node.value
            moved[hit] = (isinstance(value, ast.Call)
                          and isinstance(value.func, ast.Attribute)
                          and value.func.attr == "to")

        for loss in losses:
            if loss not in moved:
                problems.append(f"{filename}: expected an assignment to `{loss}`")
            elif not moved[loss]:
                problems.append(f"{filename}: `{loss}` is built without .to(device)")

    assert not problems, "\n  ".join(["device placement gaps:", *problems])


def test_no_bare_numpy_on_possibly_device_tensors() -> None:
    offenders: list[str] = []
    for path in (ROOT / "src" / "docscanner" / "engine").glob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if ".numpy()" in line and ".cpu().numpy()" not in line:
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, (
        "these raise on CUDA — use .cpu().numpy():\n  " + "\n  ".join(offenders))


def test_pyproject_dependencies_match_requirements() -> None:
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = {re.split(r"[><=\[]", d)[0].strip().lower().replace("_", "-")
                for group in [pyproject["project"]["dependencies"]]
                for d in group}
    for extra in pyproject["project"]["optional-dependencies"].values():
        declared |= {re.split(r"[><=\[]", d)[0].strip().lower().replace("_", "-")
                     for d in extra}

    required = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            required.add(re.split(r"[><=\[]", line)[0].strip().lower().replace("_", "-"))

    missing = sorted(required - declared - {"docscanner"})
    assert not missing, (
        f"in requirements.txt but not declared in pyproject.toml: {missing}")


def test_no_references_to_deleted_scripts() -> None:
    gone = ["train_all.sh", "finalize.sh", "train_ablations.sh",
            "train_corners_only.sh", "train_flagship_long.sh", "stop_training.sh"]

    searchable = [p for p in ROOT.rglob("*")
                  if p.is_file()
                  and not any(part in {".git", ".venv", "outputs", "runs", "data",
                                       "__pycache__", ".pytest_cache", ".ruff_cache"}
                              for part in p.parts)
                  and p.suffix in {".py", ".yml", ".yaml", ".md", ".sh", ".toml", ".cfg", ""}]

    offenders: list[str] = []
    for path in searchable:
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if path.name in {"CHANGELOG.md", Path(__file__).name}:
            continue
        for script in gone:
            if script in text:
                offenders.append(f"{path.relative_to(ROOT)} still references {script}")

    assert not offenders, "\n".join(offenders)


def test_every_demo_and_chart_is_registered_and_used() -> None:
    import inspect

    from docscanner.eval import charts, figures

    main_src = inspect.getsource(figures.main)
    unregistered = sorted(
        name for name in vars(figures)
        if (name.startswith(("gif_", "figure_"))
            and inspect.isfunction(getattr(figures, name))
            and f"{name}(" not in main_src)
    )
    assert not unregistered, (
        f"drawing functions never called by figures.main(): {unregistered}")

    registered = {fn.__name__ for fn in charts.ALL}
    missing = sorted(
        name for name in vars(charts)
        if (name.startswith("chart_")
            and inspect.isfunction(getattr(charts, name))
            and name not in registered)
    )
    assert not missing, f"chart functions missing from charts.ALL: {missing}"
