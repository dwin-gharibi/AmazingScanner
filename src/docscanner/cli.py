from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

__all__ = ["main", "COMMANDS"]


@dataclass(frozen=True)
class Command:
    module: str
    summary: str
    example: str = ""


COMMANDS: dict[str, Command] = {
    "all": Command(
        "docscanner.all_in_one",
        "build data, train everything, export, regenerate every artefact",
        "amazingscanner all --device cuda --amp"),
    "scan": Command(
        "docscanner.pipeline.run", "photo -> finished scan (the full chain)",
        "amazingscanner scan photo.jpg -o outputs/ --mode whiteboard"),
    "enhance": Command(
        "docscanner.pipeline.run", "enhance an already-rectified page",
        "amazingscanner enhance page.jpg -o outputs/"),
    "corners": Command(
        "docscanner.pipeline.run", "detect the four page corners of a photo",
        "amazingscanner corners photo.jpg -o outputs/"),
    "app": Command(
        "docscanner.app.gradio_app", "the interactive interface on :7860",
        "amazingscanner app --port 7860"),
    "label": Command(
        "docscanner.app.labeler", "the corner-labelling tool on :7861",
        "amazingscanner label --out data/real/own"),
    "fetch": Command(
        "docscanner.data.fetch", "download the source corpora (~1.8 GB, idempotent)",
        "amazingscanner fetch --check"),
    "data": Command(
        "docscanner.data.prepare", "download corpora, build and freeze splits",
        "amazingscanner data --all"),
    "roboflow": Command(
        "docscanner.data.roboflow", "sync the real test set with Roboflow",
        "amazingscanner roboflow --check"),
    "check-labels": Command(
        "docscanner.data.check_labels", "validate annotations, render a review sheet",
        "amazingscanner check-labels"),
    "train-enhance": Command(
        "docscanner.engine.train_enhance", "train the enhancement network",
        "amazingscanner train-enhance --epochs 20 --device cuda"),
    "train-corners": Command(
        "docscanner.engine.train_corners", "train a corner detector",
        "amazingscanner train-corners --approach heatmap --epochs 20"),
    "finetune": Command(
        "docscanner.engine.finetune_e2e", "bonus: fine-tune the chain end to end",
        "amazingscanner finetune --minutes 20"),
    "export": Command(
        "docscanner.engine.export_models", "write fp16 deployment weights",
        "amazingscanner export --out models"),
    "eval": Command(
        "docscanner.eval.evaluate", "every benchmark table",
        "amazingscanner eval --all"),
    "stages": Command(
        "docscanner.eval.stages", "per-stage inputs, outputs and accuracy",
        "amazingscanner stages --limit 5"),
    "charts": Command(
        "docscanner.eval.charts", "one chart per measured question",
        "amazingscanner charts"),
    "figures": Command(
        "docscanner.eval.figures", "documentation figures and demo animations",
        "amazingscanner figures --all"),
    "report": Command(
        "docscanner.eval.report", "regenerate docs/REPORT.md from the measurements",
        "amazingscanner report"),
    "refresh": Command(
        "docscanner.refresh",
        "regenerate EVERYTHING from the trained weights (run this after training)",
        "amazingscanner refresh"),
}

_PIPELINE = {"scan", "enhance", "corners"}

_GROUPS = (
    ("Everything at once", ("all",)),
    ("Scan something", ("scan", "enhance", "corners")),
    ("Interfaces", ("app", "label")),
    ("Data", ("fetch", "data", "roboflow", "check-labels")),
    ("Training", ("train-enhance", "train-corners", "finetune", "export")),
    ("Measurement", ("eval", "stages", "charts", "figures", "report")),
    ("After training", ("refresh",)),
)


def _usage() -> str:
    out = [
        "AmazingScanner — photograph a page, get a scan.",
        "",
        "  amazingscanner <command> [options]",
        "  amazingscanner <command> --help      options for that command",
        "",
    ]
    width = max(len(n) for n in COMMANDS)
    for title, names in _GROUPS:
        out.append(f"{title}:")
        for n in names:
            out.append(f"  {n:<{width}}  {COMMANDS[n].summary}")
        out.append("")
    out += [
        "Examples:",
        *(f"  {COMMANDS[n].example}" for n in ("all", "scan", "refresh")
          if COMMANDS[n].example),
        "",
        "Full documentation: docs/CLI.md",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        from . import __version__
        print(__version__)
        return 0

    name, rest = argv[0], argv[1:]
    cmd = COMMANDS.get(name)
    if cmd is None:
        close = [k for k in COMMANDS if k.startswith(name[:3])]
        print(f"unknown command {name!r}", file=sys.stderr)
        if close:
            print(f"did you mean: {', '.join(close)}?", file=sys.stderr)
        print("\n" + _usage(), file=sys.stderr)
        return 2

    if name in _PIPELINE:
        rest = [name, *rest]

    module = importlib.import_module(cmd.module)
    entry = getattr(module, "main", None)
    if entry is None:
        print(f"{cmd.module} has no main()", file=sys.stderr)
        return 1
    return int(entry(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
