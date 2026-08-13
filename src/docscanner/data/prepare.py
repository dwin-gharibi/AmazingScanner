from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..utils.imageio import imwrite_rgb
from .corpus import IMAGE_EXT, BackgroundCorpus, ScanCorpus, _stable_hash, split_paths_by_parent
from .datasets import freeze_corners, freeze_pairs
from .degrade import DegradationConfig
from .fetch import COURSE_MISSING, CorpusMissingError, ensure_corpora
from .generator import SyntheticSampleGenerator

DATA = Path("data")
RAW = DATA / "raw"
FROZEN = DATA / "frozen"
REAL = DATA / "real"

DOCLAYNET_DIR = RAW / "small_dataset"
DOCLAYNET_DIRS = (RAW / "small_dataset", RAW / "base_dataset", RAW / "large_dataset")
COURSE_DIR = RAW / "course_scans"
DTD_DIR = RAW / "dtd" / "images"
MIDV_DIR = RAW / "midv500" / "data"

def build_splits(seed: int = 0,
                 ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict:
    splits: dict[str, dict] = {"scans": {}, "backgrounds": {}, "sources": {}}

    course = sorted(COURSE_DIR.glob("*.jpg")) + sorted(COURSE_DIR.glob("*.png"))
    doclaynet: list[Path] = []
    for root in DOCLAYNET_DIRS:
        for sub in ("train", "val", "test"):
            d = root / sub / "images"
            if d.exists():
                doclaynet += sorted(d.glob("*.png"))

    if not course:
        raise CorpusMissingError(COURSE_MISSING)
    if not doclaynet:
        raise CorpusMissingError(
            f"No auxiliary pages under {DOCLAYNET_DIR}.\n"
            f"\n"
            f"  Fetch them:   amazingscanner fetch\n"
            f"  Check first:  amazingscanner fetch --check\n"
            f"\n"
            f"DocLayNet supplies 804 further document pages. The generator can "
            f"make endless degradations of a page but cannot invent new "
            f"documents, so layout and typeface variety has to come from here.")

    per_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    provenance: dict[str, dict[str, int]] = {}
    by_source: dict[str, dict[str, list[str]]] = {}

    for name, paths in (("course", course), ("doclaynet", doclaynet)):
        order = np.random.default_rng((seed, _stable_hash(name))).permutation(len(paths))
        shuffled = [paths[i] for i in order]
        n = len(shuffled)
        n_train = int(round(n * ratios[0]))
        n_val = max(1, int(round(n * ratios[1]))) if n - n_train >= 2 else 0
        chunks = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }
        by_source[name] = {s: sorted(str(p) for p in v) for s, v in chunks.items()}
        provenance[name] = {s: len(v) for s, v in chunks.items()}
        for s, v in chunks.items():
            per_split[s] += [str(p) for p in v]

    for s in ("train", "val", "test"):
        splits["scans"][s] = sorted(per_split[s])
    splits["sources"] = provenance
    splits["scans_by_source"] = by_source

    bg_paths = [p for p in DTD_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXT] \
        if DTD_DIR.exists() else []
    bg = split_paths_by_parent(bg_paths, eval_fraction=0.25, seed=seed)
    splits["backgrounds"]["train"] = sorted(str(p) for p in bg["train"])
    splits["backgrounds"]["eval"] = sorted(str(p) for p in bg["eval"])
    splits["background_families"] = {
        "train": sorted({Path(p).parent.name for p in bg["train"]}),
        "eval": sorted({Path(p).parent.name for p in bg["eval"]}),
    }
    splits["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    splits["seed"] = seed
    return splits


class DataNotPreparedError(FileNotFoundError):
    """The datasets have not been built yet, and the fix is one command."""


def load_splits(path: Path = DATA / "splits.json") -> dict:
    path = Path(path)
    if not path.exists():
        raise DataNotPreparedError(
            f"{path} does not exist — the datasets have not been built yet.\n"
            f"\n"
            f"  Run this first:      amazingscanner data --all\n"
            f"  Or do everything:    amazingscanner all --device cuda\n"
            f"\n"
            f"`data --all` downloads the auxiliary corpora, splits the scans "
            f"80/10/10 by source page, and freezes the evaluation sets. It takes "
            f"a few minutes and only has to happen once."
        )
    return json.loads(path.read_text())

COURSE_WEIGHT = 0.45


def source_weights(paths, splits: dict, course_share: float = COURSE_WEIGHT):
    course = {Path(p).resolve() for p in splits.get("scans_by_source", {})
              .get("course", {}).get("train", [])}
    course |= {Path(p).resolve() for s in ("val", "test")
               for p in splits.get("scans_by_source", {}).get("course", {}).get(s, [])}
    flags = np.array([Path(p).resolve() in course for p in paths], bool)
    n_course, n_other = int(flags.sum()), int((~flags).sum())
    if n_course == 0 or n_other == 0:
        return None
    w = np.empty(len(paths), np.float64)
    w[flags] = course_share / n_course
    w[~flags] = (1.0 - course_share) / n_other
    return w


def corpora_for(split: str, splits: dict | None = None,
                bg_split: str | None = None,
                source: str | None = None) -> tuple[ScanCorpus, BackgroundCorpus]:
    splits = splits or load_splits()
    bg_key = bg_split or ("train" if split == "train" else "eval")
    if source:
        paths = splits["scans_by_source"][source][split]
        scans = ScanCorpus(paths=[Path(p) for p in paths],
                           name=f"scans-{source}-{split}", max_side=1800, cache_size=12)
        bgs = BackgroundCorpus(paths=[Path(p) for p in splits["backgrounds"][bg_key]],
                               name=f"bg-{bg_key}", max_side=900)
        return scans, bgs

    paths = [Path(p) for p in splits["scans"][split]]
    scans = ScanCorpus(paths=paths, name=f"scans-{split}", max_side=1800,
                       cache_size=12,
                       weights=source_weights(paths, splits, COURSE_WEIGHT))
    bgs = BackgroundCorpus(paths=[Path(p) for p in splits["backgrounds"][bg_key]],
                           name=f"bg-{bg_key}", max_side=900)
    return scans, bgs


def make_generator(split: str, cfg: DegradationConfig | None = None,
                   splits: dict | None = None, source: str | None = None,
                   **kw) -> SyntheticSampleGenerator:
    scans, bgs = corpora_for(split, splits, source=source)
    missing = [p for p in (*scans.paths, *bgs.paths) if not p.exists()]
    if missing:
        raise DataNotPreparedError(
            f"data/splits.json names {len(missing)} source files that are not "
            f"on disk (first: {missing[0]}).\n"
            f"\n"
            f"  Run this first:      amazingscanner data --all\n"
            f"  Or do everything:    amazingscanner all\n"
        )
    return SyntheticSampleGenerator(scans, bgs, cfg=cfg, **kw)


def build_frozen(splits: dict, n_val: int = 48, n_test: int = 64,
                 n_corner_val: int = 128, n_corner_test: int = 256,
                 n_pseudo: int = 96, page_long_side: int = 768) -> None:
    t0 = time.time()

    for split, n, seed in (("val", n_val, 101), ("test", n_test, 202)):
        gen = make_generator(split, cfg=DegradationConfig.for_enhancement(),
                             splits=splits, photo_long_side=1280)
        out = freeze_pairs(gen, FROZEN / f"enhance_{split}", n=n, seed=seed,
                           page_long_side=page_long_side)
        print(f"  enhance_{split:<5s} {n:4d} pages -> {out}")

    for split, n, seed in (("val", n_corner_val, 303), ("test", n_corner_test, 404)):
        gen = make_generator(split, cfg=DegradationConfig.for_corners(),
                             splits=splits, photo_long_side=640)
        out = freeze_corners(gen, FROZEN / f"corners_{split}", n=n, seed=seed, size=512)
        print(f"  corners_{split:<5s} {n:4d} photos -> {out}")

    gen = make_generator("test", cfg=DegradationConfig.for_enhancement(),
                         splits=splits, source="course", photo_long_side=1280)
    freeze_pairs(gen, FROZEN / "enhance_test_course", n=40, seed=707,
                 page_long_side=page_long_side)
    gen_c = make_generator("test", cfg=DegradationConfig.for_corners(),
                           splits=splits, source="course", photo_long_side=640)
    freeze_corners(gen_c, FROZEN / "corners_test_course", n=96, seed=808, size=512)
    print(f"  course-only  {40:4d} pages + {96} photos (graded distribution)")

    gen = make_generator("test", cfg=DegradationConfig.hard(), splits=splits)
    out = freeze_pairs(gen, FROZEN / "pseudo_real", n=n_pseudo, seed=505,
                       page_long_side=page_long_side)
    gen_c = make_generator("test", cfg=DegradationConfig.hard(), splits=splits,
                           photo_long_side=640)
    freeze_corners(gen_c, FROZEN / "pseudo_real_corners", n=n_pseudo, seed=606, size=512)
    print(f"  pseudo_real  {n_pseudo:4d} pages+photos -> {out}")
    print(f"  frozen sets built in {time.time() - t0:.0f}s")


def build_midv(limit: int = 220, stride: int = 3) -> None:
    from .real import from_midv_parquet

    shards = sorted(MIDV_DIR.glob("*.parquet"))
    if not shards:
        print("  [skip] no MIDV parquet shards found")
        return
    shards = sorted(shards, key=lambda p: (0 if "valid" in p.name else 1, p.name))
    out = from_midv_parquet(shards, REAL / "midv500", limit=limit, stride=stride)
    n = json.loads(out.read_text())["count"]
    print(f"  midv500      {n:4d} real photos -> {out}")


def scaffold_own_photos() -> None:
    root = REAL / "own"
    for sub in ("photos", "reference", "export"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Your own real photos\n\n"
            "1. Put 10-15 phone photos of documents in `photos/`.\n"
            "2. Put the matching scanner-app outputs (CamScanner / Adobe Scan /\n"
            "   the built-in scanner) in `reference/`, using the **same file stem**.\n"
            "3. Annotate the four page corners in Roboflow (keypoint project,\n"
            "   order: top-left, top-right, bottom-right, bottom-left), export as\n"
            "   *COCO Keypoints*, and unzip it into `export/`.\n"
            "4. Run:\n\n"
            "       python -m docscanner.data.prepare --own\n\n"
            "   which converts the export into `annotations.json`; every script and\n"
            "   the Gradio app pick it up automatically.\n\n"
            "These photos are for **evaluation only** and are never trained on.\n"
        )
    print(f"  own photos   scaffold ready -> {root}")


def build_own() -> None:
    from .real import from_coco

    root = REAL / "own"
    exports = sorted(p for d in root.glob("export*") if d.is_dir()
                     for p in d.rglob("*.json"))
    exports.sort(key=lambda p: (0 if "annotations" in p.name else 1, len(p.parts)))
    if not exports:
        print(f"  [skip] no COCO json under {root}/export*")
        return
    coco = exports[0]
    images_dir = coco.parent if any(coco.parent.glob("*.jpg")) else root / "photos"
    ref = root / "reference"
    out = from_coco(coco, images_dir, root, name="own-photos",
                    reference_dir=ref if any(ref.glob("*")) else None)
    data = json.loads(out.read_text())
    n = data["count"]
    n_ref = sum(1 for it in data["items"] if "reference" in it)
    print(f"  own photos   {n:4d} annotated -> {out}"
          + (f"  ({n_ref} with a commercial reference scan)" if n_ref else
             "  (no reference scans: the commercial-baseline column stays empty)"))


def build_previews(splits: dict, n: int = 8) -> None:
    out = Path("docs/assets/examples")
    out.mkdir(parents=True, exist_ok=True)
    gen = make_generator("test", splits=splits)
    for i in range(n):
        s = gen.generate(np.random.default_rng((909, i)))
        imwrite_rgb(out / f"photo_{i:02d}.jpg", s.photo, quality=92)
    print(f"  previews     {n:4d} example photos -> {out}")


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except CorpusMissingError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare docscanner datasets")
    ap.add_argument("--all", action="store_true", help="run every step")
    ap.add_argument("--splits", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--midv", action="store_true")
    ap.add_argument("--own", action="store_true")
    ap.add_argument("--previews", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fetch", action="store_true",
                    help="assume data/raw is already populated")
    ap.add_argument("--no-midv", action="store_true",
                    help="skip the 795 MB real-photo evaluation download")
    args = ap.parse_args(argv)
    if not any([args.all, args.splits, args.frozen, args.midv, args.own, args.previews]):
        args.all = True

    DATA.mkdir(exist_ok=True)

    if (args.all or args.splits or args.frozen) and not args.no_fetch:
        print("[0/5] corpora")
        names = tuple(n for n in ("doclaynet", "dtd", "midv")
                      if not (args.no_midv and n == "midv"))
        try:
            ensure_corpora(names)
        except CorpusMissingError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1

    if args.all or args.splits:
        print("[1/5] splits")
        splits = build_splits(seed=args.seed)
        (DATA / "splits.json").write_text(json.dumps(splits, indent=1))
        print(f"  scans      train={len(splits['scans']['train'])} "
              f"val={len(splits['scans']['val'])} test={len(splits['scans']['test'])}")
        for src, c in splits.get("sources", {}).items():
            print(f"    {src:<12s} train={c['train']} val={c['val']} test={c['test']}")
        print(f"  textures   train={len(splits['backgrounds']['train'])} "
              f"({len(splits['background_families']['train'])} families) "
              f"eval={len(splits['backgrounds']['eval'])} "
              f"({len(splits['background_families']['eval'])} families)")
    else:
        splits = load_splits()

    if args.all or args.frozen:
        print("[2/5] frozen evaluation sets")
        build_frozen(splits)
    if args.all or args.midv:
        print("[3/5] MIDV-500 real photos")
        build_midv()
    if args.all or args.own:
        print("[4/5] your own photos")
        scaffold_own_photos()
        build_own()
    if args.all or args.previews:
        print("[5/5] preview images")
        build_previews(splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
