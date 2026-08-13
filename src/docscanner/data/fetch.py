from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw"

CORPORA: dict[str, dict] = {
    "doclaynet": {
        "repo": "pierreguillou/DocLayNet-small",
        "allow": ["data/dataset_small.zip"],
        "dest": RAW / "doclaynet",
        "size_mb": 381,
        "role": "auxiliary training pages — 804 scans of reports, laws, manuals, papers",
        "extract": (RAW / "doclaynet" / "data" / "dataset_small.zip", RAW / "small_dataset"),
        "verify": RAW / "small_dataset",
    },
    "doclaynet_base": {
        "repo": "pierreguillou/DocLayNet-base",
        "allow": ["data/dataset_base.zip"],
        "dest": RAW / "doclaynet_base",
        "size_mb": 2500,
        "role": "10x more auxiliary training pages (~6-8k) for a robustness run",
        "extract": (RAW / "doclaynet_base" / "data" / "dataset_base.zip",
                    RAW / "base_dataset"),
        "verify": RAW / "base_dataset",
    },
    "dtd": {
        "url": "https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz",
        "repo": "cansa/Describable-Textures-Dataset-DTD",
        "allow": ["images/**"],
        "dest": RAW / "dtd",
        "size_mb": 625,
        "role": "background textures — the surfaces a page is photographed on",
        "extract_to": RAW / "dtd",
        "verify": RAW / "dtd" / "images",
    },
    "midv": {
        "repo": "Noaman/midv500",
        "allow": ["data/**", "id2label.json"],
        "dest": RAW / "midv500",
        "size_mb": 795,
        "role": "real phone photos — EVALUATION ONLY, never trained on",
        "verify": RAW / "midv500" / "data",
        "optional": True,
    },
}

COURSE_DIR = RAW / "course_scans"

COURSE_MISSING = f"""\
The 50 course scans are missing from {COURSE_DIR}.

These are the pages the project is graded on and the ground truth every
training pair is built from — without them there is nothing to train.

They ship with the repository, so an empty directory usually means a partial
clone or a deleted folder. Either restore it:

    git checkout -- {COURSE_DIR}

or unpack the assignment ZIP into it:

    unzip <the 50-scan zip> -d {COURSE_DIR}

Then re-run. `amazingscanner fetch --check` will confirm.\
"""


class CorpusMissingError(RuntimeError):
    """Raised when a corpus training depends on is absent and unfetchable."""


def _count(path: Path, patterns=("*.jpg", "*.png", "*.jpeg", "*.parquet")) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in patterns for _ in path.rglob(p))


def status() -> dict[str, tuple[bool, int]]:
    out = {"course": (COURSE_DIR.is_dir() and _count(COURSE_DIR) > 0, _count(COURSE_DIR))}
    for name, spec in CORPORA.items():
        n = _count(spec["verify"])
        out[name] = (n > 0, n)
    return out


def _download_parallel(url: str, dest: Path, jobs: int = 8,
                       size_mb: int = 0) -> Path:
    import concurrent.futures

    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")

    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers.get("Content-Length", 0))
    ranged = head.headers.get("Accept-Ranges", "").lower() == "bytes" and total > 0

    if not ranged:
        print(f"    server will not serve ranges — single stream "
              f"({size_mb or total // 1_000_000} MB)", flush=True)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        part.rename(dest)
        return dest

    with open(part, "wb") as fh:
        fh.truncate(total)

    chunk = (total + jobs - 1) // jobs
    spans = [(i * chunk, min((i + 1) * chunk, total) - 1) for i in range(jobs)]
    spans = [(a, b) for a, b in spans if a <= b]
    done = [0]

    def pull(span):
        start, end = span
        r = requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                         stream=True, timeout=300)
        r.raise_for_status()
        with open(part, "r+b") as fh:
            fh.seek(start)
            for block in r.iter_content(1 << 20):
                fh.write(block)
                done[0] += len(block)
        return end - start + 1

    print(f"    {total / 1e6:.0f} MB in {len(spans)} parallel ranges ...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        got = sum(f.result() for f in
                  concurrent.futures.as_completed([pool.submit(pull, s) for s in spans]))
    if got != total:
        part.unlink(missing_ok=True)
        raise OSError(f"expected {total} bytes, assembled {got}")
    part.rename(dest)
    return dest


def _extract(archive: Path, into: Path) -> None:
    marker = into / ".extracted"
    if marker.exists():
        return
    into.mkdir(parents=True, exist_ok=True)
    print(f"    unpacking {archive.name} ...", flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        import tarfile
        with tarfile.open(archive) as tf:
            tf.extractall(into)

    entries = [p for p in into.iterdir()
               if not p.name.startswith(".") and p != archive
               and p.suffix not in {".zip", ".gz", ".tar", ".part"}]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in list(inner.iterdir()):
            target = into / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
        if not any(inner.iterdir()):
            inner.rmdir()
    marker.write_text("ok\n")

    if archive.is_file() and archive.parent == into:
        freed = archive.stat().st_size / 1e6
        archive.unlink()
        print(f"    removed {archive.name} ({freed:.0f} MB reclaimed)", flush=True)


def _fetch_from_hub(name: str, spec: dict, jobs: int) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(f"  [fail] {name:<11s} needs huggingface_hub — pip install 'docscanner[data]'")
        return False
    try:
        snapshot_download(repo_id=spec["repo"], repo_type="dataset",
                          local_dir=str(spec["dest"]), allow_patterns=spec["allow"],
                          max_workers=max(8, jobs))
    except Exception as exc:
        print(f"  [fail] {name:<11s} {type(exc).__name__}: {exc}")
        return False
    if "extract" in spec:
        archive, into = spec["extract"]
        if archive.exists():
            _extract(archive, into)
    return True


def fetch_one(name: str, force: bool = False, jobs: int = 8) -> bool:
    spec = CORPORA[name]
    have = _count(spec["verify"])
    if have and not force:
        print(f"  [have] {name:<11s} {have} files")
        return True

    source = spec.get("url") or spec["repo"]
    print(f"  [get ] {name:<11s} {source}  (~{spec['size_mb']} MB) — {spec['role']}",
          flush=True)

    if spec.get("url"):
        try:
            archive = _download_parallel(
                spec["url"], spec["dest"] / Path(spec["url"]).name,
                jobs=jobs, size_mb=spec["size_mb"])
            _extract(archive, spec.get("extract_to", spec["dest"]))
        except Exception as exc:
            print(f"  [warn] {name:<11s} {type(exc).__name__}: {exc}")
            print(f"  [warn] {name:<11s} falling back to the Hugging Face mirror "
                  f"(slower — many small files)")
            if not _fetch_from_hub(name, spec, jobs):
                return False
    elif not _fetch_from_hub(name, spec, jobs):
        return False

    got = _count(spec["verify"])
    if not got:
        print(f"  [fail] {name:<11s} downloaded but {spec['verify']} is empty")
        return False
    print(f"  [ok  ] {name:<11s} {got} files")
    return True


def ensure_corpora(names: tuple[str, ...] = ("doclaynet", "dtd", "midv"),
                   force: bool = False, strict: bool = True,
                   jobs: int = 8) -> dict[str, bool]:
    RAW.mkdir(parents=True, exist_ok=True)

    if not (COURSE_DIR.is_dir() and _count(COURSE_DIR)):
        if strict:
            raise CorpusMissingError(COURSE_MISSING)
        print(f"  [WARN] course scans missing from {COURSE_DIR}")

    results = {name: fetch_one(name, force=force, jobs=jobs) for name in names}

    if strict:
        essential = [n for n, ok in results.items()
                     if not ok and not CORPORA[n].get("optional")]
        if essential:
            raise CorpusMissingError(
                f"Could not obtain: {', '.join(essential)}.\n"
                f"Training needs these. Check the network, then re-run "
                f"`amazingscanner fetch`.\n"
                f"If a download keeps failing, `amazingscanner fetch --check` "
                f"shows exactly what is present.")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the source corpora")
    ap.add_argument("--check", action="store_true",
                    help="report what is present and download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--no-midv", action="store_true",
                    help="skip the 795 MB real-photo evaluation set")
    ap.add_argument("--only", nargs="*", choices=list(CORPORA),
                    help="fetch just these")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel range requests / hub workers (default 8)")
    args = ap.parse_args(argv)

    if args.check:
        print("Corpora\n" + "-" * 62)
        rows = status()
        present, total_missing_mb = True, 0
        course_ok, course_n = rows["course"]
        print(f"  {'course scans':<16s} {'ok' if course_ok else 'MISSING':<8s} "
              f"{course_n:>5d} files   the graded distribution (in the repo)")
        present &= course_ok
        for name, spec in CORPORA.items():
            ok, n = rows[name]
            tag = "ok" if ok else ("optional" if spec.get("optional") else "MISSING")
            print(f"  {name:<16s} {tag:<8s} {n:>5d} files   {spec['role']}")
            if not ok:
                total_missing_mb += spec["size_mb"]
                present &= bool(spec.get("optional"))
        print("-" * 62)
        if present:
            print("Everything needed is present. Next: amazingscanner data --all")
            return 0
        print(f"Missing roughly {total_missing_mb} MB. Fetch it with: amazingscanner fetch")
        if not course_ok:
            print("\n" + COURSE_MISSING)
        return 1

    names = tuple(args.only) if args.only else tuple(
        n for n in CORPORA if not (args.no_midv and n == "midv"))
    print(f"Fetching corpora into {RAW}/\n" + "-" * 62)
    try:
        ensure_corpora(names, force=args.force, jobs=args.jobs)
    except CorpusMissingError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print("-" * 62)
    print("Done. Next: amazingscanner data --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
