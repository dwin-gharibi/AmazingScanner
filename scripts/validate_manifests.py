#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OK, FAIL, SKIP = "ok", "fail", "skip"
_MARK = {OK: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{_MARK[status]}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def run(cmd: list[str], name: str, cwd: Path | None = None) -> None:
    if not shutil.which(cmd[0]):
        record(SKIP, name, f"{cmd[0]} not installed")
        return
    proc = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        record(OK, name)
    else:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        record(FAIL, name, " / ".join(tail[-3:]) if tail else f"exit {proc.returncode}")


_METADATA_OPTIONAL = {"Kustomization", "Component"}


def _argv_tokens_are_split(doc: dict, where: str) -> list[str]:
    problems: list[str] = []
    interpreter_flags = {"-c", "-lc", "-ec", "-euc", "-e", "--command", "-Command"}

    def check(seq, field: str) -> None:
        if not isinstance(seq, list):
            return
        for i, tok in enumerate(seq):
            if not isinstance(tok, str) or " " not in tok.strip():
                continue
            if i and isinstance(seq[i - 1], str) and seq[i - 1] in interpreter_flags:
                continue
            if "\n" in tok:
                continue
            problems.append(f"{where}: {field}[{i}] is one string with spaces: {tok!r}")

    for key in ("command", "args", "entrypoint"):
        check(doc.get(key), key)
    return problems


def check_yaml_and_argv() -> None:
    import yaml

    targets = sorted(
        list((ROOT / "deploy").rglob("*.y*ml"))
        + list((ROOT / ".github").rglob("*.y*ml"))
        + [ROOT / "docker-compose.yml", ROOT / "Taskfile.yml"]
        + list((ROOT / "configs").glob("*.y*ml"))
    )
    targets = [p for p in targets if "helm/templates" not in p.as_posix()]

    bad_parse: list[str] = []
    bad_argv: list[str] = []
    k8s_docs = 0

    for path in targets:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        try:
            docs = [d for d in yaml.safe_load_all(path.read_text()) if isinstance(d, dict)]
        except yaml.YAMLError as exc:
            bad_parse.append(f"{rel}: {str(exc).splitlines()[0]}")
            continue

        for doc in docs:
            if {"apiVersion", "kind"} <= doc.keys():
                k8s_docs += 1
                if "metadata" not in doc and doc["kind"] not in _METADATA_OPTIONAL:
                    bad_parse.append(f"{rel}: {doc['kind']} has no metadata")
                for container in _containers(doc):
                    name = container.get("name", "?")
                    bad_argv += _argv_tokens_are_split(container, f"{rel}:{name}")

            for svc_name, svc in (doc.get("services") or {}).items():
                if isinstance(svc, dict):
                    bad_argv += _argv_tokens_are_split(svc, f"{rel}:{svc_name}")

    if bad_parse:
        record(FAIL, "yaml parses", f"{len(bad_parse)} problem(s)")
        for p in bad_parse:
            print(f"         {p}")
    else:
        record(OK, "yaml parses", f"{len(targets)} files, {k8s_docs} kubernetes documents")

    if bad_argv:
        record(FAIL, "container argv is tokenised", f"{len(bad_argv)} problem(s)")
        for p in bad_argv:
            print(f"         {p}")
    else:
        record(OK, "container argv is tokenised")


def _containers(doc: dict) -> list[dict]:
    found: list[dict] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key in ("containers", "initContainers"):
                if isinstance(node.get(key), list):
                    found.extend(c for c in node[key] if isinstance(c, dict))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(doc)
    return found


def check_entrypoints_exist() -> None:
    import re

    try:
        from docscanner.cli import COMMANDS
    except Exception as exc:
        record(SKIP, "manifest commands resolve", f"docscanner not importable ({exc})")
        return

    pattern = re.compile(
        r"""(?:^[ \t>|\-]*|["'\[/])amazingscanner["'\]]?[ \t]*,?[ \t]*["'\[]?([a-z][a-z0-9-]*)""",
        re.M)
    unknown: list[str] = []
    checked = 0
    for path in [
        *(ROOT / "deploy").rglob("*"),
        ROOT / "docker-compose.yml", ROOT / "Makefile", ROOT / "Taskfile.yml",
        ROOT / "Dockerfile", ROOT / "Dockerfile.gpu",
    ]:
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for cmd in pattern.findall(text):
            checked += 1
            if cmd not in COMMANDS:
                unknown.append(f"{path.relative_to(ROOT)}: 'amazingscanner {cmd}' is not a command")

    if unknown:
        record(FAIL, "manifest commands resolve", f"{len(unknown)} problem(s)")
        for u in unknown:
            print(f"         {u}")
    else:
        record(OK, "manifest commands resolve", f"{checked} invocation(s)")


def main(argv: list[str] | None = None) -> int:
    print("Validating deployment manifests\n" + "-" * 62)

    check_yaml_and_argv()
    check_entrypoints_exist()

    k8s = ROOT / "deploy" / "k8s"
    if (k8s / "kustomization.yaml").exists():
        run(["kubectl", "kustomize", str(k8s)], "kustomize renders")

    helm = ROOT / "deploy" / "helm"
    if helm.is_dir():
        run(["helm", "lint", str(helm)], "helm lint")
        run(["helm", "template", "amazingscanner", str(helm)], "helm template renders")

    tf = ROOT / "deploy" / "terraform"
    if tf.is_dir():
        tool = "tofu" if shutil.which("tofu") else "terraform"
        run([tool, "fmt", "-check", "-recursive"], f"{tool} fmt", cwd=tf)
        run([tool, "init", "-backend=false", "-input=false"], f"{tool} init", cwd=tf)
        run([tool, "validate"], f"{tool} validate", cwd=tf)

    ansible = ROOT / "deploy" / "ansible" / "playbook.yml"
    if ansible.exists():
        run(["ansible-playbook", "--syntax-check",
             "-i", str(ansible.parent / "inventory.ini"), str(ansible)],
            "ansible syntax-check")

    compose = ROOT / "docker-compose.yml"
    if compose.exists() and shutil.which("docker"):
        run(["docker", "compose", "-f", str(compose), "config", "--quiet"], "compose config")

    print("-" * 62)
    failed = [r for r in results if r[0] == FAIL]
    skipped = [r for r in results if r[0] == SKIP]
    passed = [r for r in results if r[0] == OK]
    print(f"{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped"
          + (" (missing tools are not failures)" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
