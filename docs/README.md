# Documentation

Eight documents, each with one job. Start wherever your question is.

| If you want to… | Read |
|---|---|
| **see what this is** and what it measures | the [project README](../README.md) — demos, results, the whole tour |
| **run it** — every command, every flag | **[CLI.md](CLI.md)** |
| **grade it** — every rubric line, its evidence and its number | **[SCORING.md](SCORING.md)** |
| **check the numbers** — the full measured tables | **[REPORT.md](REPORT.md)** *(generated)* |
| **read one chart at a time**, each with the sentence that explains it | **[CHARTS.md](CHARTS.md)** *(generated)* |

> [!CAUTION]
> **Two of these are generated — do not hand-edit them.**

## Two of these are generated — do not hand-edit them

`REPORT.md` and `CHARTS.md` are written from `outputs/report/*.json` by
`amazingscanner report` and `amazingscanner charts`. A hand edit survives
exactly until the next `amazingscanner refresh`, and CI fails the
`documentation integrity` job when a committed copy no longer matches what
regeneration produces — deliberately, because a number that drifts from its
measurement is worse than no number. Change the measurement, or change the
template in `src/docscanner/eval/`.

The same applies to the README's results block, which sits between
`<!-- RESULTS:START -->` and `<!-- RESULTS:END -->`.

## What lives where else

| Path | What it holds |
|---|---|
| `assets/` | every figure, chart and animation the docs embed |
| `assets/stages/` | one strip per real photograph — photo, detection, rectification, enhancement, reference |
| `../CHANGELOG.md` | the project log, newest first, with the measurement behind each entry |
| `../CONTRIBUTING.md` | how to run the checks before pushing |
| `../SECURITY.md` | reporting a vulnerability |
| `../models/enhance.pt.pin` | *why* the shipped enhancement checkpoint is the one that ships |
