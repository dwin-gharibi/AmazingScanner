from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

__all__ = ["PRESETS", "main"]


@dataclass
class Preset:
    name: str
    description: str
    enh_epochs: int
    enh_steps: int
    enh_batch: int
    enh_minutes: float
    cor_epochs: int
    cor_steps: int
    cor_batch: int
    cor_minutes: float
    abl_minutes: float
    e2e_minutes: float
    with_ablations: bool = True
    with_finetune: bool = True
    extra: dict = field(default_factory=dict)

PRESETS: dict[str, Preset] = {
    "smoke": Preset(
        "smoke", "every step at a toy budget — proves the pipeline runs (~10 min)",
        enh_epochs=2, enh_steps=12, enh_batch=6, enh_minutes=3,
        cor_epochs=2, cor_steps=12, cor_batch=8, cor_minutes=3,
        abl_minutes=1.5, e2e_minutes=1.5),
    "cpu": Preset(
        "cpu", "what a few CPU cores can finish (~4 h)",
        enh_epochs=12, enh_steps=170, enh_batch=10, enh_minutes=100,
        cor_epochs=12, cor_steps=130, cor_batch=16, cor_minutes=90,
        abl_minutes=15, e2e_minutes=18),
    "gpu": Preset(
        "gpu", "one GPU, every epoch actually finishing (~3 h)",
        enh_epochs=18, enh_steps=200, enh_batch=24, enh_minutes=75,
        cor_epochs=18, cor_steps=160, cor_batch=32, cor_minutes=65,
        abl_minutes=25, e2e_minutes=25),
    "paper": Preset(
        "paper", "longer schedules for the best numbers (~8 h)",
        enh_epochs=40, enh_steps=320, enh_batch=32, enh_minutes=180,
        cor_epochs=40, cor_steps=260, cor_batch=48, cor_minutes=160,
        abl_minutes=45, e2e_minutes=45),
}


def _run(label: str, module: str, argv: list[str]) -> tuple[str, str, float]:
    import importlib
    import traceback

    print(f"\n{'=' * 70}\n=== {label}\n{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        m = importlib.import_module(module)
        rc = m.main(argv) or 0
        status = "ok" if rc == 0 else f"exit {rc}"
    except Exception as exc:
        status = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return label, status, time.time() - t0


def _device_default() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train everything and regenerate every artefact, in one command")
    ap.add_argument("--preset", choices=list(PRESETS), default=None,
                    help="budget for every run (default: gpu if CUDA, else cpu)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast; a CUDA win, ignored on CPU")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--workers", type=int, default=None,
                    help="loader processes (default: one per core, less one)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-data", action="store_true",
                    help="assume data/ is already prepared")
    ap.add_argument("--skip-ablations", action="store_true",
                    help="skip Section 6 dropout arms and the Section 3.2 loss ablation")
    ap.add_argument("--skip-refresh", action="store_true",
                    help="train only; do not regenerate tables, charts and demos")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit")
    ap.add_argument("--only", default=None, metavar="KEY[,KEY...]",
                    help="run only these steps, by key (see --list). Retraining "
                         "one arm after a fix costs minutes instead of hours; "
                         "everything else keeps the checkpoint it already has.")
    ap.add_argument("--list", action="store_true",
                    help="print the step keys this preset would run, and exit")
    args = ap.parse_args(argv)

    preset = PRESETS[args.preset or _device_default()]
    from .engine.common import default_workers
    workers = default_workers(args.workers)
    common = ["--out", args.out, "--device", args.device,
              "--workers", str(workers), "--threads", str(args.threads)]
    common.append("--auto-steps")
    if args.amp:
        common.append("--amp")

    enh = ["--epochs", str(preset.enh_epochs), "--steps", str(preset.enh_steps),
           "--batch", str(preset.enh_batch), "--minutes", str(preset.enh_minutes),
           "--lr", "2.5e-3"]
    cor = ["--epochs", str(preset.cor_epochs), "--steps", str(preset.cor_steps),
           "--batch", str(preset.cor_batch), "--minutes", str(preset.cor_minutes),
           "--lr", "2e-3"]
    abl_enh = ["--epochs", str(max(2, preset.enh_epochs // 2)),
               "--steps", str(preset.enh_steps), "--batch", str(preset.enh_batch),
               "--minutes", str(preset.abl_minutes), "--lr", "2.5e-3"]

    plan: list[tuple[str, str, str, list[str]]] = []

    if not args.skip_data:
        plan.append(("data", "datasets — corpora, 80/10/10 split by scan, frozen eval sets",
                     "docscanner.data.prepare", ["--all"]))

    plan += [
        ("enhance", "Task 1 — the enhancement network (§3)",
         "docscanner.engine.train_enhance",
         common + enh + ["--name", "enhance_main", "--loss", "combined",
                         "--no-bg-prior"]),
        ("heatmap", "Task 2 — Approach B, heatmaps (§5)",
         "docscanner.engine.train_corners",
         common + cor + ["--approach", "heatmap", "--name", "corner_heatmap"]),
        ("regression", "Task 2 — Approach A, direct regression (§5), matched budget",
         "docscanner.engine.train_corners",
         common + cor + ["--approach", "regression", "--name", "corner_regression"]),
    ]

    if preset.with_ablations and not args.skip_ablations:
        abl_cor = ["--epochs", str(max(2, preset.cor_epochs // 2)),
                   "--steps", str(preset.cor_steps), "--batch", str(preset.cor_batch),
                   "--minutes", str(preset.abl_minutes), "--lr", "2e-3"]
        plan += [
            ("abl_heat_ctrl", "§6 — heatmap control (dropout 0.0)",
             "docscanner.engine.train_corners",
             common + abl_cor + ["--approach", "heatmap", "--dropout", "0.0",
                                 "--name", "abl_heat_ctrl"]),
            ("abl_heat_drop", "§6 — heatmap + dropout 0.15, matched budget",
             "docscanner.engine.train_corners",
             common + abl_cor + ["--approach", "heatmap", "--dropout", "0.15",
                                 "--name", "abl_heat_drop"]),
            ("abl_seg_on", "§5 — heatmap + page-mask head",
             "docscanner.engine.train_corners",
             common + abl_cor + ["--approach", "heatmap", "--seg-weight", "0.5",
                                 "--name", "abl_seg_on"]),
            ("abl_seg_off", "§5 — heatmap without the mask head, matched budget",
             "docscanner.engine.train_corners",
             common + abl_cor + ["--approach", "heatmap", "--seg-weight", "0.0",
                                 "--name", "abl_seg_off"]),
            ("abl_reg_ctrl", "§6 — regression control (dropout 0.0)",
             "docscanner.engine.train_corners",
             common + abl_cor + ["--approach", "regression", "--dropout", "0.0",
                                 "--name", "abl_reg_ctrl"]),
            ("abl_reg_drop", "§6 — regression + dropout 0.3, matched budget",
             "docscanner.engine.train_corners",
             common + abl_cor + ["--approach", "regression", "--dropout", "0.3",
                                 "--name", "abl_reg_drop"]),
            ("abl_enhance_ctrl", "§3.2/§6 — enhancement control (combined loss, dropout 0.0)",
             "docscanner.engine.train_enhance",
             common + abl_enh + ["--name", "abl_enhance_ctrl", "--loss", "combined"]),
            ("abl_enhance_drop", "§6 — enhancement + dropout 0.15, matched budget",
             "docscanner.engine.train_enhance",
             common + abl_enh + ["--name", "abl_enhance_drop", "--loss", "combined",
                                 "--dropout", "0.15"]),
            ("abl_loss_mse", "§3.2 — loss ablation: MSE",
             "docscanner.engine.train_enhance",
             common + abl_enh + ["--name", "abl_loss_mse", "--loss", "mse"]),
            ("abl_loss_l1", "§3.2 — loss ablation: L1",
             "docscanner.engine.train_enhance",
             common + abl_enh + ["--name", "abl_loss_l1", "--loss", "l1"]),
            ("abl_bg_prior", "§3.2 — architecture ablation: with background prior",
             "docscanner.engine.train_enhance",
             common + abl_enh + ["--name", "abl_bg_prior", "--loss", "combined"]),
        ]

    if preset.with_finetune:
        plan.append(("finetune", "Bonus §7 — fine-tune the chain through the differentiable warp",
                     "docscanner.engine.finetune_e2e",
                     ["--out", args.out, "--device", args.device,
                      "--threads", str(args.threads),
                      "--minutes", str(preset.e2e_minutes)]))

    if not args.skip_refresh:
        refresh_argv = ["--runs", args.out]
        if preset.name == "smoke":
            refresh_argv += ["--skip", "export"]
            print("  note     smoke preset: export skipped, models/ left alone")
        plan.append(("refresh", "regenerate every table, chart, figure, demo and report",
                     "docscanner.refresh", refresh_argv))

    if args.only:
        wanted = [k.strip() for k in args.only.split(",") if k.strip()]
        known = {k for k, _, _, _ in plan}
        unknown = [k for k in wanted if k not in known]
        if unknown:
            print(f"unknown step key(s): {', '.join(unknown)}\n"
                  f"available with this preset: {', '.join(sorted(known))}")
            return 2
        plan = [p for p in plan if p[0] in wanted]

    if args.list:
        print(f"steps for preset '{preset.name}':")
        for key, label, _, _ in plan:
            print(f"  {key:<18} {label}")
        print("\nRetrain one:   amazingscanner all --only enhance --skip-data")
        print("Then refresh:  amazingscanner all --only refresh --skip-data")
        return 0

    print(f"AmazingScanner — train everything\n"
          f"  preset   {preset.name}: {preset.description}\n"
          f"  device   {args.device}{' + amp' if args.amp else ''}\n"
          f"  workers  {workers} (data generation is the bottleneck, not the GPU)\n"
          f"  steps    {len(plan)}\n")
    for i, (key, label, _, _) in enumerate(plan, 1):
        print(f"  {i:2d}. {label:<58} --only {key}")
    if args.dry_run:
        print("\n(dry run — nothing executed)")
        return 0

    results = []
    t_all = time.time()
    for _key, label, module, sub in plan:
        results.append(_run(label, module, sub))
        if results[-1][1] != "ok" and module.endswith("prepare"):
            print("\nDataset preparation failed; the rest cannot run. Stopping.")
            break

    print(f"\n{'=' * 70}")
    print(f"{'step':<62} {'status':<14}")
    for label, status, secs in results:
        print(f"{label[:60]:<62} {status:<14} {secs / 60:6.1f} min")
    failed = [r for r in results if r[1] != "ok"]
    print("=" * 70)
    print(f"total {(time.time() - t_all) / 60:.1f} min"
          + (f" — {len(failed)} step(s) failed" if failed else " — all steps ok"))
    if not failed:
        print(f"\nCheckpoints       {args.out}/*/best.pt")
        if args.skip_refresh:
            print("\nNothing was exported or measured — this run had --skip-refresh.")
            print(f"  Export and regenerate everything:  amazingscanner refresh "
                  f"--runs {args.out}")
        elif preset.name == "smoke":
            print("Measured tables   outputs/report/*.json, docs/REPORT.md")
            print("Charts and demos  docs/assets/")
            print("\nThe smoke preset does not export: models/ still holds the "
                  "weights it did\nbefore, which is deliberate — a two-epoch run "
                  "must not become what ships.")
        else:
            print("Deployment weights models/*.pt")
            print("Measured tables    outputs/report/*.json, docs/REPORT.md")
            print("Charts and demos   docs/assets/")
            print("\nNext:  git add -A && git commit -m 'Retrain' && git push")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
