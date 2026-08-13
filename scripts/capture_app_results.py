#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ASSETS = Path("docs/assets")
URL = "http://127.0.0.1:7860/"
LAUNCH_ARGS = ["--no-proxy-server", "--proxy-bypass-list=<-loopback>"]
MAX_WIDTH = 1600


def _shrink(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as im:
        if im.width <= MAX_WIDTH:
            im.save(path, optimize=True)
            return
        h = round(im.height * MAX_WIDTH / im.width)
        im.resize((MAX_WIDTH, h), Image.LANCZOS).save(path, optimize=True)


def _photo(stem: str) -> str:
    photos = Path("data/real/own/photos")
    for f in sorted(photos.glob("*.jpg")):
        if f.name.split("_")[0].split(".")[0] == stem:
            return str(f)
    have = ", ".join(sorted(f.name.split("_")[0].split(".")[0]
                            for f in photos.glob("*.jpg"))[:8])
    raise SystemExit(f"no photograph named {stem} under {photos} (have: {have} ...)")


def _plan() -> list[tuple[str, list[str], str, str]]:
    page = "data/test_pack/ground_truth/targets/pack_00_clean.png"
    return [
        ("Auto Scan", [_photo("img1")], "Scan document", "app_auto_scan_result.png"),
        ("Manual Corners", [_photo("img17")], "Rectify", "app_manual_corners_result.png"),
        ("Enhance Only", [page], "Enhance", "app_enhance_result.png"),
        ("Batch / Multi-page",
         [_photo("img3"), _photo("img4"), _photo("img10")],
         "Scan", "app_batch_result.png"),
        ("Corner Lab", [_photo("img17")], "Compare", "app_corner_lab_result.png"),
        ("OCR / Text", [_photo("img1")], "Recognise text", "app_ocr_result.png"),
        ("Degradation Lab", [], "Generate", "app_degradation_lab_result.png"),
        ("Benchmarks", [], "", "app_benchmarks_result.png"),
    ]


async def _run(out_dir: Path, check: bool) -> int:
    from playwright.async_api import async_playwright

    failures: list[str] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium",
                                          args=LAUNCH_ARGS)
        page = await browser.new_page(viewport={"width": 1500, "height": 1150},
                                      device_scale_factor=2)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        await page.wait_for_selector("button[role='tab']", timeout=90_000)
        await page.wait_for_timeout(2500)

        for label, files, button, name in _plan():
            try:
                await page.get_by_role("tab", name=label, exact=False).first.click()
                await page.wait_for_timeout(1800)
                panel = page.locator("div[role='tabpanel']:visible").first

                if files:
                    inputs = panel.locator("input[type='file']")
                    if await inputs.count():
                        await inputs.first.set_input_files(files)
                        await page.wait_for_timeout(3500)

                if button:
                    btn = panel.get_by_role("button", name=button, exact=False).first
                    await btn.click(timeout=15_000)
                    before, stable = -1, 0
                    for _ in range(90):
                        await page.wait_for_timeout(2000)
                        now = await panel.locator("img, table").count()
                        stable = stable + 1 if now == before and now else 0
                        before = now
                        if stable >= 3:
                            break
                    if before <= 0:
                        failures.append(f"{label}: no output after clicking {button!r}")

                await page.wait_for_timeout(1500)
                await page.screenshot(path=str(out_dir / name), full_page=True)
                _shrink(out_dir / name)
                print(f"  {out_dir / name}  "
                      f"({(out_dir / name).stat().st_size / 1e6:.1f} MB)")
            except Exception as exc:
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                print(f"  [fail] {label}: {type(exc).__name__}: {exc}")

        await browser.close()

    if errors:
        failures.append(f"page errors: {errors[:3]}")
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    if failures and check:
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ASSETS))
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any tab produced no output")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[app result screenshots] {URL}")
    return asyncio.run(_run(out, args.check))


if __name__ == "__main__":
    raise SystemExit(main())
