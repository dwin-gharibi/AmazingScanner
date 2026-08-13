# Unseen test pack

Photographs generated for **testing only** — every source page is from the held-out test split (no model here has trained on it in any form) and every composite uses seeds no training or frozen-set tooling draws from.

18 photographs across three styles: `clean` (the enhancement policy), `corner-hard` (full circle of orientations, small pages), and `ood` (curl, glare, motion blur — degradations no model trained on). `ground_truth/corners.json` has the exact quads and `ground_truth/targets/` the clean pages, so runs can be scored:

```bash
amazingscanner scan data/test_pack/photos --out /tmp/pack_scans
python scripts/make_test_pack.py --n 18   # regenerate / extend
```

Regenerating with a different `--seed` (staying above 7000) mints a fresh pack the models have still never seen.
