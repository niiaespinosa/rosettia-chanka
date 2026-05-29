"""Build the v34-RAW MT corpus: identical row set to clean3 but with the
UN-NORMALIZED (original) Quechua orthography as the target.

Why: the AmericasNLP 2021 test references are in the original (un-normalized)
orthography (~44% of ref lines contain e/o, which MINEDU normalization strips
to i/u). Training on MINEDU-normalized targets makes the model output spellings
that systematically mismatch the references -> ChrF penalty. v34a (normalized)
scored 37.9 < v30 40.55 despite 56x more in-domain data. This rebuild swaps the
target back to original orthography, isolating orthography as the only variable.

Join key: (spanish, chanka_normalized) -> original, recovered from the
per-sentence normalizer-apply jsonls. Falls back to keeping the normalized form
if no original is found (should be ~0 rows).
"""
import argparse
import json
import re
from pathlib import Path

import polars as pl


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def load_orig_map(jsonl_paths):
    """(spanish_key, normalized_key) -> original quy."""
    m = {}
    for p in jsonl_paths:
        if not Path(p).exists():
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            spa = (d.get("spanish") or "").strip()
            normd = (d.get("chanka_normalized") or "").strip()
            orig = (d.get("original") or "").strip()
            if not orig:
                continue
            m[(norm_key(spa), norm_key(normd))] = orig
            # also index by spanish alone as a weaker fallback
            m.setdefault((norm_key(spa), None), orig)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean3-parquet", required=True,
                    help="the normalized clean3 parquet (reviewed_spanish/reviewed_chanka_quechua)")
    ap.add_argument("--amnlp-jsonl", required=True)
    ap.add_argument("--v30-jsonl", required=True)
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    test_keys = set()
    if Path(args.test_quy).exists():
        test_keys = {norm_key(l) for l in open(args.test_quy) if l.strip()}

    omap = load_orig_map([args.amnlp_jsonl, args.v30_jsonl])
    print(f"orig-map entries: {len(omap)}")

    df = pl.read_parquet(args.clean3_parquet)
    print(f"clean3 rows: {df.height}  cols={df.columns}")

    out_rows = []
    n_exact = n_spa = n_fallback_norm = n_leak = 0
    for r in df.iter_rows(named=True):
        spa = str(r["reviewed_spanish"]).strip()
        normd = str(r["reviewed_chanka_quechua"]).strip()
        if not spa or not normd:
            continue
        sk = norm_key(spa)
        orig = omap.get((sk, norm_key(normd)))
        if orig is not None:
            n_exact += 1
        else:
            orig = omap.get((sk, None))
            if orig is not None:
                n_spa += 1
            else:
                orig = normd  # last-resort: keep normalized (rare)
                n_fallback_norm += 1
        if norm_key(orig) in test_keys:
            n_leak += 1
            continue
        out_rows.append({"reviewed_spanish": spa, "reviewed_chanka_quechua": orig})

    out = pl.DataFrame(out_rows)
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out_parquet)
    print(f"matched exact(spa+norm): {n_exact}  spa-only: {n_spa}  kept-normalized: {n_fallback_norm}")
    print(f"leak-dropped: {n_leak}")
    print(f"TOTAL v34-RAW rows: {out.height} -> {args.out_parquet}")


if __name__ == "__main__":
    main()
