"""Build the round-2 NLLB corpus: the cleaned 124k aggregate + target-side
synthetic data (forward-translated Spanish->quy from a strong model). Output a
local parquet (reviewed_spanish / reviewed_chanka_quechua) for train_nllb_chanka.py.

Leakage guard: drop any pair whose Spanish OR Quechua matches the AmericasNLP 2021
test set (normalized key). Dedup exact (src,tgt). Light length-ratio filter.
"""
import argparse
import json
import re
from pathlib import Path

import polars as pl


def nk(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.strip().lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-parquet", required=True)
    ap.add_argument("--synth-jsonl", nargs="+", required=True)
    ap.add_argument("--test-es", default="docs/references/americasnlp_test/2021_test.es")
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    test_keys = set()
    for p in (args.test_es, args.test_quy):
        if Path(p).exists():
            test_keys |= {nk(l) for l in open(p).read().splitlines() if l.strip()}

    rows = []
    seen = set()
    n_agg = n_synth = n_leak = n_dup = n_len = 0

    df = pl.read_parquet(args.aggregate_parquet)
    for r in df.iter_rows(named=True):
        spa = str(r["reviewed_spanish"]).strip(); quy = str(r["reviewed_chanka_quechua"]).strip()
        if not spa or not quy:
            continue
        k = (nk(spa), nk(quy))
        if k in seen:
            n_dup += 1; continue
        seen.add(k)
        rows.append((spa, quy)); n_agg += 1

    for sj in args.synth_jsonl:
        for line in open(sj):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            spa = (d.get("spanish") or "").strip(); quy = (d.get("chanka") or "").strip()
            if not spa or not quy:
                continue
            if nk(spa) in test_keys or nk(quy) in test_keys:
                n_leak += 1; continue
            cr = len(quy) / max(1, len(spa))
            if cr > 2.5 or cr < 0.3:
                n_len += 1; continue
            k = (nk(spa), nk(quy))
            if k in seen:
                n_dup += 1; continue
            seen.add(k)
            rows.append((spa, quy)); n_synth += 1

    out = pl.DataFrame({"reviewed_spanish": [r[0] for r in rows],
                        "reviewed_chanka_quechua": [r[1] for r in rows]})
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out_parquet)
    print(f"aggregate kept: {n_agg}  synth kept: {n_synth}  "
          f"(leak {n_leak}, dup {n_dup}, len {n_len})")
    print(f"TOTAL round-2 corpus: {out.height} -> {args.out_parquet}")


if __name__ == "__main__":
    main()
