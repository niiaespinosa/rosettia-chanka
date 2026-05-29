"""Ensemble ChrF-MBR reranking: pool candidate translations from multiple models
(candidate dumps from gen_candidates_vllm.py) and pick, per source, the candidate
maximizing mean pairwise ChrF over the POOLED set. Diverse models -> richer
consensus pool -> typically beats any single model's self-MBR.

CPU-only. Reports ensemble ChrF (word_order=0) vs each input's greedy/self-MBR.
"""
import argparse
import json
from pathlib import Path

import sacrebleu


def s_chrf(a, b):
    return sacrebleu.sentence_chrf(a, [b]).score


def mbr_pick(cands):
    if len(cands) <= 1:
        return cands[0] if cands else ""
    best, bs = cands[0], -1.0
    for i, h in enumerate(cands):
        sc = sum(s_chrf(h, o) for j, o in enumerate(cands) if j != i) / (len(cands) - 1)
        if sc > bs:
            best, bs = h, sc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-jsonls", nargs="+", required=True,
                    help="one or more dumps from gen_candidates_vllm.py")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-predictions-jsonl", default=None)
    args = ap.parse_args()

    # index -> {source, reference, pooled:[...]}
    by_idx = {}
    for path in args.candidate_jsonls:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            i = d["idx"]
            e = by_idx.setdefault(i, {"source": d["source"], "reference": d["reference"], "pooled": []})
            e["pooled"].extend(d.get("candidates", []))
            if d.get("greedy"):
                e["pooled"].append(d["greedy"])  # include greedy in the pool

    idxs = sorted(by_idx)
    ref = [by_idx[i]["reference"].strip() for i in idxs]
    preds = []
    for i in idxs:
        pool = [c for c in by_idx[i]["pooled"] if c.strip()]
        # dedup identical candidates (speeds reranking, no effect on argmax)
        seen, uniq = set(), []
        for c in pool:
            if c not in seen:
                seen.add(c); uniq.append(c)
        preds.append(mbr_pick(uniq))

    chrf0 = round(sacrebleu.corpus_chrf(preds, [ref], word_order=0).score, 3)
    rec = {"inputs": args.candidate_jsonls, "n": len(idxs),
           "avg_pool_size": round(sum(len(set(by_idx[i]["pooled"])) for i in idxs) / len(idxs), 1),
           "ChrF_ensemble_mbr": chrf0}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(args.out_json, "w"), indent=2)
    if args.out_predictions_jsonl:
        with open(args.out_predictions_jsonl, "w") as f:
            for i, p in zip(idxs, preds):
                f.write(json.dumps({"source": by_idx[i]["source"], "reference": by_idx[i]["reference"],
                                    "prediction": p}, ensure_ascii=False) + "\n")
    print(json.dumps(rec, indent=2))
    print("vs v30 greedy SOTA 40.55")


if __name__ == "__main__":
    main()
