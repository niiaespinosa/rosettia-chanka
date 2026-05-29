"""Quantify the orthographic ChrF penalty on AmericasNLP from training on
MINEDU-normalized targets while the test references are un-normalized.

Takes a predictions jsonl ({source, reference, prediction}) from a model trained
on normalized data. Reports ChrF(pred vs original ref) [the official score] and
ChrF after applying the SAME deterministic normalization to the references (and
to both sides). If the normalized-ref score is much higher, the gap is orthographic.
"""
import argparse
import json
import sys

sys.path.insert(0, "scripts/normalizer")
from apply_normalizer_vllm import _safe_token_transform  # noqa: E402
import sacrebleu  # noqa: E402


def det_norm(s: str) -> str:
    return " ".join(_safe_token_transform(t) for t in s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-jsonl", required=True)
    args = ap.parse_args()

    P, R = [], []
    for l in open(args.pred_jsonl):
        l = l.strip()
        if not l:
            continue
        d = json.loads(l)
        P.append((d.get("prediction") or "").strip())
        R.append((d.get("reference") or "").strip())

    def chrf(p, r):
        return round(sacrebleu.corpus_chrf(p, [r], word_order=0).score, 3)

    Pn = [det_norm(p) for p in P]
    Rn = [det_norm(r) for r in R]

    print(f"n={len(P)}")
    print(f"[official]      ChrF(pred,            ref)            = {chrf(P, R)}")
    print(f"[ref-norm]      ChrF(pred,            norm(ref))      = {chrf(P, Rn)}")
    print(f"[both-norm]     ChrF(norm(pred),      norm(ref))      = {chrf(Pn, Rn)}")
    print(f"[pred-norm]     ChrF(norm(pred),      ref)            = {chrf(Pn, R)}")
    print("If ref-norm/both-norm >> official, the gap is orthographic "
          "(model outputs normalized spelling, refs are un-normalized).")


if __name__ == "__main__":
    main()
