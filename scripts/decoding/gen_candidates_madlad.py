"""Generate N candidate translations per AmericasNLP source from MADLAD-400
(google/madlad400-*-mt) and dump them in the SAME jsonl schema as the other
candidate generators ({idx, source, reference, greedy, candidates:[...]}), so
ensemble_mbr_rerank.py can pool MADLAD + NLLB + Qwen candidates.

MADLAD (T5-based) controls the target language via an input PREFIX '<2quy>',
not a forced BOS token. Output is the plain translation.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/madlad400-3b-mt")
    ap.add_argument("--test-es", default="docs/references/americasnlp_test/2021_test.es")
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--tgt-tag", default="<2quy>")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-beams-greedy", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--suppress-apostrophe", action="store_true", default=True)
    ap.add_argument("--allow-apostrophe", dest="suppress_apostrophe", action="store_false")
    ap.add_argument("--out-candidates", required=True)
    args = ap.parse_args()

    import torch, sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    src = [l.strip() for l in open(args.test_es).read().strip().split("\n")]
    ref = [l.strip() for l in open(args.test_quy).read().strip().split("\n")]
    assert len(src) == len(ref)
    inputs = [f"{args.tgt_tag} {s}" for s in src]

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    bad_words_ids = None
    if args.suppress_apostrophe:
        bad = set()
        for t, tid in tok.get_vocab().items():
            piece = t.replace("▁", "")
            if any(c in piece for c in ["'", "’", "ʼ"]):
                bad.add(tid)
        bad_words_ids = [[i] for i in sorted(bad)] or None
        print(f"[{time.strftime('%H:%M:%S')}] suppressing {len(bad)} apostrophe tokens", flush=True)

    greedy, cand_lists = [], [[] for _ in src]
    for i in range(0, len(inputs), args.batch_size):
        batch = inputs[i:i+args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, num_beams=args.num_beams_greedy,
                               max_new_tokens=args.max_new, bad_words_ids=bad_words_ids)
            s = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                               num_return_sequences=args.n_samples, max_new_tokens=args.max_new,
                               bad_words_ids=bad_words_ids)
        greedy.extend(tok.batch_decode(g, skip_special_tokens=True))
        dec = tok.batch_decode(s, skip_special_tokens=True)
        for b in range(len(batch)):
            cand_lists[i+b] = [c.strip() for c in dec[b*args.n_samples:(b+1)*args.n_samples] if c.strip()]
        print(f"{min(i+args.batch_size,len(src))}/{len(src)}", flush=True)

    Path(args.out_candidates).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_candidates, "w") as f:
        for i, (sline, r) in enumerate(zip(src, ref)):
            f.write(json.dumps({"idx": i, "source": sline, "reference": r,
                                "greedy": greedy[i].strip(), "candidates": cand_lists[i]}, ensure_ascii=False) + "\n")

    chrf0 = lambda P: round(sacrebleu.corpus_chrf(P, [ref], word_order=0).score, 3)
    print(json.dumps({"model": args.model_id, "n": args.n_samples,
                      "ChrF_greedy_beam": chrf0([g.strip() for g in greedy]),
                      "out": args.out_candidates}, indent=2))


if __name__ == "__main__":
    main()
