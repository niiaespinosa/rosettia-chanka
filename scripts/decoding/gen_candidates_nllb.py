"""Generate N candidate translations per AmericasNLP source from an NLLB-200
(base or LoRA) model and dump them in the SAME jsonl schema as
gen_candidates_vllm.py ({idx, source, reference, greedy, candidates:[...]}), so
ensemble_mbr_rerank.py can pool NLLB + Qwen candidates for cross-architecture MBR.

NLLB is encoder-decoder -> uses transformers generate (sampling for candidates,
beam for greedy), with apostrophe suppression (Ayacucho quy refs have none).
"""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="facebook/nllb-200-1.3B")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir")
    ap.add_argument("--test-es", default="docs/references/americasnlp_test/2021_test.es")
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--src-lang", default="spa_Latn")
    ap.add_argument("--tgt-lang", default="quy_Latn")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-beams-greedy", type=int, default=5)
    ap.add_argument("--no-repeat-ngram", type=int, default=3,
                    help="block repeating any n-gram of this size (kills degenerate loops); 0 disables")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--suppress-apostrophe", action="store_true", default=True)
    ap.add_argument("--allow-apostrophe", dest="suppress_apostrophe", action="store_false")
    ap.add_argument("--out-candidates", required=True)
    args = ap.parse_args()

    import torch, sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    src = [l.strip() for l in open(args.test_es).read().strip().split("\n")]
    ref = [l.strip() for l in open(args.test_quy).read().strip().split("\n")]
    assert len(src) == len(ref)

    tok = AutoTokenizer.from_pretrained(args.model_id, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).cuda()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    bos = tok.convert_tokens_to_ids(args.tgt_lang)

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
    for i in range(0, len(src), args.batch_size):
        batch = src[i:i+args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        nrg = {"no_repeat_ngram_size": args.no_repeat_ngram} if args.no_repeat_ngram > 0 else {}
        with torch.no_grad():
            g = model.generate(**enc, forced_bos_token_id=bos, num_beams=args.num_beams_greedy,
                               max_new_tokens=args.max_new, bad_words_ids=bad_words_ids, **nrg)
            s = model.generate(**enc, forced_bos_token_id=bos, do_sample=True,
                               temperature=args.temperature, top_p=args.top_p,
                               num_return_sequences=args.n_samples, max_new_tokens=args.max_new,
                               bad_words_ids=bad_words_ids, **nrg)
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
    print(json.dumps({"model": args.model_id, "adapter": args.adapter, "n": args.n_samples,
                      "ChrF_greedy_beam": chrf0([g.strip() for g in greedy]),
                      "out": args.out_candidates}, indent=2))


if __name__ == "__main__":
    main()
