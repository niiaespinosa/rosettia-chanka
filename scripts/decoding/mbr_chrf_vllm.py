"""ChrF-MBR decoding for Spanish->Chanka Quechua on the AmericasNLP 2021 test.

Minimum Bayes Risk decoding with ChrF as the (reference-free) utility:
  1. sample N candidate translations per source (temperature / epsilon sampling)
  2. pick the candidate maximizing mean pairwise sentence-ChrF against the others
     (the consensus / centroid hypothesis).

ChrF as utility is language-agnostic and directly optimizes the eval metric, and
unlike COMET it needs no quy-aware model (XLM-R has poor quy coverage). Expected
+1-3 ChrF on low-resource pairs (mbrs, arxiv 2408.04167).

Works on the Qwen chanka models via vLLM (merged base or base+LoRA adapter).
Reports both greedy/beam baseline (n=1) and MBR (n=N) ChrF word_order=0.
"""
import argparse
import json
import time
from pathlib import Path


# EXACT prompt format used by eval_americasnlp_2021_base_only.py (v30's 40.55).
SYSTEM = 'You are a careful Spanish→Quechua (Chanka/Ayacuchana variety) translator.'
INSTR = 'Traduce el siguiente texto del español al quechua chanka. Responde solo con la traducción.'


def build_prompt(tokenizer, src_text):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{INSTR}\n\nEspañol: {src_text}"}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def mbr_select(cands, chrf_fn):
    """Return the candidate with max mean pairwise ChrF utility."""
    if len(cands) == 1:
        return cands[0]
    best, best_score = cands[0], -1.0
    for i, hyp in enumerate(cands):
        # utility = mean sentence-ChrF of hyp vs every other candidate (pseudo-refs)
        others = [c for j, c in enumerate(cands) if j != i]
        score = sum(chrf_fn(hyp, o) for o in others) / max(1, len(others))
        if score > best_score:
            best, best_score = hyp, score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="merged base model dir or HF id")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter")
    ap.add_argument("--lora-rank", type=int, default=512)
    ap.add_argument("--test-es", default="docs/references/americasnlp_test/2021_test.es")
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-predictions-jsonl", default=None)
    args = ap.parse_args()

    import sacrebleu
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    src = [l.strip() for l in open(args.test_es) if l.strip()]
    ref = [l.strip() for l in open(args.test_quy) if l.strip()]
    assert len(src) == len(ref), f"{len(src)} != {len(ref)}"

    tok = AutoTokenizer.from_pretrained(args.base)
    prompts = [build_prompt(tok, s) for s in src]

    llm_kwargs = dict(model=args.base, enforce_eager=True,
                      gpu_memory_utilization=args.gpu_mem_frac, max_model_len=1024)
    lora_req = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.lora_rank)
        lora_req = LoRARequest("adapter", 1, args.adapter)
    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM base={args.base} adapter={args.adapter}", flush=True)
    llm = LLM(**llm_kwargs)

    def gen(sampling):
        kw = {"lora_request": lora_req} if lora_req else {}
        return llm.generate(prompts, sampling, **kw)

    # cheap sentence-ChrF utility
    def s_chrf(a, b):
        return sacrebleu.sentence_chrf(a, [b]).score

    STOP = ['<|im_end|>', '<|endoftext|>']
    # --- baseline: greedy (n=1) ---
    greedy = gen(SamplingParams(temperature=0.0, max_tokens=args.max_new, stop=STOP))
    base_preds = [o.outputs[0].text.strip() for o in greedy]

    # --- MBR: N samples, pick consensus ---
    samp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                          n=args.n_samples, max_tokens=args.max_new, stop=STOP)
    sampled = gen(samp)
    mbr_preds = []
    for o in sampled:
        cands = [c.text.strip() for c in o.outputs if c.text.strip()]
        mbr_preds.append(mbr_select(cands, s_chrf) if cands else "")

    def chrf0(P):
        return sacrebleu.corpus_chrf(P, [ref], word_order=0).score

    rec = {"base": args.base, "adapter": args.adapter, "n_samples": args.n_samples,
           "temperature": args.temperature,
           "ChrF_greedy": round(chrf0(base_preds), 3),
           "ChrF_mbr": round(chrf0(mbr_preds), 3),
           "n": len(src)}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(args.out_json, "w"), indent=2)
    if args.out_predictions_jsonl:
        with open(args.out_predictions_jsonl, "w") as f:
            for s, r, g, m in zip(src, ref, base_preds, mbr_preds):
                f.write(json.dumps({"source": s, "reference": r,
                                    "greedy": g, "mbr": m}, ensure_ascii=False) + "\n")
    print(json.dumps(rec, indent=2))
    print(f"vs v30 SOTA 40.55")


if __name__ == "__main__":
    main()
