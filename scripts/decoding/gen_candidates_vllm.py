"""Generate N candidate translations per AmericasNLP source from one model and
dump them to jsonl ({idx, source, reference, candidates:[...]}). Feeds the
ensemble MBR reranker (ensemble_mbr_rerank.py), which pools candidates across
models. Also reports this model's own greedy ChrF and self-MBR ChrF for context.
"""
import argparse
import json
import time
from pathlib import Path

SYSTEM = 'You are a careful Spanish→Quechua (Chanka/Ayacuchana variety) translator.'
INSTR = 'Traduce el siguiente texto del español al quechua chanka. Responde solo con la traducción.'
STOP = ['<|im_end|>', '<|endoftext|>']


def build_prompt(tok, s):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{INSTR}\n\nEspañol: {s}"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def mbr_pick(cands, chrf_fn):
    if len(cands) <= 1:
        return cands[0] if cands else ""
    best, bs = cands[0], -1.0
    for i, h in enumerate(cands):
        sc = sum(chrf_fn(h, o) for j, o in enumerate(cands) if j != i) / (len(cands) - 1)
        if sc > bs:
            best, bs = h, sc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--lora-rank", type=int, default=512)
    ap.add_argument("--test-es", default="docs/references/americasnlp_test/2021_test.es")
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--out-candidates", required=True)
    args = ap.parse_args()

    import sacrebleu
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    src = [l.strip() for l in open(args.test_es).read().strip().split("\n")]
    ref = [l.strip() for l in open(args.test_quy).read().strip().split("\n")]
    assert len(src) == len(ref)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    prompts = [build_prompt(tok, s) for s in src]

    kw = dict(model=args.base, enforce_eager=True, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem_frac, max_model_len=1024)
    lora_req = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        kw.update(enable_lora=True, max_lora_rank=args.lora_rank)
        lora_req = LoRARequest("adapter", 1, args.adapter)
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.base} adapter={args.adapter}", flush=True)
    llm = LLM(**kw)
    gkw = {"lora_request": lora_req} if lora_req else {}

    greedy = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new, stop=STOP), **gkw)
    g_preds = [o.outputs[0].text.strip() for o in greedy]

    sampled = llm.generate(prompts, SamplingParams(temperature=args.temperature, top_p=args.top_p,
                           n=args.n_samples, max_tokens=args.max_new, stop=STOP), **gkw)
    cand_lists = [[c.text.strip() for c in o.outputs if c.text.strip()] for o in sampled]

    def s_chrf(a, b):
        return sacrebleu.sentence_chrf(a, [b]).score

    Path(args.out_candidates).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_candidates, "w") as f:
        for i, (s, r, cs) in enumerate(zip(src, ref, cand_lists)):
            f.write(json.dumps({"idx": i, "source": s, "reference": r,
                                "greedy": g_preds[i], "candidates": cs}, ensure_ascii=False) + "\n")

    def chrf0(P):
        return round(sacrebleu.corpus_chrf(P, [ref], word_order=0).score, 3)
    self_mbr = [mbr_pick(cs, s_chrf) if cs else "" for cs in cand_lists]
    print(json.dumps({"base": args.base, "n": args.n_samples,
                      "ChrF_greedy": chrf0(g_preds), "ChrF_self_mbr": chrf0(self_mbr),
                      "out": args.out_candidates}, indent=2))


if __name__ == "__main__":
    main()
