"""Forward-translate Spanish monolingual -> Chanka Quechua to build target-side
synthetic parallel data (BSC-2024's single biggest lever, +2.9 ChrF++).

Uses the best available translator (a merged Qwen chanka model via vLLM, greedy
for throughput) to translate a Spanish text file into quy, applies light quality
filters, and writes {spanish, chanka, source} pairs to add to the NLLB/MT corpus.
"""
import argparse
import json
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


def ok_pair(spa, quy):
    if not quy or not spa:
        return False
    cr = len(quy) / max(1, len(spa))
    if cr > 2.5 or cr < 0.3:          # length-ratio sanity
        return False
    if len(quy.split()) < 1:
        return False
    if quy.strip().lower() == spa.strip().lower():  # untranslated copy
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="merged translator model")
    ap.add_argument("--src-file", required=True, help="Spanish monolingual, one sentence/line")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--source-tag", default="synth_fwd")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="cap #sentences (0=all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    src = [l.strip() for l in open(args.src_file) if l.strip()]
    if args.limit:
        src = src[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    prompts = [build_prompt(tok, s) for s in src]

    llm = LLM(model=args.base, enforce_eager=True, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem_frac, max_model_len=1024)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new, stop=STOP))

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n_keep = n_drop = 0
    with open(args.out_jsonl, "w") as f:
        for s, o in zip(src, outs):
            quy = o.outputs[0].text.strip()
            if ok_pair(s, quy):
                f.write(json.dumps({"spanish": s, "chanka": quy, "source": args.source_tag}, ensure_ascii=False) + "\n")
                n_keep += 1
            else:
                n_drop += 1
    print(json.dumps({"src": args.src_file, "kept": n_keep, "dropped": n_drop,
                      "out": args.out_jsonl}, indent=2))


if __name__ == "__main__":
    main()
