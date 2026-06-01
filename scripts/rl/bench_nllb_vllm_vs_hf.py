"""Throughput benchmark: NLLB-200-1.3B spa->quy, HF transformers vs vLLM (our port).
Greedy, batched, on N AmericasNLP test sentences. Reports wall-time + sents/sec."""
import time

MODEL = "facebook/nllb-200-1.3B"
N = 256


def load_src():
    src = [l.strip() for l in open("docs/references/americasnlp_test/2021_test.es")
           if l.strip()][:N]
    return src


def bench_hf(src):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(MODEL, src_lang="spa_Latn", tgt_lang="quy_Latn")
    m = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    bos = tok.convert_tokens_to_ids("quy_Latn")
    bs = 32
    t0 = time.time()
    n = 0
    for i in range(0, len(src), bs):
        b = src[i:i+bs]
        enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            m.generate(**enc, forced_bos_token_id=bos, num_beams=1, max_new_tokens=64)
        n += len(b)
    dt = time.time() - t0
    print(f"[HF]   {n} sents in {dt:.1f}s = {n/dt:.1f} sents/s", flush=True)
    del m
    torch.cuda.empty_cache()


def bench_vllm(src):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, enforce_eager=True, max_model_len=512, max_num_seqs=64,
              gpu_memory_utilization=0.5, dtype="bfloat16")
    prompts = [{"encoder_prompt": {"prompt": "", "multi_modal_data": {"text": s}},
                "decoder_prompt": "quy_Latn"} for s in src]
    params = SamplingParams(temperature=0.0, max_tokens=64)
    t0 = time.time()
    llm.generate(prompts, params)
    dt = time.time() - t0
    print(f"[vLLM] {len(src)} sents in {dt:.1f}s = {len(src)/dt:.1f} sents/s", flush=True)


def main():
    src = load_src()
    print(f"benchmarking {len(src)} sentences, greedy, bf16", flush=True)
    bench_hf(src)
    bench_vllm(src)


if __name__ == "__main__":
    main()
