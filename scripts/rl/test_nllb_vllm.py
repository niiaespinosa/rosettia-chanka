"""Validate the vLLM NLLB (M2M-100) port: load facebook/nllb-200-1.3B in vLLM and
translate a few Spanish sentences to Chanka Quechua, comparing to HF transformers.
"""
MODEL = "facebook/nllb-200-1.3B"
SRCS = [
    "Mis abuelos eran una pareja muy cariñosa.",
    "No sé por qué sucedió eso.",
]


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from vllm import LLM, SamplingParams

    # --- HF reference (known-good) ---
    tok = AutoTokenizer.from_pretrained(MODEL, src_lang="spa_Latn", tgt_lang="quy_Latn")
    hf = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    bos = tok.convert_tokens_to_ids("quy_Latn")
    print("=== HF reference ===", flush=True)
    for s in SRCS:
        enc = tok(s, return_tensors="pt").to("cuda")
        o = hf.generate(**enc, forced_bos_token_id=bos, num_beams=5, max_new_tokens=64)
        print("HF :", repr(tok.batch_decode(o, skip_special_tokens=True)[0]), flush=True)
    del hf
    torch.cuda.empty_cache()

    # --- vLLM port ---
    print("=== vLLM port ===", flush=True)
    llm = LLM(model=MODEL, enforce_eager=True, max_model_len=512, max_num_seqs=4,
              gpu_memory_utilization=0.5, dtype="bfloat16")
    params = SamplingParams(temperature=0.0, max_tokens=64)
    prompts = []
    for s in SRCS:
        # NLLB: encoder gets the source (tokenizer prepends src_lang); decoder starts
        # with the target language token. Mirror the bart-plugin enc/dec prompt format.
        prompts.append({
            "encoder_prompt": {"prompt": "", "multi_modal_data": {"text": s}},
            "decoder_prompt": "quy_Latn",
        })
    outs = llm.generate(prompts, params)
    for s, o in zip(SRCS, outs):
        print("vLLM:", repr(o.outputs[0].text),
              "| gen_ids:", list(o.outputs[0].token_ids)[:20],
              "| prompt_ids:", list(getattr(o, "prompt_token_ids", []) or [])[:8],
              "| enc_ids:", list(getattr(o, "encoder_prompt_token_ids", []) or [])[:12],
              flush=True)


if __name__ == "__main__":
    main()
