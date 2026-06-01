"""GSPO on NLLB-200 with ChrF reward, using our in-tree vLLM NLLB port for FAST
in-process rollouts (TRL/Unsloth-style: vLLM runs in the same process; after each
optimizer step we push the updated weights straight into the vLLM engine via
load_weights — GPU->GPU, no disk, no IPC).

Throughput: rollouts via vLLM (PagedAttention + continuous batching) instead of
HF generate. Training/logprobs/update stay in HF (need gradients).

Policy = NLLB-r2 LoRA (trainable). Per step: merge_adapter -> sync weights to vLLM
-> generate G rollouts -> unmerge -> ChrF reward -> GSPO update.
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # in-process engine
os.environ.setdefault("NLLB_SRC_LANG", "spa_Latn")
os.environ.setdefault("NLLB_TGT_LANG", "quy_Latn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse, random, time
from pathlib import Path


def get_vllm_model(llm):
    """Best-effort handle to the in-process worker model across vLLM layouts."""
    cands = []
    eng = getattr(llm, "llm_engine", None)
    ec = getattr(eng, "engine_core", None)
    # V1 in-process: llm.llm_engine.engine_core.engine_core.model_executor...
    inner = getattr(ec, "engine_core", None)
    for base in (inner, ec, eng):
        me = getattr(base, "model_executor", None)
        if me is None:
            continue
        dw = getattr(me, "driver_worker", None)
        # uniproc executor may store worker directly or in a list
        for w in (dw, getattr(me, "workers", [None])[0] if getattr(me, "workers", None) else None):
            if w is None:
                continue
            worker = getattr(w, "worker", w)
            mr = getattr(worker, "model_runner", None)
            m = getattr(mr, "model", None)
            if m is not None:
                return m
        # some executors expose apply_model
    # fallback via collective_rpc
    raise RuntimeError("could not locate in-process vLLM model handle")


def make_reward(kind):
    """Reward variants for ablation. NOTE: their scales differ, so compare runs
    by held-out VAL ChrF (w0), never by the raw training reward."""
    import sacrebleu

    def chrf(h, r):
        return sacrebleu.sentence_chrf(h, [r]).score

    def lenpen(h, r):
        hl, rl = max(1, len(h.split())), max(1, len(r.split()))
        return abs(hl - rl) / rl

    def reppen(h):
        w = h.split()
        return 0.0 if len(w) < 2 else (1.0 - len(set(w)) / len(w))

    if kind == "chrf":
        return lambda h, r: chrf(h, r)
    if kind == "chrfpp":
        return lambda h, r: sacrebleu.sentence_chrf(h, [r], word_order=2).score
    if kind == "chrf_brevity":
        return lambda h, r: chrf(h, r) - 20.0 * lenpen(h, r)
    if kind == "chrf_rep":
        return lambda h, r: chrf(h, r) - 30.0 * reppen(h)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="facebook/nllb-200-1.3B")
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--src-field", default="reviewed_spanish")
    ap.add_argument("--tgt-field", default="reviewed_chanka_quechua")
    ap.add_argument("--src-lang", default="spa_Latn")
    ap.add_argument("--tgt-lang", default="quy_Latn")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompt-batch", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--reward-type", default="chrf",
                    choices=["chrf", "chrfpp", "chrf_brevity", "chrf_rep"],
                    help="RL reward: chrf=sentence-ChrF(w0); chrfpp=ChrF++(w2); "
                         "chrf_brevity=ChrF - 20*|len_ratio-1|; chrf_rep=ChrF - 30*rep_rate")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-src", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--vllm-mem-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import torch, polars as pl, sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel
    from vllm import LLM, SamplingParams

    torch.manual_seed(args.seed); random.seed(args.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    bos = tok.convert_tokens_to_ids(args.tgt_lang)
    dec_start = 2  # NLLB decoder_start_token_id
    pad_id = tok.pad_token_id

    def load_policy(trainable):
        m = AutoModelForSeq2SeqLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(m, args.init_adapter, is_trainable=trainable)
        return m.to(dev)

    policy = load_policy(True); policy.train()
    ref = load_policy(False); ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    # in-process vLLM engine using our NLLB port
    print(f"[{time.strftime('%H:%M:%S')}] starting in-process vLLM engine...", flush=True)
    llm = LLM(model=args.base, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.vllm_mem_frac, max_model_len=args.max_src + args.max_new,
              max_num_seqs=args.prompt_batch * args.group_size)
    vmodel = get_vllm_model(llm)
    print(f"[{time.strftime('%H:%M:%S')}] vLLM model handle: {type(vmodel).__name__}", flush=True)

    def sync_weights_to_vllm():
        # merge LoRA into base, push HF state_dict into the vLLM model, then unmerge
        policy.merge_adapter()
        sd = {}
        for name, p in policy.named_parameters():
            if "lora_" in name:  # skip LoRA A/B (deltas already merged into base_layer)
                continue
            # PEFT wraps target modules: strip the wrapper prefix AND the
            # ".base_layer." infix so names match the HF checkpoint
            # (model.encoder...q_proj.weight, lm_head.weight, ...).
            n = name.replace("base_model.model.", "").replace(".base_layer.", ".")
            sd[n] = p.detach()
        loaded = vmodel.load_weights(list(sd.items()))
        policy.unmerge_adapter()
        return len(sd), (len(loaded) if loaded is not None else -1)

    df = pl.read_parquet(args.train_parquet)
    data = [(str(r[args.src_field]).strip(), str(r[args.tgt_field]).strip())
            for r in df.select([args.src_field, args.tgt_field]).iter_rows(named=True)
            if str(r[args.src_field]).strip() and str(r[args.tgt_field]).strip()]
    random.shuffle(data)
    reward_fn = make_reward(args.reward_type)
    print(f"RL data: {len(data)} real pairs | G={args.group_size} B={args.prompt_batch} | reward={args.reward_type}", flush=True)

    def seq_logprobs(model, src_ids, src_mask, gen_ids):
        dec_in = gen_ids[:, :-1]; labels = gen_ids[:, 1:]
        out = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=dec_in)
        logp = torch.log_softmax(out.logits.float(), -1)
        tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels != pad_id).float()
        return (tok_lp * mask).sum(-1), mask.sum(-1).clamp(min=1.0)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    G = args.group_size
    ptr = 0
    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        batch = data[ptr:ptr + args.prompt_batch]
        ptr = (ptr + args.prompt_batch) % max(1, len(data) - args.prompt_batch)
        srcs = [b[0] for b in batch]; refs = [b[1] for b in batch]

        # --- sync current policy weights into vLLM, then rollout ---
        sync_weights_to_vllm()
        prompts = [{"encoder_prompt": {"prompt": "", "multi_modal_data": {"text": s}},
                    "decoder_prompt": args.tgt_lang} for s in srcs]
        sp = SamplingParams(n=G, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new)
        t_roll = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        roll_s = time.time() - t_roll

        # build decoder sequences [dec_start, tgt_lang, <generated...>] + rewards
        texts, gen_seqs = [], []
        for o in outs:
            for c in o.outputs:
                gen = list(c.token_ids)
                texts.append(tok.decode(gen, skip_special_tokens=True))
                gen_seqs.append([dec_start, bos] + gen)
        rewards = torch.tensor([reward_fn(texts[i], refs[i // G])
                                for i in range(len(texts))], dtype=torch.float32)
        r = rewards.view(-1, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1).to(dev)

        # pad decoder seqs to rectangle; repeat-interleave sources by G
        maxlen = max(len(g) for g in gen_seqs)
        gen_ids = torch.full((len(gen_seqs), maxlen), pad_id, dtype=torch.long, device=dev)
        for i, g in enumerate(gen_seqs):
            gen_ids[i, :len(g)] = torch.tensor(g, device=dev)
        enc = tok(srcs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_src).to(dev)
        src_ids = enc.input_ids.repeat_interleave(G, 0); src_mask = enc.attention_mask.repeat_interleave(G, 0)

        # Only the FROZEN ref needs a no-grad pass. Single inner-epoch GSPO means
        # old_logprob == current policy, so we reuse the update pass's lp.detach()
        # instead of a separate policy forward (quality-identical, ~1/3 fewer fwds).
        with torch.no_grad():
            ref_lp = []
            for s in range(0, gen_ids.size(0), args.micro_batch):
                sl = slice(s, s + args.micro_batch)
                rlp, _ = seq_logprobs(ref, src_ids[sl], src_mask[sl], gen_ids[sl]); ref_lp.append(rlp)
            ref_lp = torch.cat(ref_lp)

        opt.zero_grad(); total = 0.0
        for s in range(0, gen_ids.size(0), args.micro_batch):
            sl = slice(s, s + args.micro_batch)
            lp, L = seq_logprobs(policy, src_ids[sl], src_mask[sl], gen_ids[sl])
            ratio = torch.exp((lp - lp.detach()) / L)  # ==1; gradient flows via lp
            a = adv[sl]
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a)
            kl = (lp - ref_lp[sl]) / L
            frac = gen_ids[sl].size(0) / gen_ids.size(0)
            loss = -(surr - args.kl_coef * kl).mean() * frac
            loss.backward(); total += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 5 == 0 or step == 1:
            print(f"step {step} | reward(ChrF) mean={rewards.mean():.2f} max={r.max(1).values.mean():.2f} "
                  f"| loss={total:.4f} | roll={roll_s:.1f}s tot={time.time()-t0:.1f}s "
                  f"| {len(texts)/roll_s:.0f} rollouts/s", flush=True)
        if step % args.save_steps == 0:
            d = os.path.join(args.output_dir, f"checkpoint-{step}")
            policy.save_pretrained(d); tok.save_pretrained(d)
            print(f"saved {d}", flush=True)

    policy.save_pretrained(os.path.join(args.output_dir, "final"))
    tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print("GSPO(vLLM) done ->", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
