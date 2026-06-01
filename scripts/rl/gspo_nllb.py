"""GSPO (Group Sequence Policy Optimization) for NLLB-200 seq2seq, with ChrF as a
verifiable reward — the first application (to our knowledge) of GSPO to an
encoder-decoder NMT model. Directly optimizes the eval metric for low-resource
Spanish -> Chanka Quechua.

GSPO (Zheng et al., 2025, arXiv:2507.18071): sequence-level (length-normalized)
importance ratio + clipping, with group-relative advantages — more stable than
token-level GRPO for sequence rewards like ChrF.

Throughput notes: NLLB-1.3B is small, so we push big rollout batches in bf16 with
short max_new. Policy = NLLB-r2 LoRA (continued, trainable); KL reference = frozen
NLLB-r2 (PEFT adapter disabled gives the base; we keep a frozen adapter copy).
Train on REAL parallel data only (real refs => clean ChrF reward).
"""
import argparse, json, os, random, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="facebook/nllb-200-1.3B")
    ap.add_argument("--init-adapter", required=True, help="SFT LoRA to start the policy from (NLLB-r2 final)")
    ap.add_argument("--train-parquet", required=True, help="REAL parallel data (real refs for reward)")
    ap.add_argument("--src-field", default="reviewed_spanish")
    ap.add_argument("--tgt-field", default="reviewed_chanka_quechua")
    ap.add_argument("--src-lang", default="spa_Latn")
    ap.add_argument("--tgt-lang", default="quy_Latn")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--group-size", type=int, default=8)        # G rollouts per prompt
    ap.add_argument("--prompt-batch", type=int, default=48)     # prompts per step
    ap.add_argument("--inner-epochs", type=int, default=1)      # PPO-style updates per rollout (mu)
    ap.add_argument("--micro-batch", type=int, default=64)      # seqs per fwd/bwd micro-batch
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--clip", type=float, default=0.2)          # GSPO seq-ratio clip eps
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-src", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import torch, polars as pl, sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    torch.manual_seed(args.seed); random.seed(args.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    bos = tok.convert_tokens_to_ids(args.tgt_lang)
    pad_id = tok.pad_token_id

    def load_policy(trainable):
        m = AutoModelForSeq2SeqLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(m, args.init_adapter, is_trainable=trainable)
        return m.to(dev)

    policy = load_policy(True); policy.train()
    ref = load_policy(False); ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    df = pl.read_parquet(args.train_parquet)
    data = [(str(r[args.src_field]).strip(), str(r[args.tgt_field]).strip())
            for r in df.select([args.src_field, args.tgt_field]).iter_rows(named=True)
            if str(r[args.src_field]).strip() and str(r[args.tgt_field]).strip()]
    random.shuffle(data)
    print(f"RL data: {len(data)} real pairs | G={args.group_size} B={args.prompt_batch}", flush=True)

    def seq_logprobs(model, src_ids, src_mask, gen_ids):
        # teacher-force gen_ids (which start with decoder_start + forced_bos); score tokens 1..T
        dec_in = gen_ids[:, :-1]
        labels = gen_ids[:, 1:]
        out = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=dec_in)
        logp = torch.log_softmax(out.logits.float(), dim=-1)
        tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels != pad_id).float()
        seq_lp = (tok_lp * mask).sum(-1)
        length = mask.sum(-1).clamp(min=1.0)
        return seq_lp, length

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ptr = 0
    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        batch = data[ptr:ptr + args.prompt_batch]
        ptr = (ptr + args.prompt_batch) % max(1, len(data) - args.prompt_batch)
        srcs = [b[0] for b in batch]; refs = [b[1] for b in batch]
        enc = tok(srcs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_src).to(dev)

        # ---- rollout: G samples per prompt ----
        policy.eval()
        with torch.no_grad():
            gen = policy.generate(**enc, forced_bos_token_id=bos, do_sample=True,
                                  temperature=args.temperature, top_p=args.top_p,
                                  num_return_sequences=args.group_size, max_new_tokens=args.max_new)
        policy.train()
        texts = tok.batch_decode(gen, skip_special_tokens=True)

        # rewards (ChrF vs real ref) + group-normalized advantages
        G = args.group_size
        rewards = torch.tensor([sacrebleu.sentence_chrf(texts[i], [refs[i // G]]).score
                                for i in range(len(texts))], dtype=torch.float32)
        r = rewards.view(-1, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1).to(dev)

        # expand sources to match G samples, pad gen to rectangle
        src_ids = enc.input_ids.repeat_interleave(G, 0)
        src_mask = enc.attention_mask.repeat_interleave(G, 0)
        gen_ids = gen  # (B*G, T)

        # old logprobs (snapshot) + ref logprobs
        with torch.no_grad():
            old_lp, lens = [], []
            ref_lp = []
            for s in range(0, gen_ids.size(0), args.micro_batch):
                sl = slice(s, s + args.micro_batch)
                lp, L = seq_logprobs(policy, src_ids[sl], src_mask[sl], gen_ids[sl])
                old_lp.append(lp); lens.append(L)
                rlp, _ = seq_logprobs(ref, src_ids[sl], src_mask[sl], gen_ids[sl])
                ref_lp.append(rlp)
            old_lp = torch.cat(old_lp); lens = torch.cat(lens); ref_lp = torch.cat(ref_lp)

        # ---- GSPO update(s) ----
        for _ in range(args.inner_epochs):
            opt.zero_grad()
            total = 0.0
            for s in range(0, gen_ids.size(0), args.micro_batch):
                sl = slice(s, s + args.micro_batch)
                lp, L = seq_logprobs(policy, src_ids[sl], src_mask[sl], gen_ids[sl])
                # GSPO sequence-level (length-normalized) importance ratio
                ratio = torch.exp((lp - old_lp[sl]) / L)
                a = adv[sl]
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a)
                # per-sequence KL to ref (length-normalized)
                kl = (lp - ref_lp[sl]) / L
                # weight micro-batch mean by its fraction so the sum == full-batch mean
                frac = gen_ids[sl].size(0) / gen_ids.size(0)
                loss = -(surr - args.kl_coef * kl).mean() * frac
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step()

        if step % 10 == 0 or step == 1:
            print(f"step {step} | reward(ChrF) mean={rewards.mean():.2f} max={r.max(1).values.mean():.2f} "
                  f"| loss={total:.4f} | {time.time()-t0:.1f}s/step", flush=True)
        if step % args.save_steps == 0:
            d = os.path.join(args.output_dir, f"checkpoint-{step}")
            policy.save_pretrained(d); tok.save_pretrained(d)
            print(f"saved {d}", flush=True)

    policy.save_pretrained(os.path.join(args.output_dir, "final"))
    tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print("GSPO done ->", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
