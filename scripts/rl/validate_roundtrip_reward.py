"""Decisive gate for the round-trip adequacy reward (BEFORE wiring it into GSPO).

Idea: translate candidate_quy -> spa with a reverse model, then compare that
Spanish to the ORIGINAL Spanish source. The whole adequacy comparison happens in
Spanish (high-resource) -> ChrF/embeddings are reliable; we never need a quy metric.

The signal is only useful for GSPO if it RANKS candidates the way the true
reference does. GSPO uses GROUP-RELATIVE advantage, so the metric that matters is
the WITHIN-SOURCE (per-group) Spearman correlation between round-trip-ChrF and
reference-ChrF, NOT the global correlation. We report both, plus how often the
round-trip-best candidate is also the reference-best (top-1 agreement) and the
copy-exploit rate (candidates that just echo Spanish -> high round-trip, bad quy).

Run AFTER the reward ablation frees the GPU. Zero-shot reverse first (NLLB has
real quy); if correlation is weak, train a reverse LoRA and pass --reverse-adapter.
"""
import argparse, statistics
import torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def load(base, adapter, dtype=torch.bfloat16):
    m = AutoModelForSeq2SeqLM.from_pretrained(base, torch_dtype=dtype)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    return m.to("cuda").eval()


@torch.no_grad()
def translate(model, tok, texts, src_lang, tgt_lang, n, temperature, max_new, bs=64):
    """Batched generate. n>1 => sample n per input (returns flat list len*n)."""
    tok.src_lang = src_lang
    fbos = tok.convert_tokens_to_ids(tgt_lang)
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        kw = dict(forced_bos_token_id=fbos, max_new_tokens=max_new)
        if n > 1:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=n)
        else:
            kw.update(num_beams=1)
        g = model.generate(**enc, **kw)
        out.extend(tok.batch_decode(g, skip_special_tokens=True))
    return out


def copy_rate(hyp, src):
    """Fraction of hypothesis word-types that are verbatim Spanish source words
    (the round-trip exploit: echo the source -> trivially perfect round-trip)."""
    h = set(hyp.lower().split()); s = set(src.lower().split())
    return 0.0 if not h else len(h & s) / len(h)


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def chrf(h, r):
    return sacrebleu.sentence_chrf(h, [r]).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="facebook/nllb-200-1.3B")
    ap.add_argument("--policy-adapter", required=True, help="forward spa->quy model to draw candidates from")
    ap.add_argument("--reverse-adapter", default=None, help="quy->spa reverse model; omit = NLLB zero-shot")
    ap.add_argument("--val-es", required=True)
    ap.add_argument("--val-quy", required=True)
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-sources", type=int, default=200, help="subset of val for speed")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    src = [l.strip() for l in open(args.val_es)][:args.max_sources]
    ref = [l.strip() for l in open(args.val_quy)][:args.max_sources]
    print(f"sources={len(src)}  n_cand={args.n_cand}  reverse={'LoRA:' + args.reverse_adapter if args.reverse_adapter else 'zero-shot base'}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)

    # 1) forward: sample n candidates per source
    fwd = load(args.base, args.policy_adapter)
    cands = translate(fwd, tok, src, "spa_Latn", "quy_Latn", args.n_cand, args.temperature, args.max_new)
    del fwd; torch.cuda.empty_cache()
    G = args.n_cand

    # 2) reverse: translate each candidate quy -> spa (greedy)
    rev = load(args.base, args.reverse_adapter)
    back = translate(rev, tok, cands, "quy_Latn", "spa_Latn", 1, 1.0, args.max_new)
    del rev; torch.cuda.empty_cache()

    # 3) per-candidate signals
    chrf_ref, chrf_rt, copies = [], [], []
    for i, c in enumerate(cands):
        s_i = i // G
        chrf_ref.append(chrf(c, ref[s_i]))      # TRUE quality (vs held-out reference)
        chrf_rt.append(chrf(back[i], src[s_i]))  # round-trip adequacy (Spanish-Spanish)
        copies.append(copy_rate(c, src[s_i]))

    # 4) correlations — within-source is what GSPO actually uses
    glob = spearman(chrf_rt, chrf_ref)
    within, top1_hits = [], 0
    for s_i in range(len(src)):
        sl = slice(s_i * G, s_i * G + G)
        cr, ct = chrf_ref[sl], chrf_rt[sl]
        if len(set(ct)) > 1 and len(set(cr)) > 1:
            within.append(spearman(ct, cr))
        # does round-trip pick the same best candidate as the reference?
        if cr and ct and max(range(G), key=lambda k: ct[k]) == max(range(G), key=lambda k: cr[k]):
            top1_hits += 1

    print("\n===== ROUND-TRIP REWARD VALIDATION =====")
    print(f"global Spearman(round-trip ChrF, reference ChrF) = {glob:.3f}")
    print(f"mean WITHIN-SOURCE Spearman (what GSPO ranks on)  = {statistics.mean(within):.3f}  (n={len(within)} usable groups)")
    print(f"top-1 agreement (round-trip best == reference best) = {top1_hits}/{len(src)} = {top1_hits/len(src):.1%}")
    print(f"random-baseline top-1 agreement                     = {1.0/G:.1%}")
    print(f"mean reference ChrF of candidates = {statistics.mean(chrf_ref):.2f}")
    print(f"mean round-trip ChrF of candidates= {statistics.mean(chrf_rt):.2f}")
    print(f"mean copy-rate (source echo)      = {statistics.mean(copies):.3f}  (high => copy-exploit risk; pair with anti-copy penalty)")
    # is the copy-exploit real here? correlation of copy-rate with round-trip reward
    print(f"Spearman(copy-rate, round-trip ChrF) = {spearman(copies, chrf_rt):.3f}  (positive => exploit present)")
    print("VERDICT: within-source Spearman >~0.3 and top-1 >> random => round-trip is a valid GSPO reward signal.")


if __name__ == "__main__":
    main()
