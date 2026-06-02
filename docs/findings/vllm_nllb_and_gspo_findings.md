# vLLM NLLB port + GSPO-on-NLLB — findings

Two efforts toward "push the frontier further" on Spanish→Chanka Quechua (`quy`),
beyond the 45.01 ChrF decoding/ensemble SOTA.

---

## 1. NLLB-200 / M2M-100 support for vLLM (novel — first working, to our knowledge)

**Why:** NLLB rollouts/inference were HF-`generate`-bound. vLLM (and SGLang) don't
support encoder-decoder NMT models; only BART/Florence2 via the external
`bart-plugin`. CTranslate2 supports NLLB but is offline-only (no gradients/live
weights). So serving NLLB fast in vLLM required implementing the model.

**Result: WORKING + faster.** `facebook/nllb-200-1.3B` loads in vLLM 0.21 and
translates spa→quy correctly, matching HF transformers (greedy). Pushed to the
fork `github.com/Sekinal/vllm`, branch `add-nllb-m2m100-support`
(`vllm/model_executor/models/m2m_100.py`, registered in `_MULTIMODAL_MODELS`).

**Throughput (NLLB-1.3B, 256 sents, greedy, bf16, enforce_eager):**
| engine | sents/s |
|---|---|
| HF transformers | 28.8 |
| **vLLM (this port)** | **46.2 (1.6×)** |
Greedy/eager is the *worst* case for vLLM; sampling-heavy workloads (MBR n=64,
GSPO rollouts G=8) widen the gap via PagedAttention + continuous batching.

**The port = bart-plugin's BART adapted to M2M-100 (= NLLB architecture). Bugs
fixed, in order of how much they hurt:**
1. **PRE-norm vs POST-norm (the killer bug).** M2M-100/NLLB applies LayerNorm
   *before* each sub-layer (attention/FFN), then adds the residual. BART is
   post-norm. Porting BART's layer order verbatim made the decoder predict EOS +
   special tokens immediately (empty output) — the model loaded and ran with no
   error, but every layer computed wrong. Fixed both encoder & decoder layers.
   *Lesson: a model can load, run, and produce plausible-magnitude activations
   while being silently wrong; diagnose by dumping top-k logits, not just errors.*
2. **Final encoder/decoder `layer_norm`** after the stacks (pre-norm nets need it),
   replacing BART's post-embedding `layernorm_embedding` (absent in M2M). Named to
   match HF keys (`model.{encoder,decoder}.layer_norm`).
3. **Sinusoidal positional embeddings** (non-learned, non-persistent buffer, not in
   the checkpoint). Had to cast to the hidden dtype at the add site (bf16) — the
   buffer is built in fp32 → `Float vs BFloat16` matmul error otherwise.
4. **NLLB language tokens**: encoder tokenized with `src_lang` + special tokens
   (`spa_Latn … </s>`); decoder started with `[decoder_start(2), <tgt_lang>]`. The
   placeholder-count tokenizers had to match (`add_special_tokens=True`) or the MM
   placeholder alignment breaks. Langs via `NLLB_SRC_LANG`/`NLLB_TGT_LANG` env.

**Debug method that worked:** control experiment (the bart-plugin's own example ran
fine on this vLLM → platform OK, bug is mine) + staged instrumentation (encoder
output norm → decoder receives enc_hs? → top-k logits) localized it to the
within-layer norm placement.

**Caveat (general port):** NLLB needs explicit src/tgt langs, currently env-driven
(fine for a single pair; a real upstream PR needs a per-request language API). The
LM-head `/embed_scale` division is inherited from BART and is argmax-invariant for
greedy, but should be revisited for sampling/logprobs.

---

## 2. GSPO on NLLB-1.3B with a ChrF reward (novel — first GSPO on enc-dec NMT)

**Why:** GSPO (Group Sequence Policy Optimization, Zheng et al. 2507.18071, Jul
2025) is an LLM-RL method; no prior work applies it to NLLB or any encoder-decoder
NMT model (verified via search). With ChrF as a verifiable reward it's effectively
sequence-level Minimum-Risk-Training, directly optimizing the eval metric.

**Implementation** (`scripts/rl/gspo_nllb.py`, custom seq2seq RL loop, since trl's
GRPOTrainer and the project's GSPO infra are decoder-only):
- Policy = NLLB-r2 LoRA (continued, trainable); frozen NLLB-r2 = KL reference.
- Per step: sample G translations/source → ChrF reward vs the REAL reference →
  group-normalized advantage → GSPO sequence-level (length-normalized) importance
  ratio + clip + KL penalty. Train on the 124k REAL pairs (real refs = clean
  reward; the synthetic set's machine refs would reward matching v30).
- Throughput-maxed for HF generate: bf16, big rollout batch (B48×G8=384/step),
  short `max_new=64`, LoRA. ~24 rollouts/s, ~14 s/step.

**Status: SUCCESS — new SOTA.** Trained 1200+ steps via the in-process vLLM-rollout
loop (77 rollouts/s, 3.2× HF) on the clean held-out Ayacucho data. Held-out reward
climbed **40.2 → ~51** ChrF. On the AmericasNLP 2021 test:

| | ChrF (w0) | vs |
|---|---|---|
| NLLB-r2 (pre-RL) standalone beam5 | 42.95 | — |
| **GSPO-NLLB (ckpt-1200) standalone beam5** | **45.49** | **+2.54 from RL** |
| GSPO-NLLB + dedup-MBR (self) | 46.33 | +0.84 decode |
| **ensemble[v30 + GSPO-NLLB] dedup-MBR** | **46.44** | **+5.89 vs 40.55 baseline** |

**45.49 from a single model already beats the prior 45.01 decoding/ensemble SOTA; the
full stack reaches 46.44.** (v30 adds little now — GSPO-NLLB is strong enough that its
own self-MBR 46.33 ≈ the v30 ensemble 46.44.)

### Learning curve + continuing
Held-out reward (windowed) climbed monotonically and was STILL RISING at the stop:
42.0 (step 0-99) → 45.5 (200) → 47.1 (400) → 48.6 (600) → 50.3 (1000) → 51.3 (1100-1199).
Not plateaued → resumed from ckpt-1200 for up to 3000 more steps to find the ceiling.

### Methodology rigor
- Test (AmNLP 2021) NEVER trained on — single final eval.
- Checkpoint selection now uses a **held-out val split** (2,000 rows carved from the RL
  set, EXCLUDED from the resume's training) — `clean_chanka/rl_val_split.parquet`
  (+ rl_val.es/.quy). Best-on-val checkpoint → one test eval. Reported 45.49/46.44 used
  the principled latest/highest-reward checkpoint (single test eval, not cherry-picked).
Outputs verified genuine, fluent Chanka (not ChrF-gaming) — e.g. "No sé por qué
sucedió eso" → "Manam yachanichu imarayku chay pasarqa". GSPO self dedup-MBR and the
v30+GSPO ensemble (computing) are expected to push higher still.

### Validity safeguards (important)
- **No contamination:** RL ran on held-out real Ayacucho pairs that NLLB-r2 *never*
  trained on (deduped against its exact 323k training corpus), and NEVER on the
  AmericasNLP test. So the +2.54 is genuine generalization, not memorized-reference
  reproduction.
- **Right dialect:** RL data filtered to Ayacucho/Chanka quy only (dropped ~32k
  Cuzco/Ancash/Kichwa rows) — rewarding wrong-dialect refs would have mis-steered it.
- **Weight-sync correctness:** the PEFT `.base_layer.` naming bug initially synced a
  half-base Frankenstein into vLLM (reward stuck ~22); fixed → reward 40+ = true
  NLLB-r2 quality.

**Honest caveats:** (a) RL reward (ChrF) overlaps with what MBR already optimizes at
decode time → marginal gain possible; unique value is sharpening the candidate pool
for the ensemble. (b) RL is finicky at low-resource (reward hacking) — KL penalty
guards it. (c) Using vLLM to speed up GSPO *rollouts* needs an async LoRA-hot-swap
setup (merge NLLB-r2 → base, train fresh LoRA, vLLM serves base + hot-swapped
adapter) — not yet built; current loop uses HF generate.

**Next:** eval GSPO checkpoints on AmNLP; if the RL'd NLLB beats 42.95 standalone,
fold it into the v30+NLLB ensemble for a push past 45.01.

---

## Reward design study (2026-06-01) — what makes a good GSPO-MT reward

Clean 200-step ablations from NLLB-r2 (G=8, identical settings, ranked by held-out
**val ChrF w0**; never test). Baseline NLLB-r2 (no RL) val = **46.98**.

| reward | val ChrF (w0) | note |
|---|---|---|
| **chrf** (sentence-ChrF w0) | **50.05** | metric-aligned; the winner |
| chrf_brevity (−20·\|len_ratio−1\|) | 50.05 | tie — NLLB-r2 already length-calibrated, penalty inert |
| chrfpp (ChrF++ w2) | 49.94 | reward should match the *eval* metric (w0), not w2 |
| chrf_rt_copy (+0.5·roundtrip −0.3·copy) | 49.63 | round-trip reward — see below |

**Takeaway 1: plain ChrF wins; +3.07 over baseline in just 200 steps.** Reshaping the
surface metric (length/repetition penalties) doesn't help — NLLB-r2 is already well
calibrated, so those terms are inert. Match the reward to the eval metric (w0 not w2).

### Round-trip / back-translation reward — investigated, falsified (negative result)
Idea: translate candidate quy→spa with a reverse model, score adequacy by comparing
that Spanish to the *original Spanish source* — the whole comparison lives in Spanish
(high-resource) so ChrF/embeddings are reliable; no quy QE needed. Reference-free,
should resist ChrF-surface-gaming. **Paired with an anti-copy penalty** (round-trip
alone has a trivial exploit: echo the source → perfect round-trip).

We **gated it before committing** a long run, via a cheap correlation check
(`scripts/rl/validate_roundtrip_reward.py`): does round-trip ChrF *rank* candidates
the way the true reference does? The metric that matters for GSPO is the
**within-source (per-group) Spearman**, since advantages are group-relative.

| reverse model | within-src Spearman | top-1 agree (vs 12.5% random) |
|---|---|---|
| NLLB zero-shot | 0.16 | 14.5% |
| trained quy→spa LoRA, ckpt-3k/6k/9k | 0.22 / 0.28 / 0.24 | 20% / 20.5% / 24% |

Signal is **real but moderate and plateaus ~0.25** (better reverse model → better
signal, but it tops out well under the ~0.3 confidence bar), and it is **~0.29
globally correlated with ChrF itself → partly redundant**. The copy-exploit was
empirically **absent** (copy-rate ~0.01; Spearman(copy, round-trip) ~0). Ground-truth
GSPO ablation confirmed the proxy's warning: **chrf_rt_copy val 49.63 vs chrf 50.05
(−0.43)** — adding a noisy, partly-redundant adequacy term to an already-strong clean
ChrF reward dilutes the gradient rather than adding orthogonal info, and costs ~75%
more compute/step (the reverse pass). **Conclusion: round-trip does not beat plain
ChrF for this pair/setup.** Cheap to falsify because we gated it.

**Silver lining:** the trained reverse quy→spa model
(`outputs/nllb_reverse_quy2spa_20260601/checkpoint-9000`) is a real ~1-epoch
back-translation model — its useful role is **data augmentation** (back-translate the
~175k quy monolingual → synthetic (spa,quy) pairs, Edunov-style), not the reward.

**Method note (reusable):** before wiring any new RL reward into a long run, gate it
with the within-source-Spearman / top-1-agreement proxy on val candidates. It ruled
out the zero-shot reverse (0.16 ≈ noise) and correctly predicted the trained-reverse
reward would not help — saving a 1200-step run each time.

---

## Clean long GSPO run (2026-06-01) — honest results & the single-model win

Fresh GSPO from the NLLB-r2 SFT base (NOT the old ckpt-1200 — reproducibility),
plain **ChrF reward** (the ablation winner), **G=16** (lower-variance advantage),
fused-cross_entropy logprobs, vLLM mem-frac 0.30. Planned 1600 steps; **stopped
early at the val plateau**. Held-out val (rl_val, 2k) curve:

| ckpt | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| val ChrF (w0) | 50.70 | 51.98 | **52.99** | 52.94 | 52.05 ↓ |

- **G=16 > G=8** at matched steps (ckpt-200 = 50.70 vs the G=8 ablation's 50.05) —
  the lower-variance group-relative advantage is quality-positive, as predicted.
- **Over-optimization is real:** the *training* reward kept creeping up while *val*
  peaked at 600 then declined at 1000. → always val-select; do NOT run to the end.
- Val-selected checkpoint = **ckpt-600 (52.99)**.

### Test results (AmNLP 2021, ChrF w0) — and a candid read
| system | ChrF w0 |
|---|---|
| ckpt-600 standalone beam5 + apostrophe-suppress | 45.53 |
| ckpt-600 self dedup-MBR, candidate **T=0.5** | 46.14 |
| ckpt-600 self dedup-MBR, candidate **T=0.7** | **46.43** |
| ckpt-600 ⊕ v30 ensemble (T=0.7) | 46.40 |
| ckpt-800 self dedup-MBR (T=0.7) | 46.55 |
| prior SOTA (old ckpt-1200 ⊕ v30 ensemble) | 46.44 |

**Honest calibration (don't oversell):** strictly val-selected → ckpt-600 self-MBR
**46.43 ≈ the prior 46.44** — a **reproduction/tie, not a clear numeric SOTA**.
ckpt-800's 46.55 is +0.11 but (a) within single-reference noise on 1003 sentences
and (b) **test-selected** (val said ckpt-600 ≥ ckpt-800). So: we matched the frontier,
we did not meaningfully move it.

**The real win is qualitative — single model now equals the prior ensemble.** The
prior 46.44 needed a **9B Qwen (v30) + 1.3B NLLB** cross-arch ensemble; we now get the
same ~46.4 from **one 1.3B NLLB**. And crucially **v30 now HURTS** the ensemble (46.40 <
46.43 self-MBR) — GSPO pulled NLLB past v30, so the "diverse AND comparable quality"
condition broke and the cross-arch lever is **exhausted** until a second member is
brought back up to par. Simpler, ~7× smaller, equal quality, fully reproducible.

### MBR temperature lesson (reusable)
MBR quality is **very sensitive to candidate-pool diversity → temperature**:
T=0.5 → 46.14, T=0.7 → 46.43 (+0.29) on the *same* checkpoint. Generate MBR pools at
**T≈0.7** (the `_n64` recipe), not 0.5. And **select T on val, never test.**

### Where the frontier actually sits now
We're likely near the ceiling single-reference ChrF can *show* on this benchmark
(low-50s even for humans; we're at ~46.4 test / ~53 val). Further single-ref-ChrF
chasing of the same levers = diminishing returns / noise. Real gains need a *different*
lever: (a) a second ensemble member restored to parity (GSPO the Qwen, or ByT5);
(b) **backtranslation data** from the reverse quy->spa model we built; or (c) accept the
plateau. Net project arc: **40.55 -> ~46.5 ChrF (+~6), ~+7 over best published (39.40).**
