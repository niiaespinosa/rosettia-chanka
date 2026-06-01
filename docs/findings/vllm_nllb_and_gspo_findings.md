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

**Status: running.** Smoke test passed end-to-end (reward/advantage/loss/backward).
Per-batch reward is noisy (different sentences each step) — the real signal is
evaluating a checkpoint on the AmericasNLP test (greedy + dedup-MBR) vs NLLB-r2's
42.95 / 44.42. Pending.

**Honest caveats:** (a) RL reward (ChrF) overlaps with what MBR already optimizes at
decode time → marginal gain possible; unique value is sharpening the candidate pool
for the ensemble. (b) RL is finicky at low-resource (reward hacking) — KL penalty
guards it. (c) Using vLLM to speed up GSPO *rollouts* needs an async LoRA-hot-swap
setup (merge NLLB-r2 → base, train fresh LoRA, vLLM serves base + hot-swapped
adapter) — not yet built; current loop uses HF generate.

**Next:** eval GSPO checkpoints on AmNLP; if the RL'd NLLB beats 42.95 standalone,
fold it into the v30+NLLB ensemble for a push past 45.01.
