# Spanish→Chanka Quechua MT — methodology & reproducibility guide

Canonical "how we run experiments here." Follow this for every future run so
results stay comparable and leakage-free. Progression so far: **40.55 → 46.44 ChrF**
(and GSPO climbing further). Companion result docs: `sota_push_findings.md`,
`vllm_nllb_and_gspo_findings.md`.

## 0. Benchmark & metric (NON-NEGOTIABLE)
- **Benchmark:** AmericasNLP 2021 spa→quy test, `docs/references/americasnlp_test/2021_test.{es,quy}` (1003 lines). Direction is **Spanish→Quechua** (es source → quy reference).
- **Metric:** ChrF, `sacrebleu.corpus_chrf(preds,[refs], word_order=0)`. NOT ChrF++ (w2). Published systems often report ChrF++ — not comparable.
- The test set is **eval-only, forever**. Never train, RL, tune, or select on it.

## 1. Data hygiene (the rules that bit us — follow exactly)
1. **Leakage guard everywhere.** Every corpus build dedups against the 2021 test on a normalized key (lowercase, strip punct, collapse whitespace), BOTH sides (es and quy). Verify 0 overlap and print it.
2. **RL/eval data must be UNSEEN by the model being trained.** For RL on NLLB-r2, the prompts were deduped against NLLB-r2's *exact* training corpus `clean_chanka/nllb_v2_corpus.parquet` (323k) AND the test. Training RL on data the model memorized = trivial reward, no generalization.
3. **Validation ≠ test.** For checkpoint/hyperparam selection use a held-out **val** split (`clean_chanka/rl_val_split.parquet`, 2k, excluded from RL training), then do exactly ONE final test eval of the val-selected model.
4. **Dialect matters.** Target is **Ayacucho/Chanka quy**. Drop Cuzco (apostrophe ejectives `p' t' k' q' ch'`), Central/Ancash, and Kichwa rows — rewarding wrong-dialect refs mis-steers RL. Filter validated against the gold AmNLP dev (0 false positives). See `scripts/rl/build_heldout_rl.py`.
5. **Orthography:** do NOT apply MINEDU normalization to training targets for this benchmark — the test refs are un-normalized; the penalty for normalizing is only ~0.4 ChrF and it *hurts*. (Proven in `score_orthography_penalty.py`.)

## 2. The SOTA recipe stack (each layer is reproducible)
| layer | what | result | script |
|---|---|---|---|
| Base MT | v30 = Qwen3.5-9B, broad→Chanka LoRA chain→merged | 40.55 greedy | (v25/v30 chain) |
| Diverse model | NLLB-200-1.3B + LoRA r256/α512, lr2e-4 inv-sqrt (BSC recipe) | 39.46 | `scripts/nllb/train_nllb_chanka.py` |
| **Synthetic distill** | forward-translate 200k ES mono (v30 teacher) → +198k synth → retrain NLLB (NLLB-r2) | 42.95 | `forward_translate_synth.py` + `build_nllb_v2_corpus.py` |
| **ChrF-MBR decode** | sample N=64 (T=0.5), pick max mean-pairwise sentence-ChrF over the **deduped** candidate pool (+greedy in pool) | +1.5-2.5 | `gen_candidates_*.py` + `ensemble_mbr_rerank.py` |
| **Cross-arch ensemble** | pool candidates from diverse, **comparable-quality** models → dedup-MBR | 45.01 (v30⊕NLLB-r2) | `ensemble_mbr_rerank.py` |
| **GSPO RL** (novel) | RL NLLB on held-out data, ChrF reward | **45.49 standalone / 46.44 ens** | `scripts/rl/gspo_nllb_vllm.py` |

**MBR rules that matter:** (a) **dedup the candidate pool** before consensus (+0.5 over raw) — duplicates bias the centroid; (b) ensemble members must be *diverse AND comparable quality* — a weak member (v32@34.5, MADLAD@19.78, wrong-dialect) HURTS. **Update (2026-06-01): GSPO pulled NLLB past v30, so v30 now *hurts* the NLLB ensemble (46.40 < 46.43 self-MBR) — cross-arch lever exhausted until a 2nd member is restored to parity;** (c) apostrophe suppression at decode is a free gain for NLLB; (d) **candidate-pool temperature ≈0.7, NOT 0.5** — MBR is very sensitive to pool diversity (same ckpt: T=0.5→46.14, T=0.7→46.43). Pick temperature on **val**, never test.

## 3. GSPO-on-NLLB recipe (the frontier-pusher)
First GSPO applied to an encoder-decoder NMT model. `scripts/rl/gspo_nllb_vllm.py`.
- **Reward:** sentence-ChrF vs the real (unseen, Ayacucho) reference. Sequence-level (length-normalized) GSPO importance ratio + clip + group-normalized advantage + KL-to-frozen-ref.
- **Policy:** NLLB-r2 LoRA (trainable). **Ref:** frozen copy (KL anchor).
- **Rollouts via in-process vLLM** (our NLLB port) — TRL-style: after each step push merged weights into the engine with `load_weights` (GPU→GPU, no disk). 3.2× HF (77 rollouts/s).
- **Hyperparams (working):** group-size 8–16, prompt-batch 48, lr 2e-6, clip 0.2, kl-coef 0.04, temp 1.0, max-new 64, micro-batch 48. **G=16 > G=8 at matched steps** (val 50.70 vs 50.05 @ step 200) — lower-variance advantage is quality-positive.
- **Reward design (2026-06-01 ablation, ranked on val):** plain **ChrF wins** (50.05) > chrf_brevity 50.05 ≈ chrfpp 49.94 > chrf_rt_copy 49.63. Match the reward to the *eval* metric (w0). **Round-trip/back-translation reward was built, gated, and falsified** (does not beat plain ChrF) — see findings doc.
- **Learning curve / STOP RULE:** held-out **val** ChrF peaks then DECLINES (clean run: 50.70→51.98→52.99@600→52.94→52.05@1000) while the *training* reward keeps creeping up = over-optimization. **Val-select the peak checkpoint; do NOT run to the end.** Stop early once val plateaus (saved ~6h here).
- **Eval:** dump candidates (`gen_candidates_nllb.py`, **T≈0.7**), self dedup-MBR; ensemble only with comparable-quality members; select checkpoint+decode on val, report test once.

## 4. vLLM NLLB port (for fast rollouts/offline gen) — `Sekinal/vllm` branch `add-nllb-m2m100-support`
- File `vllm/model_executor/models/m2m_100.py` (covers NLLB + M2M-100), registered in `_MULTIMODAL_MODELS`. Ported from the bart-plugin.
- **Critical fix:** M2M/NLLB is **PRE-norm** (LayerNorm before each sub-layer), not BART's post-norm — porting BART verbatim → decoder emits EOS immediately. Also: final enc/dec layer_norm, sinusoidal pos embeddings (cast to hidden dtype for bf16), NLLB src/tgt language-token tokenization (env `NLLB_SRC_LANG`/`NLLB_TGT_LANG`).
- To use: copy `m2m_100.py` into the installed vLLM `model_executor/models/`, add the `_MULTIMODAL_MODELS` registry line; run with `VLLM_ENABLE_V1_MULTIPROCESSING=0` for in-process weight sync.

## 5. Throughput / GPU-util gotchas (NLLB-1.3B has a 256k vocab)
- The logprob forward materializes `(micro_batch × seq × 256k)` logits → the OOM source. Keep micro-batch modest, but as large as fits (≈48 with vLLM at 0.20 mem-frac). Don't go to 96 with vLLM co-resident.
- **Single inner-epoch GSPO:** `old_logprob == current policy` → reuse `lp.detach()` instead of a separate no-grad policy forward (~1/3 fewer forwards, quality-identical).
- vLLM co-resident: set `gpu_memory_utilization` low (0.20) so HF training gets the rest; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Rollout (vLLM, ~5-10s) is NOT the bottleneck; the HF logprob/update phase is — optimize *that* (micro-batch, reuse-forward), not the rollout.

## 6. Ops gotchas (remote box)
- SSH is flaky → wrap commands in a retry loop; prefer instant commands; `nohup` for launches.
- **`pkill -f <name>` self-matches** the kill command's own shell → 255/kills itself. Kill by PID via `ps -eo pid,cmd | grep "[x]name" | awk '{print $1}'`.
- **vLLM `kill -9` orphans the EngineCore worker** (holds GPU) — kill those too / verify `nvidia-smi` returns to 0 MiB before relaunch.
- Models on the HF dataset repo `Thermostatic/rosettia-chanka-data`; trained models backed up to `Thermostatic/rosettia-quy-*` (private).
- Tokens (HF/DeepSeek/GitHub) are env-vars only, NEVER committed.

## 7. Models / artifacts
- Best Qwen: `outputs/merged_full_models/20260527-v30b-9b-broad-chanka-expanded-merged` (HF `Thermostatic/rosettia-quy-v30b-9b-merged`).
- Best NLLB (pre-RL): `outputs/nllb13b_v2_20260529/final` (HF `Thermostatic/rosettia-quy-nllb13b-r2-lora`).
- GSPO-NLLB: `outputs/gspo_nllb_vllm_20260601/checkpoint-1200` (45.49) → long resume `outputs/gspo_nllb_long_20260601`.
- Candidate dumps + ensemble results: `outputs/eval_mbr/*.json(l)`.
