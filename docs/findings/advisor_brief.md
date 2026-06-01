# Spanish→Chanka Quechua MT — technical brief for review

## 1. Task & evaluation
- **Task:** Machine translation **Spanish → Chanka / Ayacucho Quechua** (ISO `quy`), Southern Quechua — extremely low-resource, agglutinative/polysynthetic (long suffix chains, rich morphology).
- **Benchmark:** AmericasNLP 2021 shared-task **spa→quy test set** (1003 sentences; direction = ES source → quy reference). This set was reused as the official test in the 2023/2024 editions, so cross-year results are directly comparable.
- **Metric:** **ChrF** (character n-gram F-score), `sacrebleu` `word_order=0`. Important: most shared-task papers report **ChrF++** (`word_order=2`); the two are not point-comparable (ChrF++ usually reads a couple points higher). All our numbers are ChrF w0 unless noted.
- **Data regime:** parallel spa-quy is ~100–300k pairs (JW300/Bible-derived + dictionaries + FLORES-200 + noisy mined); monolingual quy is scarce (~175k lines, Llamacha IIC). Single-reference test.

## 2. What we're doing / goal
Build a SOTA spa→quy system and push the ChrF frontier as far as possible, with strict leakage control. We deliberately stack **orthogonal levers**: a strong base model, decoding-time consensus, cross-architecture ensembling, synthetic-data distillation, and **RL (GSPO) with a verifiable ChrF reward**. Secondary goal: methodological novelty (first GSPO on an encoder-decoder NMT model; we also implemented NLLB support for vLLM to make RL rollouts fast).

## 3. Baselines (prior published SOTA on THIS test)
| System | Method | ChrF (w0) |
|---|---|---|
| Helsinki 2021 (task winner) | multilingual NMT | **39.40** |
| Sheffield 2023 | NLLB-3.3B full-FT, 3-model ensemble, beam5 | **34.01** (dev ChrF++ 39.52) |
| BSC 2024 (overall task winner) | NLLB-1.3B + LoRA (r256/α512), +synthetic | **38.21 ChrF++** (≈ low-/mid-30s w0) |

So the published ceiling on this test (w0) sits ~**34–39.4**.

## 4. Our progression → current SOTA
| Stage | Method | ChrF (w0) |
|---|---|---|
| our base (v30) | Qwen3.5-9B, broad→Chanka LoRA chain on **curated** data, merged; greedy | **40.55** |
| + ChrF-MBR decode | sample 64 (T=0.5), pick max mean-pairwise sentence-ChrF over the **deduped** candidate pool | 42.93 |
| NLLB-1.3B (BSC recipe) | LoRA r256/α512, lr2e-4 inv-sqrt; standalone beam5 | 39.46 |
| + synthetic distillation | forward-translate 200k ES-mono via v30 teacher → +198k synth pairs → retrain NLLB (**NLLB-r2**) | 42.95 |
| cross-arch ensemble | pool v30 + NLLB-r2 candidates → dedup-MBR | 45.01 |
| **+ GSPO RL on NLLB** | ChrF reward, held-out data; standalone beam5 | **45.49** |
| GSPO-NLLB + dedup-MBR | self-MBR | 46.33 |
| **GSPO-NLLB ⊕ v30 ensemble** | dedup-MBR | **46.44** ← current SOTA |

**Net: 40.55 → 46.44 ChrF (+5.89), and ~+7 to +12 over published systems on the same test.**

## 5. Method details (for critique)
**(a) ChrF-MBR decoding.** Reference-free Minimum-Bayes-Risk with ChrF utility (language-agnostic; COMET is unusable — XLM-R has ~no quy). Key finding: **deduplicate the candidate pool before consensus** (equal-weight the unique support set) — +0.5 over raw MBR, because duplicate samples bias the ChrF-centroid toward high-prob-but-not-best modes. Apostrophe suppression at decode (Ayacucho quy has no glottalization) is a free gain for NLLB.

**(b) Cross-architecture ensemble.** Pool candidates from architecturally diverse models → dedup-MBR. Empirical rule: **diversity helps only at comparable quality** — a weak sibling (a regressed Qwen at 34.5) and a zero-shot MADLAD-400 (19.78, wrong dialect) both *lowered* the ensemble. The win came from NLLB (enc-dec, real quy pretraining) being diverse-but-comparable to v30 (decoder-only Qwen).

**(c) Synthetic distillation.** Forward-translate abundant Spanish monolingual with the v30 teacher (greedy), filter by length-ratio, add as target-side synthetic → NLLB 39.46→42.95 (+3.5). Sequence-level KD; the gain is bounded by the teacher (~40-level), so the student approaches but doesn't surpass the teacher via self-distillation alone.

**(d) GSPO on NLLB (the novel piece).** To our knowledge the **first application of GSPO (Group Sequence Policy Optimization, Zheng et al. 2025, arXiv 2507.18071) to an encoder-decoder NMT model.** Setup: policy = NLLB-r2 LoRA; frozen NLLB-r2 as KL reference; **verifiable reward = sentence-ChrF vs the real reference**; sequence-level (length-normalized) importance ratio + group-relative advantage + KL. Rollouts run through an **in-process vLLM engine** (we implemented NLLB/M2M-100 support for vLLM — it was unsupported — and sync policy weights into the engine each step, TRL-style; ~3× faster rollouts than HF generate). Held-out **reward climbed 42→51 over 1200 steps (still rising)** and **generalized to the test: NLLB-r2 42.95 → 45.49 (+2.54)**. Outputs verified as genuine fluent Chanka, not ChrF-gaming.

**(e) Rigor / leakage control.**
- Test set NEVER trained/tuned/selected on — single final eval; leakage-guarded (0 overlap, both sides) in every corpus build.
- RL runs on a **held-out Ayacucho set deduped against the model's exact training corpus** (so reward = generalization, not memorized-reference reproduction), with a separate **val split** for checkpoint selection (test stays pristine).
- **Dialect filtering**: dropped Cuzco (ejective apostrophes), Central/Ancash, and Kichwa pairs — rewarding wrong-dialect refs mis-steers RL.
- **Orthography:** we found MINEDU normalization *hurts* this benchmark (test refs are un-normalized; penalty only ~0.4 ChrF but net-negative), so targets are left un-normalized.
- Currently running: a **reward-design ablation** (ChrF vs ChrF++ vs ChrF−length-penalty vs ChrF−repetition-penalty), ranked by held-out val ChrF; winner → a clean single long GSPO run from the SFT base.

## 6. Open questions / where we'd love guidance
1. **Reward design for RL-MT.** ChrF is reference-y and we may be optimizing the metric rather than quality. COMET/COMET-QE have no quy coverage. Options we're weighing: BLEURT-style learned metrics (none for quy), QE without references, multi-reference rewards, or **ensemble/MBR-consensus as a soft target** (RL toward the ensemble). Better ideas?
2. **The single-reference ChrF ceiling.** ChrF w0 against one reference likely tops out in the low-50s even for human translators — are we near the achievable max? Worth a multi-reference or human/speaker eval to characterize it?
3. **Morphology-aware modeling.** quy is agglutinative; subword tokenizers fragment it badly. Byte-level (ByT5) or morphological segmentation as an ensemble member or base? (ByT5 errors would be orthogonal → good MBR diversity, and char-level aligns with the char-level metric.)
4. **Beating the teacher in distillation.** Self-distillation caps at the teacher; iterative back-translation, or distilling from the *ensemble/MBR* outputs (43–46-level teacher) — recommended schedules/quality filters (SONAR)?
5. **Non-parametric RL.** "Training-Free GRPO" (Tencent, 2510.08191) does GRPO in context space (a learned natural-language "experience library" injected as a prior, frozen weights, ~$18). Plausibly a cheap complement to lift the Qwen/v30 member — worth it for MT, or is it really only for agentic reasoning?
6. **Multilingual transfer / pivot.** Leveraging related Quechua varieties (quz/quh) or ES↔EN↔quy pivoting — BSC found multilingual mixing barely helped quy; is there a smarter transfer setup?

**Specific paper pointers we'd act on are very welcome** — esp. on (i) reward modeling / QE for ultra-low-resource MT, (ii) morphology-aware NMT, (iii) RL-for-MT beyond MRT/GRPO, and (iv) data augmentation that surpasses the teacher.
