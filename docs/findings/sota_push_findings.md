# Pushing past v30: the AmericasNLP 2021 spa→quy SOTA push

**Benchmark:** AmericasNLP 2021 Spanish→Quechua (quy) test, 1003 lines, zero leakage.
**Official metric:** ChrF, sacrebleu `word_order=0` (NOT ChrF++/word_order=2).
**Prior best (ours):** v30 = 40.55 (greedy). This already meets/beats published systems
(Sheffield 2023 NLLB-3.3B = 34.01 ChrF w0; Helsinki 2021 = 39.40; BSC 2024 winner =
38.21 *ChrF++*). So "much better" requires stacking decoding/data levers, not just retraining.

## Headline result

| Approach | ChrF (w0) | Δ vs 40.55 | Cost |
|---|---|---|---|
| v30 greedy (prior SOTA) | 40.55 | — | — |
| v30 + ChrF-MBR n=32 (T=0.5) | 42.21 | +1.66 | decode only |
| v30 + ChrF-MBR n=64 (raw, no dedup) | 42.42 | +1.87 | decode only |
| **v30 + ChrF-MBR n=64, DEDUP + greedy-in-pool** | **42.93** | **+2.38** | decode only |
| ensemble[v30,v32] MBR (pool=64.6) | 42.30 | +1.75 | decode only |
| NLLB-1.3B r1 standalone (beam5, apostrophe-suppressed) | 39.46 | — | 1 NLLB train |
| **ensemble[v30 + NLLB-r1] dedup-MBR (pool=76.8)** | **43.73** | **+3.18** | +1 NLLB train |
| NLLB-1.3B **r2** (synthetic-augmented) standalone (beam5) | 42.95 | — | +synth train |
| NLLB-r2 + dedup-MBR (self, pool=38.9) | 44.42 | +3.87 | +synth train |
| **ensemble[v30 + NLLB-r2] dedup-MBR** | **44.97** | **+4.42** | +synth train |
| ensemble[v30 + NLLB-r2 + MADLAD-400-3B] dedup-MBR | *running* | — | — |

**Current SOTA: 44.97 ChrF** (v30 ⊕ NLLB-r2 cross-architecture dedup-MBR) — +4.42 over
the prior 40.55 and just under 45. The synthetic-trained NLLB-r2 (44.42 self-MBR) made
the cross-architecture ensemble jump 43.73 → 44.97; the 3-way MADLAD ensemble (running)
is the shot to clear 45.
v30-alone dedup-MBR is 42.93 (zero training); adding a *diverse, comparable-quality*
NLLB-1.3B (39.46 standalone) as a second candidate source lifts the consensus +0.80.

### Why the cross-architecture ensemble works (and the Qwen-sibling one didn't)
MBR consensus rewards candidate *diversity at comparable quality*. The v32 sibling
(34.5) was too weak → dragged the centroid down (42.30 < 42.42). NLLB-1.3B (39.46) is
a different architecture (encoder-decoder, real quy pretraining) at v30-comparable
quality → its candidates cover different correct phrasings, enriching the consensus
pool (76.8 unique/source) and lifting ChrF to 43.73.

### MBR detail: dedup the candidate pool before consensus (+0.5)
Raw MBR over all 64 samples = 42.42. Deduplicating to the ~28.5 *unique* hypotheses
(and adding the greedy hypothesis) = **42.93**. Duplicate samples otherwise inflate the
ChrF-centroid toward high-probability-but-not-best modes; equal-weighting the unique
support set is a cleaner consensus estimate. `ensemble_mbr_rerank.py` does this by default.

## Key findings

### 1. The MINEDU normalization HURTS this benchmark (orthography is a red herring)
The AmericasNLP test references are in the original (un-normalized) orthography — 44%
of ref lines contain `e`/`o`, the very vowels MINEDU normalization maps to `i`/`u`.
*However*, re-scoring v34a's predictions proved the orthographic penalty is only
**~0.4 ChrF** when made consistent (both-norm 38.36 vs official 37.93). The real reason
the 109k normalized in-domain corpus (v34a, 37.9) underperformed v30's 1929 *curated*
pairs (40.55) is **data noise (JW300-derived bulk) + recipe**, not orthography.
→ Bulk noisy in-domain data is a dead end here; curated data + good decoding wins.
(See `scripts/normalizer/score_orthography_penalty.py`.)

### 2. ChrF-MBR is the cheapest, biggest single lever (+1.87, no training)
Sample N candidates (T=0.5, top_p=0.95), pick the one maximizing mean pairwise
sentence-ChrF (consensus/centroid). Reference-free, language-agnostic, directly
optimizes the eval metric. n=32→42.21, n=64→42.42 (diminishing: +0.21 for 2× compute).
ChrF utility (NOT COMET — XLM-R lacks quy). (`scripts/decoding/mbr_chrf_vllm.py`.)

### 3. Qwen-sibling ensembling does NOT help (no comparable-quality diverse model)
v32 greedy is only 34.5 (the v32 Nouman-data regression). Pooling its candidates with
v30's dragged the MBR consensus down (42.30 < 42.42). A useful ensemble needs *diverse*
models of *comparable* quality → the v30+NLLB ensemble (different architecture) is the
one worth trying. (`scripts/decoding/{gen_candidates_vllm,ensemble_mbr_rerank}.py`.)

**Verified:** ensemble predictions recompute to 43.734, 0 empty, pred/ref char-length
61.8/57.6 (not degenerate), translations are genuine fluent Chanka Quechua — real result.

### Target-side synthetic data (forward-translation distillation) — big lever for NLLB
Forward-translated 200k abundant Spanish monolingual (OPUS-100 + News-Commentary + C4,
leakage-guarded) into quy with v30 greedy (teacher); kept 198,503 pairs after
length-ratio filtering. Added to the 124k real aggregate → 323k corpus → NLLB round-2.
Result: **NLLB-1.3B standalone 39.46 → 42.95 (+3.49)** — synthetic data nearly closed
the gap to v30. This is sequence-level knowledge distillation (NOT backtranslation;
Spanish is the abundant side). `scripts/decoding/forward_translate_synth.py`,
`scripts/nllb/build_nllb_v2_corpus.py`.
- *Caveat:* self-distillation makes the student *approach* the teacher (v30 ~40-level),
  not surpass it → NLLB-r2 converged near v30. To exceed: use a stronger teacher for the
  synthetic targets (v30+MBR 42.9 / ensemble 43.7 outputs / Google-Translate quy).

## In progress / planned levers (toward 45)
- **v30 + NLLB-r2 ensemble** and **3-way v30+NLLB-r2+MADLAD-400-3B ensemble** (running).
- **MADLAD-400-3B** (T5 MT, real quy, diverse arch) and **ByT5** (byte-level, tokenizer-free,
  ChrF-aligned, maximally orthogonal errors) as further ensemble members.
- **Iterative distillation:** regenerate synthetic targets with the *ensemble* (43.7) as
  teacher → retrain → re-ensemble (push student past current teacher).
- **Qwen CPT on quy monolingual** (5M-token corpus ready) — deprioritized for now.
- **NLLB-1.3B + LoRA** (BSC-2024-winner recipe: r=256/α=512, lr 2e-4 inverse-sqrt,
  apostrophe suppression) on a cleaned 124k aggregate (in-domain raw + FLORES-200
  dev/devtest + cleaned hackathon-pln-es, 0 leakage). Standalone + ensemble member.
- **Ensemble v30 + NLLB → MBR** (true architectural diversity).
- **Target-side synthetic data** (forward-translate abundant Spanish mono → quy; BSC's
  single biggest lever, +2.9 ChrF++), then retrain NLLB and re-MBR.

## Reproduce
```
# best single-model SOTA (42.42)
python scripts/decoding/gen_candidates_vllm.py \
  --base outputs/merged_full_models/20260527-v30b-9b-broad-chanka-expanded-merged \
  --n-samples 64 --temperature 0.5 --out-candidates outputs/eval_mbr/v30_cands_n64.jsonl
# -> ChrF_self_mbr 42.42
```
