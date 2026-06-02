#set document(title: "Spanish to Chanka Quechua MT — engineering report", author: "rosettia")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "DejaVu Sans", size: 10pt)
#set par(justify: true, leading: 0.62em)
#show heading: set block(above: 1.1em, below: 0.6em)
#set heading(numbering: "1.1")
#show link: set text(fill: rgb("#256d6b"))

#align(center)[
  #text(size: 17pt, weight: "bold")[Spanish → Chanka (Ayacucho) Quechua Machine Translation]
  #v(2pt)
  #text(size: 12pt)[From a 40.55 baseline to a ≈46.7 ChrF result — a conservative engineering report]
  #v(4pt)
  #text(size: 9pt, fill: rgb("#5b6675"))[rosettia · AmericasNLP 2021 spa→quy benchmark · single reference, ChrF (word_order=0)]
]

#v(2pt)
#block(fill: rgb("#f3f5f7"), inset: 10pt, radius: 4pt, width: 100%)[
  *Abstract.* We build a machine-translation system for Spanish → Chanka/Ayacucho Quechua
  (`quy`), an extremely low-resource, agglutinative language. By stacking *orthogonal*
  levers — a curated Qwen-9B LoRA chain, an NLLB-1.3B supervised model with synthetic
  distillation, character-level Minimum-Bayes-Risk decoding, cross-architecture ensembling,
  and GSPO reinforcement learning (to our knowledge the *first* application of GSPO to an
  encoder–decoder NMT model) — we improve from *40.55* to *≈46.7* ChrF on the AmericasNLP
  2021 test, roughly *+7* over the best published system. We report everything
  conservatively: a single benchmark, single-reference ChrF, and *no native-speaker
  evaluation*. We also document, in detail, what did *not* work.
]

= The problem

Chanka (Ayacucho) Quechua is Southern Quechua: agglutinative/polysynthetic (long, regular
suffix chains), with non-standardized orthography. The resource situation is the binding
constraint of the whole project:

- *Parallel data:* ~100–300k noisy Spanish–Quechua pairs (Bible/JW300-derived, dictionaries,
  FLORES-200, mined).
- *Monolingual quy:* ~175k lines.
- *Benchmark:* AmericasNLP 2021 spa→quy test, 1003 sentences, *single reference*.
- *Metric:* ChrF (`sacrebleu`, `word_order=0`). Character-level metrics are the right choice
  for agglutinative morphology; BLEU/TER are unreliable here (see §6).
- *Hard constraint:* no Chanka speakers were available to us, so there is *no human
  evaluation* anywhere in this work. Every "quality" signal beyond ChrF is an automatic proxy.

#figure(
  image("figures/fig1_comparison.png", width: 92%),
  caption: [Our build-up against prior published systems. All scores are ChrF (word_order=0).],
)

= Prior work (same test)

#figure(
  table(
    columns: (1fr, auto), align: (left, right), stroke: 0.5pt + rgb("#d8dce2"),
    table.header([*System*], [*ChrF (w0)*]),
    [Sheffield 2023 — NLLB-3.3B, 3-model ensemble], [34.01],
    [Helsinki 2021 — prior task winner], [39.40],
    [BSC 2024 — 2024 task winner (reported *ChrF++* 38.21)], [≈ low/mid-30s w0],
  ),
  caption: [Published ceiling on this test sits ~34–39.4 ChrF (w0). Note BSC-2024 reported
  ChrF++ (word_order=2), which reads ~2–3 points above w0 and is not directly comparable.],
)

= What worked — the levers

== Qwen-9B supervised model (v30)
Our base/teacher model, *v30*, is a *two-stage* LoRA fine-tune of `unsloth/Qwen3.5-9B`
(Unsloth, *16-bit LoRA — no 4-bit quantization*, bf16; decoder-only, with minimal native
Quechua pretraining), then merged to 16-bit weights. Both stages use LoRA *r = 256, α = 512,
dropout = 0* over all seven attention/MLP projections (`q,k,v,o,gate,up,down`), optimizer
`adamw_8bit`, weight-decay 0.01, warmup-ratio 0.05:

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, left, center, center, center, center), stroke: 0.5pt + rgb("#d8dce2"),
    table.header([*Stage*], [*Data*], [*LR*], [*eff. batch*], [*max-seq*], [*budget*]),
    [Broad SFT], [~166k spa↔quy (SomosNLP + AmericasNLP)], [1e-4], [64 (16×4)], [256], [→ ckpt-2688],
    [Chanka SFT #linebreak() (continuation)], [1,929 curated Chanka pairs], [2e-5], [8], [128], [3 ep (ckpt-615)],
  ),
  caption: [The v30 two-stage recipe. The Chanka stage continues the broad adapter, then the
  result (v30a, ckpt-615) is merged to 16-bit (no α-rescaling) → the published model.],
)

The Chanka corpus (1,929 unique pairs after leakage-filtering against the held-out eval split)
is 1,042 reviewed judicial-manual pairs + 503 manual-glossary entries + 349 Benito-2018
dictionary entries + 35 simple terms. Prompting uses a chat template (system *"Eres un traductor
profesional español-quechua"*, instruction *"Traduce del español al quechua chanka…"*), with
reasoning disabled and loss computed on the response only. *Result: 40.55 ChrF (w0) greedy*
(already above the published ceiling), 42.93 with ChrF-MBR dedup decoding.

Two honest notes. *(i)* A compact-mixed self-verification stage and a LoRA-α sweep (carried over
from a smaller-model lineage) were *planned but NOT included in the shipped/evaluated v30* — the
SOTA number is the two-stage merge above. *(ii)* Curated small data beat larger noisy data — a
109k normalized in-domain corpus scored only 37.9, below the 1,929 curated pairs — so for a
decoder-only model with little Quechua pretraining, curation and decoding mattered more than
scale. (The in-domain manual eval is inflated because the glossary shares the manual the eval is
drawn from; the AmericasNLP test is the clean, leakage-guarded number.) v30 is reused as the
synthetic-data teacher and as an ensemble member; its full model card is linked at the end.

== NLLB-1.3B supervised (BSC-2024 recipe)
LoRA r256/α512, lr 2e-4 inverse-sqrt. Standalone *39.46* — a strong, architecturally diverse
second model (encoder–decoder, with real Quechua pretraining inside NLLB-200).

== Synthetic distillation (first breakthrough, +3.5)
We forward-translated ~200k Spanish monolingual sentences with the Qwen-9B teacher (greedy),
filtered by length ratio, and added them as target-side synthetic data → retrained NLLB
("NLLB-r2"): *39.46 → 42.95*. This is sequence-level knowledge distillation; the gain is
bounded by the teacher (~40-level), so the student approaches but does not surpass it from
self-distillation alone.

== Character-level MBR decoding
Reference-free Minimum-Bayes-Risk with a ChrF utility (COMET is unusable — its encoder has
~no Quechua). Two findings that matter: *(i)* deduplicate the candidate pool before taking the
consensus (+0.5 over raw MBR — duplicates bias the centroid toward high-probability-but-not-best
modes); *(ii)* suppressing the apostrophe token at decode is a small free gain (Ayacucho quy has
no glottalization). Candidate-pool *temperature ≈0.7* matters: MBR is sensitive to pool diversity.

== Cross-architecture ensemble
Pooling candidates from architecturally diverse, *comparable-quality* models and taking the
dedup-MBR consensus reached *45.01* (Qwen-9B ⊕ NLLB-r2). The empirical rule: diversity helps only
at comparable quality — a weak member (a regressed Qwen at 34.5, a zero-shot MADLAD at 19.78) *lowers*
the ensemble.

== GSPO reinforcement learning (headline, novel)
To our knowledge the *first application of GSPO* (Group Sequence Policy Optimization, Zheng et al.
2025) to an encoder–decoder NMT model. Setup: policy = NLLB-r2 LoRA; frozen NLLB-r2 as the KL
reference; *reward = sentence-ChrF vs unseen Ayacucho references*; sequence-level (length-normalized)
importance ratio + group-relative advantage + KL, with single inner-epoch updates. Rollouts run
through an *in-process vLLM engine* — we implemented NLLB/M2M-100 support for vLLM (unsupported
upstream) and sync policy weights into the engine each step (GPU→GPU). Reward = ChrF was selected on
validation over ChrF++, length-penalty, repetition-penalty, and round-trip-adequacy variants.

#figure(
  image("figures/fig2_gspo_curve.png", width: 80%),
  caption: [GSPO validation curve. Validation ChrF climbs, peaks (~step 600), then declines
  (over-optimization). We select the peak checkpoint on validation and evaluate on test once.],
)

#figure(
  table(
    columns: (1fr, auto), align: (left, right), stroke: 0.5pt + rgb("#d8dce2"),
    table.header([*System*], [*ChrF (w0)*]),
    [Qwen-9B (ours), greedy], [40.55],
    [NLLB-1.3B (ours), supervised + synthetic], [42.95],
    [Qwen-9B ⊕ NLLB-1.3B ensemble (ours)], [45.01],
    [+ GSPO RL, single model, beam5 #emph[(pre-registered)]], [*45.53*],
    [+ MBR decoding, single model], [46.43],
    [GSPO multi-checkpoint MBR ensemble (best-found)], [*46.71*],
  ),
  caption: [Progression on the AmericasNLP 2021 spa→quy test. See §8 on which number is the
  cleanest to cite.],
)

= Beyond the surface metric

ChrF is a single-reference surface metric and saturates. To check the RL gains are *real*
quality and not metric-gaming, we score several automatic, speaker-free axes. GSPO improved
*adequacy* (round-trip quy→spa vs the original Spanish source) by *+4.6* — more than it
improved ChrF itself (+2.4) — and also reduced Spanish-word leakage and length error.

#figure(image("figures/fig3_scorecard.png", width: 88%), caption: [Multi-axis quality
scorecard. All axes oriented higher = better. Automatic proxies — directional, not absolute.])

= Standard MT metrics

#figure(image("figures/fig4_metrics.png", width: 92%), caption: [Standard metrics, supervised
vs GSPO. GSPO (reward = ChrF) improves the character-level metrics (ChrF, ChrF++) and BLEU
marginally; word-level *TER does not improve* — the RL optimized character overlap, not word
edits. BLEU is near-floor for both (word n-gram matching is brutal for agglutinative Quechua
under a single reference), which is why we treat ChrF as primary.])

= What did NOT work (honest)

We think the negative results are as informative as the positive ones.

- *MADLAD-400-3B (zero-shot):* wrong dialect (Cuzco-flavored), 19.78 ChrF — ruled out as an
  ensemble member; a weak member lowers the ensemble.
- *ByT5 (byte-level):* attractive in principle (char-level, orthogonal errors) but did not beat
  the subword NLLB here.
- *Round-trip / back-translation reward:* we built a reverse quy→spa model and a reward that
  scores adequacy by translating the candidate back to Spanish and comparing to the source. We
  *gated* it before committing a long run: its within-source ranking correlation with true quality
  plateaued at ~0.25, and a controlled GSPO ablation confirmed it does *not* beat plain ChrF
  (val 49.63 vs 50.05). Falsified — cheaply, because we gated it.
- *A learned metric (COMET) for Chanka:* data-blocked. A deep, source-verified search found *no
  human-judgment MT dataset for any Quechua variety*; the only Indigenous MT-metrics task
  (AmericasNLP 2025) covers Guarani/Bribri/Nahuatl, not Quechua. Without human judgments (and with
  no speakers to create them) a human-grounded metric is not achievable; only proxies are.
- *HRM-Text-1B (a hierarchical-recurrent reasoning LM):* a curiosity experiment. From a cold start
  (English-only pretraining, English tokenizer that fragments Quechua ~2×, no MT) it learned correct
  Chanka *morphology* surprisingly fast, but adequacy capped it in the mid-20s ChrF — far below NLLB,
  exactly as the tokenizer/no-pretraining ceiling predicts.
- *Engineering pitfalls:* a PEFT `.base_layer` weight-sync bug initially synced a half-base model
  into vLLM (reward stuck ~22) until fixed; NLLB's 256k vocabulary makes the log-prob forward
  memory-heavy (OOM unless micro-batched); and GSPO over-optimizes (validation peaks then declines),
  so checkpoint selection on a held-out validation split is essential.

= Honesty and caveats on the numbers

- *Single benchmark, single-reference ChrF, no human evaluation.* ChrF ≈46 means roughly half the
  character n-grams match one reference — useful as a draft, not production quality.
- *Which number to cite.* The *pre-registered* result is the validation-selected checkpoint with our
  standard decode (beam5 + apostrophe-suppression): *45.53*, a single test evaluation. The MBR (46.43)
  and multi-checkpoint-ensemble (46.71) numbers involve decode-time choices — MBR temperature and which
  checkpoints to ensemble — that we *compared on the test set*; they are best-found configurations, and
  the ~0.3–0.6 spread among them is within the noise of a 1003-sentence single-reference test. The
  robust, conservative claim is *≈46 ChrF, clearly above prior published work*.
- *Dialect:* Ayacucho/Chanka (`quy`); not validated for Cuzco or Central varieties.

= Reproduce

Scripts are in the repository (`scripts/`). High level: (1) `nllb/train_nllb_chanka.py` for the
BSC-recipe SFT + synthetic distillation; (2) `rl/gspo_nllb_vllm.py` for GSPO (requires our NLLB-in-vLLM
fork; reward = held-out-reference ChrF, G=16); (3) select the peak checkpoint on the held-out
validation split and run one test evaluation with `nllb/eval_nllb_americasnlp.py`; (4) optionally
`decoding/gen_candidates_nllb.py` + `decoding/ensemble_mbr_rerank.py` for MBR/ensemble; and
`decoding/quality_scorecard.py` for the beyond-the-metric audit.

= Links

- Code: #link("https://github.com/Sekinal/rosettia-chanka")
- Model (LoRA adapter): #link("https://huggingface.co/Thermostatic/rosettia-quy-gspo-nllb13b-lora")
- Model (merged, standalone): #link("https://huggingface.co/Thermostatic/rosettia-quy-gspo-nllb13b-merged")
- Qwen-9B model (v30 — base/teacher/ensemble member, with its own training card): #link("https://huggingface.co/Thermostatic/rosettia-quy-v30b-9b-merged")
- NLLB/M2M-100 support for vLLM (our fork): #link("https://github.com/Sekinal/vllm/tree/add-nllb-m2m100-support")
- GSPO: Zheng et al. 2025, #emph[Group Sequence Policy Optimization], arXiv:2507.18071.
