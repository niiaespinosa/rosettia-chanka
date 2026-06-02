---
license: cc-by-4.0
language:
- es
- quy
- qu
task_categories:
- translation
tags:
- quechua
- chanka
- ayacucho-quechua
- spanish
- parallel-corpus
- judicial
- rosettia
pretty_name: RosettIA Chanka Quechua — Judicial Parallel Data (redistributable subset)
---

# RosettIA Chanka Quechua — Judicial Parallel Data

Spanish ↔ **Chanka / Ayacucho Quechua (`quy`)** parallel data for the
[RosettIA project](https://github.com/Sekinal/rosettia-chanka).

> ## ⚠️ Scope of this public release (please read)
> This repository **only hosts the data we can redistribute cleanly** — the
> Spanish–Chanka pairs and glossary extracted from a Peruvian Ministry of Culture
> manual that **explicitly permits reproduction with attribution**.
>
> Our models were *also* trained on broad, synthetic, and reinforcement-learning
> corpora built from **third-party sources with restrictive, viral, or unconfirmed
> licenses** (JW300-derived AmericasNLP data, SomosNLP, FLORES-200, a published
> dictionary, web-mined text, C4). **We do not redistribute any of that here.**
> It is fully documented below, with source links, licenses, and the scripts to
> reconstruct it from the original providers under their own terms. See
> *"Data used but NOT redistributed here."*

## What is included

All files are derived from a single, explicitly-redistributable source (see License & credits):

| File | Rows | Content |
|---|---:|---|
| `clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet` | 1,055 | Reviewed Spanish–Chanka judicial/administrative pairs (slash-alternatives split) |
| `clean_chanka/manual_quechua_chanka_parallel_reviewed.parquet` | 1,012 | Reviewed extraction rows with decisions/flags |
| `clean_chanka/manual_quechua_chanka_parallel_alternative_splits.parquet` | 81 | Rows created from reviewed slash-alternative pairs |
| `clean_chanka/manual_quechua_chanka_glossary_entries.parquet` | 423 | Glossary entries from the manual's vocabulary sections |
| `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet` | 219 | Simple glossary term pairs |

Columns: `reviewed_spanish`, `reviewed_chanka_quechua` (+ review/flag metadata).
Domain: **judicial / administration of justice** (not general-domain Chanka). Provenance
notes are in `metadata/`.

## License & credits

**License: `CC-BY-4.0`** (attribution required).

All included data is extracted from:

> **Ardito Vega, W. (2014).** *Manual para el empleo del Quechua Chanka en la administración
> de justicia.* Ministerio de Cultura del Perú, 1ª ed., Lima, abril 2014. ISBN 978-612-4126-20-8.
> Source PDF: https://repositorio.cultura.gob.pe/handle/CULTURA/51

The manual states (p. 3): *"Se permite la reproducción de esta obra siempre y cuando se cite
la fuente."* (Reproduction is permitted provided the source is cited.) We therefore release the
**extracted/derived** parallel pairs and glossary (not the PDF) under CC-BY-4.0, and require
attribution to:

- **Content author:** Wilfredo Ardito Vega
- **Collaborators:** Eber Llacctarimay Quispe, Cinthya Palomino Córdova, Juan Galiano Román,
  Apolinario Ciriaco Saldívar Bolívar, Clodomiro Landeo Lagos, Salvador Alvarado Tovar,
  Carlos Andrés Vera Vásquez
- **Editor:** Mónica Hidalgo Cornejo · **Publisher:** Ministerio de Cultura del Perú

If you use this data, please cite the manual above **and** the RosettIA project.

## Data used but NOT redistributed here

Our models were trained on additional corpora that we **cannot redistribute** (prohibited,
viral, non-commercial, or unconfirmed licenses). They are **not** in this repository. To
reproduce our training data, fetch each from its original provider under its own terms using
the build scripts in the [GitHub repo](https://github.com/Sekinal/rosettia-chanka) (`scripts/`).

| Source (used for) | Link | License / status | Why not hosted here |
|---|---|---|---|
| **AmericasNLP 2024 ST1 quy** (broad + NLLB + RL data) — **JW300-derived** | [github.com/AmericasNLP/americasnlp2024](https://github.com/AmericasNLP/americasnlp2024); [JW300 paper](https://aclanthology.org/P19-1310/) | **Not redistributable** — JW300 was distributed without the copyright holder's (Watch Tower) permission and withdrawn | Prohibited |
| **SomosNLP `spanish-to-quechua`** (broad SFT) | [HF dataset](https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua) | **No license stated** (mixes Tatoeba CC-BY, biblical, AmericasNLP) | No granted right |
| **FLORES-200** (inside NLLB corpus) | [facebook/flores](https://huggingface.co/datasets/facebook/flores) | CC-BY-SA-4.0 (viral share-alike; provider discourages re-hosting eval data) | Avoid share-alike entanglement + eval-integrity |
| **Diccionario Básico Quechua Chanka** (Benito Zuasnabar, 2018) | published dictionary | Copyrighted / unconfirmed | No open license |
| **nouman-10/MT-SharedTask** incl. *Le Petit Prince* (v31/v32 experiments) | [github](https://github.com/nouman-10/MT-SharedTask) | No repo license; *Little Prince* still under copyright | Not redistributable |
| **RunaSimi.de** Ayacucho vocab (v31 experiment) | [runasimi.de](https://www.runasimi.de/) | © Philip Jacobs — non-commercial + attribution | Non-commercial |
| **Synthetic targets** (NLLB distillation): Spanish from **C4 / OPUS** → quy by our model | [allenai/c4](https://huggingface.co/datasets/allenai/c4) (ODC-BY), [OPUS](https://opus.nlpl.eu/) | quy = our model output; Spanish carries C4 ODC-BY + Common Crawl ToS | Avoid hosting C4-derived web text |

(Note: the experimental v31/v32, v34-normalizer, broad-ablation, and `eval_mbr/` result files are
also not part of this release — this dataset is scoped to the **data actually used**, and only the
cleanly-licensed part of it.)

## Citation

```bibtex
@misc{rosettia_chanka_data,
  title  = {RosettIA Chanka Quechua — Judicial Parallel Data},
  author = {RosettIA project},
  note   = {Derived from: Ardito Vega, W. (2014), Manual para el empleo del Quechua Chanka
            en la administración de justicia, Ministerio de Cultura del Perú,
            ISBN 978-612-4126-20-8},
  url    = {https://github.com/Sekinal/rosettia-chanka},
  year   = {2026}
}
```
