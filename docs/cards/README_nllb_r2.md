---
language:
- quy
- es
tags:
- translation
- machine-translation
- quechua
- chanka
- ayacucho-quechua
- low-resource
- lora
- peft
- nllb
license: cc-by-nc-4.0
pipeline_tag: translation
base_model: facebook/nllb-200-1.3B
library_name: peft
---

# rosettia-quy-nllb13b-r2-lora

A **LoRA adapter** for `facebook/nllb-200-1.3B` fine-tuned for Spanish →
**Chanka / Ayacucho Quechua (`quy_Latn`)**. This is the *round-2*, synthetic-data-
augmented adapter from the Rosettia Chanka project — the strongest NLLB member,
and half of the ensemble that reaches the project's 44.97 ChrF SOTA.

## Results — AmericasNLP 2021 spa→quy test (1003 lines, ChrF, sacrebleu `word_order=0`)

| Setting | ChrF (w0) |
|---|---|
| round-1 (no synthetic), beam-5 + apostrophe suppression | 39.46 |
| **round-2 (this adapter), beam-5 + apostrophe suppression** | **42.95** |
| round-2 + ChrF-MBR (dedup) decoding | 44.42 |
| **Ensemble (Qwen-9B v30 + this), dedup-MBR** | **44.97** ← project SOTA |

The **+3.49** jump from round-1 → round-2 comes entirely from **target-side synthetic
data** (sequence-level distillation): 198.5k Spanish monolingual sentences
forward-translated to quy by the Qwen-9B teacher, added to the real data.

## Training

- **Base:** `facebook/nllb-200-1.3B` (encoder-decoder; native `quy_Latn` support).
- **Method:** LoRA (r=256, α=512) on attention + FFN — the BSC AmericasNLP-2024
  winning recipe. LR 2e-4, inverse-sqrt schedule, bf16.
- **Data (323k pairs, 0 leakage vs the test set):**
  - 124k cleaned aggregate: AmericasNLP official + cleaned `hackathon-pln-es/spanish-to-quechua`
    + **FLORES-200** `quy_Latn` dev/devtest.
  - 198.5k **synthetic** pairs: Spanish monolingual (OPUS-100 / News-Commentary / C4)
    forward-translated to quy by `Thermostatic/rosettia-quy-v30b-9b-merged`.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="spa_Latn", tgt_lang="quy_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "Thermostatic/rosettia-quy-nllb13b-r2-lora").cuda().eval()

bos = tok.convert_tokens_to_ids("quy_Latn")
# (optional, +ChrF) forbid apostrophe tokens — Ayacucho quy orthography has none
enc = tok("Mis abuelos eran una pareja muy cariñosa.", return_tensors="pt").to("cuda")
out = model.generate(**enc, forced_bos_token_id=bos, num_beams=5, max_new_tokens=128)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

## Intended use & limitations

Research-grade MT for a low-resource language. Part of the synthetic data is
machine-generated (distilled from the Qwen teacher), so the adapter inherits some
of the teacher's biases. Review outputs with speakers before consequential use.
