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
- reinforcement-learning
- gspo
- grpo
- lora
- peft
- nllb
license: cc-by-nc-4.0
pipeline_tag: translation
base_model: facebook/nllb-200-1.3B
library_name: peft
---

# rosettia-quy-gspo-nllb13b-lora

**Current SOTA** for Spanish → Chanka/Ayacucho Quechua (`quy_Latn`) on the
AmericasNLP 2021 benchmark. A **LoRA adapter** for `facebook/nllb-200-1.3B`,
produced by **GSPO reinforcement learning with a ChrF verifiable reward** on top
of the synthetic-augmented NLLB-r2 SFT model.

To our knowledge this is the **first application of GSPO (Group Sequence Policy
Optimization, Zheng et al. 2025) to an encoder-decoder NMT model.**

## Results — AmericasNLP 2021 spa→quy test (1003 lines, ChrF, sacrebleu word_order=0)
| System | ChrF (w0) |
|---|---|
| Helsinki 2021 (prior task winner) | 39.40 |
| Sheffield 2023 (NLLB-3.3B ensemble) | 34.01 |
| NLLB-r2 (this adapter's SFT starting point), beam5 | 42.95 |
| **GSPO-NLLB standalone, beam5 + apostrophe suppression** | **45.49** |
| GSPO-NLLB + dedup-ChrF-MBR (self) | 46.33 |
| **GSPO-NLLB ⊕ Qwen-9B (v30) ensemble, dedup-MBR** | **46.44** |

GSPO added **+2.54 ChrF** over the SFT model and the system is **+5.89 over our
own Qwen-9B greedy baseline (40.55)** and well clear of all published systems.

## Training
- **Base / SFT:** NLLB-200-1.3B → LoRA (BSC-2024 recipe) → +198k synthetic
  (v30-forward-translated) → **NLLB-r2** (42.95). This adapter = NLLB-r2 continued
  via GSPO.
- **RL:** GSPO — sequence-level (length-normalized) importance ratio + group-relative
  advantage + KL-to-frozen-ref. **Reward = sentence-ChrF vs real references.**
- **Data:** held-out Ayacucho/Chanka pairs the SFT model never trained on (deduped
  against its exact 323k training corpus AND the test set). Dialect-filtered to
  Ayacucho only. Held-out reward climbed 42→51; generalized to the test (+2.54).
- **Infra:** rollouts via an in-process vLLM engine (we implemented NLLB support for
  vLLM) with TRL-style per-step weight sync (~3× faster than HF generate).
- **Rigor:** test never trained/tuned/selected on; separate val split for selection.

## Usage
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="spa_Latn", tgt_lang="quy_Latn")
m = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B", torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, "Thermostatic/rosettia-quy-gspo-nllb13b-lora").cuda().eval()
bos = tok.convert_tokens_to_ids("quy_Latn")
enc = tok("No sé por qué sucedió eso.", return_tensors="pt").to("cuda")
out = m.generate(**enc, forced_bos_token_id=bos, num_beams=5, max_new_tokens=128)
print(tok.batch_decode(out, skip_special_tokens=True)[0])  # -> Manam yachanichu imarayku chay pasarqa.
```
For best quality use ChrF-MBR decoding (sample ~64, dedup, pick the ChrF-consensus)
and/or ensemble with the Qwen model `Thermostatic/rosettia-quy-v30b-9b-merged`.

## Intended use & limitations
Research-grade MT for an under-served low-resource language. Validated on a single
benchmark with single-reference ChrF; native-speaker / multi-reference evaluation is
future work. Review outputs with speakers before consequential use.
