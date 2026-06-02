---
language: [quy, es]
tags: [translation, quechua, chanka, ayacucho-quechua, nllb, gspo, low-resource]
license: cc-by-nc-4.0
pipeline_tag: translation
base_model: facebook/nllb-200-1.3B
---
# rosettia-quy-gspo-nllb13b-merged (Spanish → Chanka/Ayacucho Quechua)

Standalone **merged** NLLB-200-1.3B fine-tuned with **GSPO reinforcement learning** for spa→quy.
This is `facebook/nllb-200-1.3B` with the GSPO LoRA merged in — load directly, no PEFT needed.

**ChrF 45.53 (w0), standalone**, AmericasNLP 2021 spa→quy test. Full results, figures, scorecard,
methodology, and honest caveats (which number is pre-registered vs best-found) are in the adapter
repo: https://huggingface.co/Thermostatic/rosettia-quy-gspo-nllb13b-lora

### Usage
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("Thermostatic/rosettia-quy-gspo-nllb13b-merged", src_lang="spa_Latn", tgt_lang="quy_Latn")
m = AutoModelForSeq2SeqLM.from_pretrained("Thermostatic/rosettia-quy-gspo-nllb13b-merged", torch_dtype=torch.bfloat16).cuda().eval()
bos = tok.convert_tokens_to_ids("quy_Latn")
enc = tok("No se por que sucedio eso.", return_tensors="pt").to("cuda")
print(tok.batch_decode(m.generate(**enc, forced_bos_token_id=bos, num_beams=5, max_new_tokens=128), skip_special_tokens=True)[0])
```

Research-grade; single-benchmark single-reference ChrF; no human evaluation. See the adapter repo
for limitations.

## Authors & contributions

A two-person SomosNLP hackathon project:

- **Estefanía Espinosa Fernández** — data curation, and the initial Qwen3.5 LoRA experiments
  (comparing DoRA, rsLoRA and LoRA, and exploring data mixes).
- **Irving Ernesto Quezada Ramírez** ([irvingernesto.com](https://irvingernesto.com)) — the
  subsequent modeling through the final system: synthetic-data distillation, the NLLB pipeline,
  GSPO reinforcement learning, decoding/ensembling, evaluation, and release.

The project was a close collaboration; both contributions were essential to the result.

### Links
- Code: https://github.com/Sekinal/rosettia-chanka
- NLLB/M2M-100 support for vLLM (our fork): https://github.com/Sekinal/vllm/tree/add-nllb-m2m100-support
- Data: https://huggingface.co/datasets/Thermostatic/rosettia-chanka-data
