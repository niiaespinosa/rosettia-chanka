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
license: apache-2.0
pipeline_tag: translation
base_model: unsloth/Qwen3.5-9B
---

# rosettia-quy-v30b-9b-merged

Spanish → **Chanka / Ayacucho Quechua (`quy`)** translation model. A merged
(LoRA → full-weights) fine-tune of **Qwen3.5-9B**, and the strongest *single*
model in the Rosettia Chanka project. On the **AmericasNLP 2021** spa→quy test it
sets a new state of the art for the benchmark.

## Results — AmericasNLP 2021 spa→quy test (1003 lines, ChrF, sacrebleu `word_order=0`)

| System | ChrF (w0) |
|---|---|
| Helsinki 2021 (published) | 39.40 |
| Sheffield 2023 NLLB-3.3B (published) | 34.01 |
| **This model — greedy** | **40.55** |
| This model — ChrF-MBR (dedup) decoding | 42.93 |
| **Ensemble (this + NLLB-1.3B-r2), dedup-MBR** | **44.97** ← project SOTA |

> Note: many shared-task papers report **ChrF++** (`word_order=2`), which is not
> directly comparable to the official **ChrF** (`word_order=0`) used here.

The big jumps come from **decoding**, not just the model: ChrF-based Minimum-Bayes-Risk
decoding (sample N candidates, pick the one maximizing mean pairwise ChrF over the
*deduplicated* candidate pool) adds ~+2.4 ChrF with no extra training, and pooling
candidates with a diverse NLLB-1.3B reaches 44.97.

## Training

- **Base:** Qwen3.5-9B (Unsloth), via a broad multilingual SFT stage → Chanka SFT
  on a **curated** 1,929-pair Spanish–Chanka corpus → merged to 16-bit weights.
- Decoder-only; **minimal native Quechua pretraining** (the main reason curated data
  + good decoding matter more than scale here).
- **Zero leakage**: the AmericasNLP 2021 test set was held out throughout.

## Usage

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tok = AutoTokenizer.from_pretrained("Thermostatic/rosettia-quy-v30b-9b-merged")
llm = LLM(model="Thermostatic/rosettia-quy-v30b-9b-merged", dtype="bfloat16")

SYSTEM = "You are a careful Spanish→Quechua (Chanka/Ayacuchana variety) translator."
INSTR  = "Traduce el siguiente texto del español al quechua chanka. Responde solo con la traducción."
src = "Mis abuelos eran una pareja muy cariñosa."
msgs = [{"role":"system","content":SYSTEM},
        {"role":"user","content":f"{INSTR}\n\nEspañol: {src}"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=128,
                                            stop=["<|im_end|>","<|endoftext|>"]))
print(out[0].outputs[0].text.strip())
```

For best quality, use ChrF-MBR decoding (sample 32–64 candidates at T≈0.5, pick the
consensus by mean pairwise sentence-ChrF over the deduplicated pool).

## Intended use & limitations

Research-grade MT for a low-resource, agglutinative language. Outputs should be
reviewed by speakers before any consequential use. Quality varies with domain and
sentence length; named entities and Spanish loanwords are the most error-prone.
