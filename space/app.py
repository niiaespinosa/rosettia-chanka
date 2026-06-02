"""RosettIA — Spanish → Chanka/Ayacucho Quechua translation demo (ZeroGPU)."""
import spaces
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "Thermostatic/rosettia-quy-gspo-nllb13b-merged"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="spa_Latn", tgt_lang="quy_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.eval()
BOS = tokenizer.convert_tokens_to_ids("quy_Latn")

# Apostrophe suppression — Ayacucho/Chanka quy has no glottalization; a small free gain.
_apos = ("'", "’", "ʼ")
_bad = [[tid] for tok, tid in tokenizer.get_vocab().items()
        if any(c in tok.replace("▁", "") for c in _apos)]
BAD_WORDS = _bad or None


@spaces.GPU(duration=60)
def translate(text: str, num_beams: int = 5, suppress_apostrophe: bool = True,
              block_repeats: bool = True) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    model.to("cuda")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    gen = dict(forced_bos_token_id=BOS, num_beams=int(num_beams), max_new_tokens=160)
    if block_repeats:
        gen["no_repeat_ngram_size"] = 3
    if suppress_apostrophe:
        gen["bad_words_ids"] = BAD_WORDS
    with torch.no_grad():
        out = model.generate(**enc, **gen)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


EXAMPLES = [
    "No sé por qué sucedió eso.",
    "Buenos días, ¿cómo está usted?",
    "Mis abuelos eran una pareja muy cariñosa.",
    "Le diste flores bonitas a mamá.",
    "El tiroteo fue cerca de mi casa y me da miedo salir.",
    "Tiene usted derecho a un abogado y a un intérprete.",
]

THEME = gr.themes.Soft(
    primary_hue="teal", secondary_hue="emerald", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container {max-width: 880px !important; margin: auto !important;}
#hdr {text-align:center; padding: 6px 0 2px 0;}
#hdr h1 {font-size: 2.0rem; margin-bottom: 2px;}
#hdr p {color:#5b6675; margin-top: 2px;}
#pill {display:inline-block; background:#e6f3f1; color:#1f6f6b; border-radius:999px;
       padding:3px 12px; font-size:0.82rem; font-weight:600; margin-top:8px;}
footer {visibility: hidden;}
"""

ABOUT = """
### About

This translates **Spanish → Chanka / Ayacucho Quechua** (`quy`) with a 1.3B NLLB model
fine-tuned by **GSPO reinforcement learning** — the strongest result we are aware of on the
AmericasNLP 2021 spa→quy benchmark (**≈46 ChrF**, well above prior published systems).
Decoding here uses beam search + apostrophe-suppression + no-repeat, our standard recipe.

**Please read — honest limitations.** This is a **research-grade** system for an extremely
low-resource language, validated only on a single benchmark with single-reference ChrF and
**no native-speaker evaluation**. Treat outputs as useful drafts, **review with a speaker
before any consequential use**, and expect errors on numbers, named entities, and long or
out-of-domain sentences.

**Models & data**
- Model (merged): [rosettia-quy-gspo-nllb13b-merged](https://huggingface.co/Thermostatic/rosettia-quy-gspo-nllb13b-merged)
- LoRA adapter + full report/figures: [rosettia-quy-gspo-nllb13b-lora](https://huggingface.co/Thermostatic/rosettia-quy-gspo-nllb13b-lora)
- Dataset (redistributable subset): [rosettia-chanka-data](https://huggingface.co/datasets/Thermostatic/rosettia-chanka-data)
- Code & report: [github.com/Sekinal/rosettia-chanka](https://github.com/Sekinal/rosettia-chanka)

**Authors.** A two-person SomosNLP hackathon project — **Estefanía Espinosa Fernández**
(data curation; initial Qwen3.5 LoRA experiments) and **Irving Ernesto Quezada Ramírez**
([irvingernesto.com](https://irvingernesto.com); modeling through the final system). A close
collaboration; both contributions were essential. Base model: `facebook/nllb-200-1.3B`.
"""

with gr.Blocks(title="RosettIA · Español → Quechua Chanka") as demo:
    gr.HTML(
        "<div id='hdr'><h1>🌄 RosettIA</h1>"
        "<p>Spanish → Chanka / Ayacucho Quechua (<code>quy</code>) translation</p>"
        "<span id='pill'>GSPO-NLLB · ≈46 ChrF on AmericasNLP 2021 · research demo</span></div>"
    )
    with gr.Row(equal_height=True):
        inp = gr.Textbox(label="Español", lines=5, autofocus=True,
                         placeholder="Escribe una frase en español…")
        outp = gr.Textbox(label="Chanka / Ayacucho Quechua (quy)", lines=5,
                          interactive=False)
    btn = gr.Button("Traducir  →", variant="primary")
    gr.Examples(EXAMPLES, inputs=inp, label="Ejemplos")

    with gr.Accordion("Advanced decoding options", open=False):
        beams = gr.Slider(1, 5, value=5, step=1,
                          label="Beam size — 1 = greedy (fastest) · 5 = best quality (default)")
        supp = gr.Checkbox(value=True,
                           label="Suppress apostrophes (Ayacucho/Chanka quy has no glottalization)")
        norep = gr.Checkbox(value=True, label="Block repeated n-grams (no_repeat_ngram_size = 3)")

    with gr.Accordion("About · model, limitations, credits", open=False):
        gr.Markdown(ABOUT)

    ins = [inp, beams, supp, norep]
    btn.click(translate, ins, outp)
    inp.submit(translate, ins, outp)

demo.queue(max_size=20).launch(theme=THEME, css=CSS)
