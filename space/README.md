---
title: RosettIA · Español → Quechua Chanka
emoji: 🌄
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Spanish → Chanka/Ayacucho Quechua MT demo
models:
  - Thermostatic/rosettia-quy-gspo-nllb13b-merged
---

# RosettIA — Spanish → Chanka/Ayacucho Quechua

A demo of a 1.3B NLLB model fine-tuned with **GSPO reinforcement learning** for Spanish →
Chanka/Ayacucho Quechua (`quy`). Runs on **ZeroGPU**.

Research-grade; single-benchmark, single-reference ChrF (≈46), **no native-speaker
evaluation** — review outputs with a speaker before any consequential use.

- Model: https://huggingface.co/Thermostatic/rosettia-quy-gspo-nllb13b-merged
- Code & report: https://github.com/Sekinal/rosettia-chanka

Authors: Estefanía Espinosa Fernández & Irving Ernesto Quezada Ramírez (SomosNLP hackathon).
