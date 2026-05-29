#!/usr/bin/env bash
# NLLB-1.3B + LoRA (BSC-2024-winner recipe) on the cleaned 124k aggregate corpus,
# then eval on AmericasNLP 2021 (official ChrF w0, apostrophe-suppressed).
set -u
cd /root/rosettia-chanka
mkdir -p outputs/logs outputs/eval_americasnlp_2021

CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN}

CORPUS=clean_chanka/nllb_aggregate_corpus.parquet
OUT=outputs/nllb13b_aggregate_$(date -u +%Y%m%d)

echo "=== NLLB-1.3B LoRA on aggregate (124k) $(date -u +%FT%TZ) ==="
# 124k pairs / (bs16*ga2=32) ~= 3900 steps/epoch. 6 epochs ~= 23k steps.
# warmup 2000 (~9% — BSC's 15k would exceed our total steps), eval/save every 1000.
.venv/bin/python scripts/nllb/train_nllb_chanka.py \
  --model-id facebook/nllb-200-1.3B \
  --train-parquet "$CORPUS" \
  --src-field reviewed_spanish --tgt-field reviewed_chanka_quechua \
  --lora-r 256 --lora-alpha 512 \
  --lr 2e-4 --scheduler inverse_sqrt --warmup-steps 2000 \
  --batch-size 16 --grad-accum 2 \
  --epochs 6 --eval-steps 1000 \
  --max-len 128 \
  --output-dir "$OUT" \
  > outputs/logs/nllb13b_aggregate.log 2>&1
echo "train done. ckpts:"; ls "$OUT" | grep -E "checkpoint|final"
echo "=== NLLB chain DONE $(date -u +%FT%TZ) ==="
