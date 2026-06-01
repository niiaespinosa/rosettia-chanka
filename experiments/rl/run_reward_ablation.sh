#!/usr/bin/env bash
# Reward ablation for GSPO-NLLB: identical settings, vary ONLY the reward, 200 steps
# each from NLLB-r2, then rank by held-out VAL ChrF (w0) — the fair, common metric.
set -u
cd /root/rosettia-chanka
CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=facebook/nllb-200-1.3B
SFT=outputs/nllb13b_v2_20260529/final
TRAIN=clean_chanka/rl_train_split.parquet
VES=clean_chanka/rl_val.es
VQ=clean_chanka/rl_val.quy
STEPS=200

val_eval () {  # $1=adapter dir  $2=out json
  .venv/bin/python scripts/nllb/eval_nllb_americasnlp.py --model-id "$BASE" --adapter "$1" \
    --test-es "$VES" --test-quy "$VQ" --max-new 64 --out-json "$2" >> "$3" 2>&1
}

# baseline: NLLB-r2 (pre-RL) on val
echo "=== baseline NLLB-r2 on val $(date -u +%T) ==="
val_eval "$SFT" outputs/eval_mbr/abl_nllbr2_val.json outputs/logs/abl_baseline.log

for R in chrf chrfpp chrf_brevity chrf_rep; do
  echo "=== reward=$R: GSPO $STEPS steps $(date -u +%T) ==="
  OUT=outputs/abl_$R
  rm -rf "$OUT"
  .venv/bin/python scripts/rl/gspo_nllb_vllm.py --init-adapter "$SFT" --train-parquet "$TRAIN" \
    --reward-type "$R" --group-size 8 --prompt-batch 48 --micro-batch 48 --vllm-mem-frac 0.20 \
    --max-steps $STEPS --save-steps $STEPS --output-dir "$OUT" > outputs/logs/abl_$R.log 2>&1
  sleep 6   # let the in-process vLLM engine release the GPU before the HF eval
  echo "=== reward=$R: val eval $(date -u +%T) ==="
  val_eval "$OUT/final" outputs/eval_mbr/abl_${R}_val.json outputs/logs/abl_$R.log
done

echo "=========== REWARD ABLATION — VAL ChrF (w0) ==========="
for f in outputs/eval_mbr/abl_nllbr2_val.json outputs/eval_mbr/abl_chrf_val.json \
         outputs/eval_mbr/abl_chrfpp_val.json outputs/eval_mbr/abl_chrf_brevity_val.json \
         outputs/eval_mbr/abl_chrf_rep_val.json; do
  [ -f "$f" ] && printf "%-45s " "$f" && .venv/bin/python -c "import json,sys;print('ChrF_w0', round(json.load(open('$f'))['ChrF_w0'],3))"
done
echo "ABLATION_DONE $(date -u +%T)"
