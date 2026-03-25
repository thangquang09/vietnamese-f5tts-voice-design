#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/speech/nhandt23/06_thang"
ENV_FILE="/data1/speech/nhandt23/.env"
GPU_LIST="0,1,2,3"
REPO_ID="thangquang09/capspeech-nar-vietnamese"
TMP_ROOT="/tmp/nhandt23/capspeech"
DATA_DIR="/tmp/capspeech_data/vn_capspeech_stage2_v3"
LOG_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="$ROOT_DIR/CapSpeech/train_logs/stage2_v3"
mkdir -p "$LOG_ROOT" "$TMP_ROOT"

export HF_HOME="/tmp/nhandt23/hf_home"
export HF_HUB_CACHE="/tmp/nhandt23/hf_hub"
export TRANSFORMERS_CACHE="/tmp/nhandt23/hf_cache"

SNAPSHOT_DIR=$(python - <<PY
import os
import sys
sys.path.insert(0, "$SCRIPT_DIR")
from hf_utils import load_hf_token, snapshot_repo_to_tmp
token = load_hf_token(env_file="$ENV_FILE")
path = snapshot_repo_to_tmp(
    repo_id="$REPO_ID",
    cache_dir="$TMP_ROOT/hf_models",
    token=token,
)
print(path)
PY
)

BASE_CKPT=$(python - <<PY
from pathlib import Path
snap = Path("$SNAPSHOT_DIR")
for candidate in ("checkpoint.pt", "model.pt"):
    path = snap / candidate
    if path.exists():
        print(path)
        break
else:
    raise SystemExit("No checkpoint.pt found in downloaded snapshot")
PY
)

python "$SCRIPT_DIR/data_preprocessing/build_stage2_data_v3.py" \
  --csv-dir "$ROOT_DIR/vn-instructiontts/results/final_dataset" \
  --save-dir "$DATA_DIR" \
  --recipe "$SCRIPT_DIR/configs/stage2_v3_recipe.yaml" \
  --caption-column caption_full \
  --mount-remap /mnt/speech:/data1/speech

bash "$SCRIPT_DIR/data_preprocessing/process_stage2_v3.sh" \
  --data-dir "$DATA_DIR" \
  --gpus "$GPU_LIST" \
  --num-workers 16

CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
  --config_file "$SCRIPT_DIR/accelerate_config.yaml" \
  "$SCRIPT_DIR/finetune.py" \
  --config-name "$SCRIPT_DIR/configs/finetune_vn_stage2_v3.yaml" \
  --pretrained-ckpt "$BASE_CKPT" \
  --train-sampler stage2_v3 \
  --sampler-recipe "$SCRIPT_DIR/configs/stage2_v3_recipe.yaml" \
  --epochs 5 \
  --num-workers 8 \
  --eval-every-step 500 \
  --save-every-step 500 \
  --max-ckpts 3 \
  --log-dir /tmp/capspeech_train_v3/logs \
  --save-dir /tmp/capspeech_train_v3/ckpts \
  --amp fp16
