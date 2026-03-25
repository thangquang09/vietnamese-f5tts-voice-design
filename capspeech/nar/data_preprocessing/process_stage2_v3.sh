#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/tmp/capspeech_data/vn_capspeech_stage2_v3"
GPU_LIST="0,1,2,3"
NUM_WORKERS=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

mkdir -p "$DATA_DIR"
[[ -f "$DATA_DIR/jsons/train.json" ]] || { echo "Missing $DATA_DIR/jsons/train.json"; exit 1; }

python "$SCRIPT_DIR/build_vocab_vn.py" --save_dir "$DATA_DIR"
python "$SCRIPT_DIR/phonemize_vn.py" --save_dir "$DATA_DIR" --num_cpus "$NUM_WORKERS"

TOTAL_TRAIN=$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("$DATA_DIR/jsons/train.json").read_text(encoding="utf-8"))
print(len(data))
PY
)
CHUNK=$(( (TOTAL_TRAIN + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))

for idx in "${!GPUS[@]}"; do
  start=$(( idx * CHUNK ))
  end=$(( (idx + 1) * CHUNK ))
  CUDA_VISIBLE_DEVICES="${GPUS[$idx]}" python "$SCRIPT_DIR/caption_vn.py" \
    --save_dir "$DATA_DIR" \
    --model_name "VietAI/vit5-large" \
    --batch_size 32 \
    --split train \
    --start "$start" \
    --end "$end" &
done
wait

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python "$SCRIPT_DIR/caption_vn.py" \
  --save_dir "$DATA_DIR" \
  --model_name "VietAI/vit5-large" \
  --batch_size 32 \
  --split val

python "$SCRIPT_DIR/prepare_clap_none.py" --save_dir "$DATA_DIR"

python - <<PY
import json
import os
from pathlib import Path
base = Path("$DATA_DIR")
for split in ("train", "val"):
    entries = json.loads((base / "jsons" / f"{split}.json").read_text(encoding="utf-8"))
    ids = {item["segment_id"] for item in entries}
    missing_g2p = sum(not (base / "g2p" / f"{sid}.txt").exists() for sid in ids)
    missing_t5 = sum(not (base / "t5" / f"{sid}.npz").exists() for sid in ids)
    print(f"{split}: unique={len(ids)} missing_g2p={missing_g2p} missing_t5={missing_t5}")
    if missing_g2p or missing_t5:
        raise SystemExit(1)
PY

echo "Stage2-v3 preprocessing complete: $DATA_DIR"
