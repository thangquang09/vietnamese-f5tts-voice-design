#!/usr/bin/env bash
# =============================================================================
# Vietnamese CapSpeech NAR — Data Preprocessing on 4×A100 40GB
#
# Chạy:
#   nohup bash run_preprocess_4gpu.sh > /tmp/capspeech_data/preprocess.log 2>&1 &
#   tail -f /tmp/capspeech_data/preprocess.log
# =============================================================================
set -e

# ============ CONFIGURATION ============
CSV_DIR="/data1/speech/nhandt23/06_thang/vn-instructiontts/results/final_dataset"
SAVE_DIR="/tmp/capspeech_data/vn_capspeech"
CAPTION_COLUMN="caption_full"
NUM_WORKERS=16
MIN_DURATION=1.0
MAX_DURATION=18.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $@"
}

mkdir -p "$SAVE_DIR"

# ====================================================================
# STAGE 0: CSV → JSON + Manifest (CPU only, ~30 min)
# ====================================================================
log "========== STAGE 0: CSV → JSON + Manifest =========="
python "$SCRIPT_DIR/preprocess_vn.py" \
    --csv_dir "$CSV_DIR" \
    --save_dir "$SAVE_DIR" \
    --caption_column "$CAPTION_COLUMN" \
    --splits val \
    --task_filter general \
    --min_duration $MIN_DURATION \
    --max_duration $MAX_DURATION \
    --num_workers 8
log "STAGE 0 done."

# ====================================================================
# STAGE 1: Build Vietnamese character vocab (CPU only, ~5 min)
# ====================================================================
log "========== STAGE 1: Build Vocab =========="
python "$SCRIPT_DIR/build_vocab_vn.py" \
    --save_dir "$SAVE_DIR"
log "STAGE 1 done."

VOCAB_SIZE=$(wc -l < "$SAVE_DIR/vocab.txt")
log "Vocab size = $VOCAB_SIZE — Nhớ cập nhật text_num_embeds trong YAML config!"

# ====================================================================
# STAGE 2: Character-level tokenization (CPU only, ~10 min)
# ====================================================================
log "========== STAGE 2: Character tokenization =========="
python "$SCRIPT_DIR/phonemize_vn.py" \
    --save_dir "$SAVE_DIR" \
    --num_cpus $NUM_WORKERS
log "STAGE 2 done."

# ====================================================================
# STAGE 3: ViT5-large caption encoding — PARALLEL trên 4 GPUs
# ====================================================================
log "========== STAGE 3: ViT5-large encoding (4 GPUs parallel) =========="

# Đếm tổng số entries từ JSON
TOTAL_TRAIN=$(python -c "import json; d=json.load(open('$SAVE_DIR/jsons/train.json')); print(len(d))")
TOTAL_VAL=$(python -c "import json; d=json.load(open('$SAVE_DIR/jsons/val.json')); print(len(d))")
TOTAL=$((TOTAL_TRAIN + TOTAL_VAL))
log "Total entries: train=$TOTAL_TRAIN, val=$TOTAL_VAL, total=$TOTAL"

# Chia train set thành 4 phần cho 4 GPUs
CHUNK=$((TOTAL_TRAIN / 4))
START_0=0
END_0=$CHUNK
START_1=$CHUNK
END_1=$((CHUNK * 2))
START_2=$((CHUNK * 2))
END_2=$((CHUNK * 3))
START_3=$((CHUNK * 3))
END_3=$TOTAL_TRAIN

log "GPU 0: train[$START_0:$END_0], GPU 1: train[$START_1:$END_1]"
log "GPU 2: train[$START_2:$END_2], GPU 3: train[$START_3:$END_3]"

# Chạy 4 GPU song song cho train set
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/caption_vn.py" \
    --save_dir "$SAVE_DIR" \
    --model_name "VietAI/vit5-large" \
    --batch_size 32 \
    --split train \
    --start $START_0 --end $END_0 \
    > /tmp/capspeech_data/caption_gpu0.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/caption_vn.py" \
    --save_dir "$SAVE_DIR" \
    --model_name "VietAI/vit5-large" \
    --batch_size 32 \
    --split train \
    --start $START_1 --end $END_1 \
    > /tmp/capspeech_data/caption_gpu1.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python "$SCRIPT_DIR/caption_vn.py" \
    --save_dir "$SAVE_DIR" \
    --model_name "VietAI/vit5-large" \
    --batch_size 32 \
    --split train \
    --start $START_2 --end $END_2 \
    > /tmp/capspeech_data/caption_gpu2.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python "$SCRIPT_DIR/caption_vn.py" \
    --save_dir "$SAVE_DIR" \
    --model_name "VietAI/vit5-large" \
    --batch_size 32 \
    --split train \
    --start $START_3 --end $END_3 \
    > /tmp/capspeech_data/caption_gpu3.log 2>&1 &
PID3=$!

log "Waiting for 4 GPU processes: $PID0 $PID1 $PID2 $PID3"

# Chờ tất cả xong
FAIL=0
for pid in $PID0 $PID1 $PID2 $PID3; do
    wait $pid || { log "ERROR: Process $pid failed!"; FAIL=1; }
done

if [ $FAIL -ne 0 ]; then
    log "STAGE 3 (train) had failures! Check caption_gpu*.log"
    exit 1
fi
log "STAGE 3 (train) done."

# Chạy val set trên 1 GPU (nhỏ)
log "STAGE 3: Encoding val set on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/caption_vn.py" \
    --save_dir "$SAVE_DIR" \
    --model_name "VietAI/vit5-large" \
    --batch_size 32 \
    --split val
log "STAGE 3 (val) done."

# ====================================================================
# STAGE 4: CLAP "none" embedding (CPU/GPU, ~1 min)
# ====================================================================
log "========== STAGE 4: CLAP 'none' embedding =========="
python "$SCRIPT_DIR/prepare_clap_none.py" \
    --save_dir "$SAVE_DIR"
log "STAGE 4 done."

# ====================================================================
# STAGE 5: Verification
# ====================================================================
log "========== STAGE 5: Verification =========="
echo "  JSON files:"
ls -la "$SAVE_DIR/jsons/"
echo "  Manifest files:"
ls -la "$SAVE_DIR/manifest/"
echo "  Vocab:"
wc -l "$SAVE_DIR/vocab.txt"
echo "  G2P files:"
ls "$SAVE_DIR/g2p/" | wc -l
echo "  T5 embeddings:"
ls "$SAVE_DIR/t5/" | wc -l
echo "  CLAP embeddings:"
ls "$SAVE_DIR/clap_embs/"

log "========== ALL DONE =========="
log "Data saved to: $SAVE_DIR"
log ""
log "Disk usage:"
du -sh "$SAVE_DIR"
du -sh "$SAVE_DIR"/*
log ""
log "NEXT STEPS:"
log "  1. Vocab size = $(wc -l < $SAVE_DIR/vocab.txt)"
log "  2. Cập nhật text_num_embeds trong configs/finetune_vn.yaml"
log "  3. Chạy training:"
log "     accelerate launch finetune.py --config-name configs/finetune_vn.yaml --pretrained-ckpt YOUR_CKPT"
