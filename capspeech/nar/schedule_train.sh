#!/usr/bin/env bash
# Đợi preprocessing xong rồi tự động chạy training
# Chạy: nohup bash schedule_train.sh > /tmp/capspeech_train/schedule.log 2>&1 &

set -e

echo "[$(date)] Waiting for preprocessing to finish..."

# Đợi cho run_preprocess_4gpu.sh kết thúc (thông minh hơn hẹn giờ cố định)
while pgrep -f "run_preprocess_4gpu.sh" > /dev/null 2>&1 || \
      pgrep -f "caption_vn.py" > /dev/null 2>&1 || \
      pgrep -f "prepare_clap_none.py" > /dev/null 2>&1; do
    echo "[$(date)] Preprocessing still running... checking again in 60s"
    sleep 60
done

echo "[$(date)] Preprocessing finished! Verifying data..."

# Verify data tồn tại
SAVE_DIR="/tmp/capspeech_data/vn_capspeech"
for f in "$SAVE_DIR/jsons/train.json" "$SAVE_DIR/jsons/val.json" \
         "$SAVE_DIR/manifest/train.txt" "$SAVE_DIR/manifest/val.txt" \
         "$SAVE_DIR/vocab.txt"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f — preprocessing may have failed!"
        exit 1
    fi
done

T5_COUNT=$(ls "$SAVE_DIR/t5/" 2>/dev/null | wc -l)
G2P_COUNT=$(ls "$SAVE_DIR/g2p/" 2>/dev/null | wc -l)
echo "[$(date)] T5 embeddings: $T5_COUNT, G2P files: $G2P_COUNT"

if [ "$T5_COUNT" -lt 1000 ]; then
    echo "ERROR: Too few T5 embeddings ($T5_COUNT) — preprocessing failed!"
    exit 1
fi

echo "[$(date)] Data OK! Starting training on GPU 2,3..."

mkdir -p /tmp/capspeech_train/logs /tmp/capspeech_train/ckpts

cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 --mixed_precision fp16 \
    finetune.py \
    --config-name configs/finetune_vn.yaml \
    --pretrained-ckpt /tmp/capspeech_pretrained/nar_CapTTS.pt \
    --epochs 15 \
    --num-workers 8 \
    --eval-every-step 2000 \
    --save-every-step 2000 \
    --max-ckpts 3 \
    --log-dir /tmp/capspeech_train/logs/ \
    --save-dir /tmp/capspeech_train/ckpts/ \
    --amp fp16
