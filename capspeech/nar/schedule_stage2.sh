#!/bin/bash
# ==============================================================================
# Schedule Stage 2 Training After Stage 1 Completes
# ==============================================================================
#
# This script monitors the Stage 1 training process. Once it finishes,
# it automatically backs up the best checkpoint and starts Stage 2 training.
#
# Usage:
#   nohup bash schedule_stage2.sh > /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar/train_logs/schedule_stage2.log 2>&1 &
#
# IMPORTANT:
#   - Stage 1 ckpts: /tmp/capspeech_train/ckpts/finetune_captts_vn/
#   - Stage 2 ckpts: /tmp/capspeech_train/ckpts_stage2/finetune_captts_vn_stage2/
#   - Stage 2 logs:  /tmp/capspeech_train/logs_stage2/
#   - Stage 2 data:  /data1/speech/nhandt23/06_thang/capspeech_stage2 (NFS)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE1_CKPT_DIR="/tmp/capspeech_train/ckpts/finetune_captts_vn"
STAGE2_CKPT_DIR="/tmp/capspeech_train/ckpts_stage2"
STAGE2_LOG_DIR="/tmp/capspeech_train/logs_stage2"
STAGE1_LOG="/data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar/train_logs/train.log"
STAGE2_LOG="/data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar/train_logs/train_stage2.log"
BACKUP_DIR="/data1/speech/nhandt23/06_thang/capspeech_stage1_ckpts"

echo "============================================================"
echo "  CapSpeech Stage 2 Scheduler"
echo "  $(date)"
echo "============================================================"
echo "Monitoring Stage 1 log: ${STAGE1_LOG}"
echo ""

# ---- Phase 1: Wait for Stage 1 to finish ----
echo "[1/4] Waiting for Stage 1 training to complete..."
echo "      (Checking every 60 seconds for process completion)"
echo ""

while true; do
    # Check if finetune.py is running
    if pgrep -f "finetune.py.*finetune_vn.yaml" > /dev/null 2>&1; then
        # Still running — check progress
        LAST_LINE=$(tail -1 "${STAGE1_LOG}" 2>/dev/null | tr -d '\r' | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' || echo "")
        echo "  $(date '+%H:%M:%S') Stage 1 still running: ${LAST_LINE:0:100}"
        sleep 60
    else
        echo ""
        echo "  ✅ Stage 1 process has ended! ($(date))"
        break
    fi
done

# ---- Phase 2: Backup Stage 1 checkpoints to NFS ----
echo ""
echo "[2/4] Backing up Stage 1 checkpoints to NFS..."
mkdir -p "${BACKUP_DIR}"

# Find and copy the latest checkpoint
LATEST_CKPT=$(ls -t "${STAGE1_CKPT_DIR}"/*.pt 2>/dev/null | head -1)
if [ -n "${LATEST_CKPT}" ]; then
    echo "  Found latest checkpoint: $(basename ${LATEST_CKPT})"
    cp -v "${LATEST_CKPT}" "${BACKUP_DIR}/"
    
    # Also copy ALL remaining checkpoints (max-ckpts=3)
    for ckpt in "${STAGE1_CKPT_DIR}"/*.pt; do
        if [ -f "${ckpt}" ]; then
            echo "  Backing up: $(basename ${ckpt})"
            cp -v "${ckpt}" "${BACKUP_DIR}/"
        fi
    done
    echo "  ✅ Checkpoints backed up to ${BACKUP_DIR}/"
else
    echo "  ⚠️ No checkpoints found in ${STAGE1_CKPT_DIR}"
    echo "  Exiting..."
    exit 1
fi

# Use the latest checkpoint as pretrained for Stage 2
PRETRAINED_CKPT="${LATEST_CKPT}"
echo ""
echo "  Using pretrained checkpoint: ${PRETRAINED_CKPT}"

# ---- Phase 3: Prepare Stage 2 directories ----
echo ""
echo "[3/4] Preparing Stage 2 directories..."
mkdir -p "${STAGE2_CKPT_DIR}"
mkdir -p "${STAGE2_LOG_DIR}"
echo "  Stage 2 ckpt dir: ${STAGE2_CKPT_DIR}"
echo "  Stage 2 log dir:  ${STAGE2_LOG_DIR}"

# Verify Stage 2 data exists
if [ ! -f "/data1/speech/nhandt23/06_thang/capspeech_stage2/jsons/train.json" ]; then
    echo "  ⚠️ Stage 2 data not found!"
    exit 1
fi
echo "  ✅ Stage 2 data verified"

# ---- Phase 4: Launch Stage 2 Training ----
echo ""
echo "[4/4] Launching Stage 2 training... ($(date))"
echo "============================================================"
echo ""

cd "${SCRIPT_DIR}"

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --config_file accelerate_config.yaml \
    finetune.py \
    --config-name configs/finetune_vn_stage2.yaml \
    --pretrained-ckpt "${PRETRAINED_CKPT}" \
    --epochs 10 \
    --num-workers 8 \
    --eval-every-step 500 \
    --save-every-step 500 \
    --max-ckpts 3 \
    --log-dir "${STAGE2_LOG_DIR}" \
    --save-dir "${STAGE2_CKPT_DIR}" \
    --amp fp16 \
    2>&1 | tee "${STAGE2_LOG}"

echo ""
echo "============================================================"
echo "  Stage 2 Training Complete! ($(date))"
echo "============================================================"
