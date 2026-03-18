#!/usr/bin/env bash
# Vietnamese data preprocessing pipeline for CapSpeech NAR
#
# Usage:
#   cd capspeech/nar/data_preprocessing
#   bash process_vn.sh
#
# Prerequisites:
#   - Vietnamese Instruction TTS dataset at CSV_DIR
#   - GPU available for caption encoding (Stage 4)
#   - CLAP checkpoint or HuggingFace access (Stage 5)

set -e

# ============ CONFIGURATION ============
# Path to Vietnamese Instruction TTS CSV files
CSV_DIR="/data1/speech/nhandt23/06_thang/vn-instructiontts/results/final_dataset"

# Path to save processed data
SAVE_DIR="/tmp/capspeech_data/vn_capspeech"

# Caption column to use: caption_full or caption_partial
CAPTION_COLUMN="caption_full"

# Processing parameters
NUM_WORKERS=16
MIN_DURATION=1.0
MAX_DURATION=18.0
SPLITS="train val"

# ============ STAGE CONTROL ============
# Set stage/stop_stage to control which steps to run
# Stage already done? Set stage > that step
stage=0
stop_stage=5

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $@"
}

# Create output directory
mkdir -p "$SAVE_DIR"

# Stage 0: Convert CSV to JSON + manifest
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Converting CSV to JSON + manifest..."
    python preprocess_vn.py \
        --csv_dir "$CSV_DIR" \
        --save_dir "$SAVE_DIR" \
        --caption_column "$CAPTION_COLUMN" \
        --splits $SPLITS \
        --min_duration $MIN_DURATION \
        --max_duration $MAX_DURATION \
        --num_workers $NUM_WORKERS
fi

# Stage 1: Build Vietnamese character vocabulary
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Building Vietnamese character vocabulary..."
    python build_vocab_vn.py \
        --save_dir "$SAVE_DIR"
fi

# Stage 2: Character-level tokenization (Vietnamese "phonemization")
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Character-level tokenization..."
    python phonemize_vn.py \
        --save_dir "$SAVE_DIR" \
        --num_cpus $NUM_WORKERS
fi

# Stage 3: Encode captions with ViT5-large (requires GPU)
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Encoding captions with ViT5-large..."
    python caption_vn.py \
        --save_dir "$SAVE_DIR" \
        --model_name "VietAI/vit5-large"
fi

# Stage 4: Prepare CLAP embedding for 'none' tag
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Preparing CLAP embedding for 'none' tag..."
    python prepare_clap_none.py \
        --save_dir "$SAVE_DIR"
fi

# Stage 5: Verify data
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Verifying processed data..."
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
fi

log "Done! Processed data saved to: $SAVE_DIR"
log ""
log "Next steps:"
log "  1. Check vocab size in $SAVE_DIR/vocab.txt"
log "  2. Update text_num_embeds in configs/finetune_vn.yaml"
log "  3. Start training:"
log "     cd capspeech/nar/"
log "     accelerate launch finetune.py --config-name configs/finetune_vn.yaml --pretrained-ckpt YOUR_CKPT"
