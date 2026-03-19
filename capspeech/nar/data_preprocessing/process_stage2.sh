#!/bin/bash
# Stage 2 preprocessing pipeline for CapSpeech NAR Vietnamese
#
# Preprocesses ONLY new segments (emotion/accent) that aren't in Stage 1.
# Both phonemize_vn.py and caption_vn.py skip existing files automatically.
#
# Usage: bash process_stage2.sh [GPU_ID]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE1_DIR="/tmp/capspeech_data/vn_capspeech"
STAGE2_DIR="/data1/speech/nhandt23/06_thang/capspeech_stage2"
GPU_ID="${1:-0}"

echo "============================================"
echo "CapSpeech NAR Stage 2 Preprocessing"
echo "============================================"
echo "Stage 1 dir (g2p/t5): ${STAGE1_DIR}"
echo "Stage 2 dir (jsons):  ${STAGE2_DIR}"
echo "GPU: ${GPU_ID}"
echo ""

# Check prereqs
if [ ! -f "${STAGE2_DIR}/jsons/train.json" ]; then
    echo "ERROR: Stage 2 JSONs not found. Run build_stage2_data.py first."
    exit 1
fi

# ---- Step 1: Create temp dir with stage2 jsons + stage1 g2p/t5 ----
WORK_DIR="/tmp/capspeech_stage2_work"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

# Symlink jsons from stage2
ln -sf "${STAGE2_DIR}/jsons" "${WORK_DIR}/jsons"

# Symlink g2p/t5/clap from stage1 (scripts will ADD new files here)
ln -sf "${STAGE1_DIR}/g2p" "${WORK_DIR}/g2p"
ln -sf "${STAGE1_DIR}/t5" "${WORK_DIR}/t5"
ln -sf "${STAGE1_DIR}/clap_embs" "${WORK_DIR}/clap_embs"

echo "[1/4] Workspace ready: ${WORK_DIR}"

# ---- Step 2: Char tokenization for new segments ----
echo ""
echo "[2/4] Char tokenization (skips existing)..."
python3 "${SCRIPT_DIR}/phonemize_vn.py" \
    --save_dir "${WORK_DIR}" \
    --num_cpus 16

# ---- Step 3: ViT5-large encoding for new segments ----
echo ""
echo "[3/4] ViT5-large caption encoding (skips existing)..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 "${SCRIPT_DIR}/caption_vn.py" \
    --save_dir "${WORK_DIR}" \
    --batch_size 32

# ---- Step 4: Setup final stage2 dir with symlinks ----
echo ""
echo "[4/4] Setting up Stage 2 data directory..."
cd "${STAGE2_DIR}"

# Symlink preprocessed dirs from stage1
for subdir in g2p t5 clap_embs; do
    if [ ! -e "${subdir}" ]; then
        ln -sf "${STAGE1_DIR}/${subdir}" "${subdir}"
        echo "  → ${subdir}/ → ${STAGE1_DIR}/${subdir}"
    fi
done

# Copy vocab
if [ ! -f "vocab.txt" ] && [ -f "${STAGE1_DIR}/vocab.txt" ]; then
    cp "${STAGE1_DIR}/vocab.txt" "vocab.txt"
    echo "  → vocab.txt copied"
fi

# Verify
echo ""
echo "Verifying..."
python3 - <<'PYEOF'
import json, os

stage2_dir = "/data1/speech/nhandt23/06_thang/capspeech_stage2"

for split in ['train', 'val']:
    json_path = os.path.join(stage2_dir, "jsons", f"{split}.json")
    if not os.path.exists(json_path):
        continue
    with open(json_path) as f:
        data = json.load(f)
    
    missing_g2p = missing_t5 = 0
    checked = set()
    for entry in data:
        sid = entry['segment_id']
        if sid in checked:
            continue
        checked.add(sid)
        if not os.path.exists(os.path.join(stage2_dir, "g2p", f"{sid}.txt")):
            missing_g2p += 1
        if not os.path.exists(os.path.join(stage2_dir, "t5", f"{sid}.npz")):
            missing_t5 += 1
    
    status = "✅" if missing_g2p == 0 and missing_t5 == 0 else "⚠️"
    print(f"  {status} {split}: {len(checked)} unique, missing g2p={missing_g2p}, t5={missing_t5}")
PYEOF

# Cleanup
rm -rf "${WORK_DIR}"

echo ""
echo "✅ Stage 2 preprocessing complete!"
echo "Data dir: ${STAGE2_DIR}"
echo "Next: run training with finetune_vn_stage2.yaml"
