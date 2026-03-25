# CapSpeech NAR Vietnamese — Stage 2 Training Status

> Last updated: **2026-03-21 14:55 UTC+7**

---

## 1. Tổng quan

Stage 2 mở rộng mô hình CapSpeech NAR Vietnamese (CrossDiT 614M params) từ task **general-only** sang đa task: emotion, age groups (children, teen, senior) + adult rehearsal buffer.

| Thông tin | Giá trị |
|:--|:--|
| **Kiến trúc** | CrossDiT (dim=1024, depth=24, heads=16) |
| **Tham số** | 614.10M (trainable) |
| **Vocoder** | BigVGAN v2 24kHz |
| **Caption Encoder** | ViT5-large (1024-dim) |
| **Precision** | FP16 (mixed precision) |
| **GPUs** | 2× A100 (CUDA_VISIBLE_DEVICES=2,3) |
| **Server** | dgx-a100-5.cm.cluster |

---

## 2. Tình trạng hiện tại

| Mục | Chi tiết |
|:--|:--|
| **Trạng thái** | 🟡 **Data ready, chờ train** |
| **Checkpoint bắt đầu** | Stage 1 final: `/tmp/capspeech_train/ckpts/finetune_captts_vn/30000.pt` |
| **Data** | ✅ Build + preprocess hoàn tất |

---

## 3. Stage 2 v1 (ĐÃ BỎ)

> ⚠️ Stage 2 v1 đã bị bỏ do data strategy kém hiệu quả.

**Vấn đề chính:**
- Dùng regex trên `caption_full` thay vì column `kw_age` → phân loại sai, miss data
- Phân nhóm theo `task` column (emotion/accent/general) thay vì theo đặc tính thực (age/emotion)
- Children chỉ 62 unique × 100 = overfitting; Adult rehearsal 50K áp đảo
- LR quá thấp (5e-6) → model "lười học" features mới
- Kết quả: không thấy sự thay đổi về age hay emotion trong output

---

## 4. Stage 2 v2 (HIỆN TẠI)

### 4.1 Thay đổi chính so với v1

| Aspect | v1 (bỏ) | v2 (mới) |
|:--|:--|:--|
| **Phân nhóm** | Theo `task` + regex caption | Theo `kw_emotion` > `kw_age` trực tiếp |
| **Ưu tiên** | task-based | Emotion > Age (emotion rarer) |
| **Adult rehearsal** | 50,000 | **30,000** |
| **Learning rate** | 5e-6 | **1e-5** |
| **drop_spk** | 0.1 | **0.15** |
| **Pretrained ckpt** | Stage 2 v1 ckpt | **Stage 1 final (30000.pt)** |

### 4.2 Dữ liệu Stage 2 v2

| Nhóm | Unique | ×Factor | Final | Tỉ trọng |
|:--|--:|--:|--:|--:|
| **teen** (thanh thiếu niên) | 11,223 | ×3 | 33,669 | 22.6% |
| **children** (trẻ em) | 15,061 | ×2 | 30,122 | 20.2% |
| **adult** (rehearsal) | 29,727 | ×1 | 29,727 | 20.0% |
| **emotion** (có kw_emotion) | 1,932 | ×15 | 28,980 | 19.5% |
| **senior** (cao tuổi) | 3,287 | ×8 | 26,296 | 17.7% |
| **TOTAL** | | | **148,794** | 100% |

**Validation set:** 12,945 entries (5,944 unique segments)

**Preprocessing verified:**
```
✅ train: 61,230 unique segments, missing g2p=0, t5=0
✅ val:    5,944 unique segments, missing g2p=0, t5=0
```

### 4.3 Config

```yaml
# finetune_vn_stage2.yaml (v2)
opt:
  learning_rate: 1.0e-05    # Higher than v1 (5e-6), lower than Stage 1 (2e-5)
  drop_spk: 0.15            # More reliance on caption
  drop_text: 0.5
  batch_size: 32
  accumulation_steps: 2     # Effective batch = 128 (32×2×2 GPUs)
  lr_scheduler:
    warmup_steps: 500
    decay_steps: 50000
```

### 4.4 Đường dẫn dữ liệu

| Item | Vị trí |
|:--|:--|
| train.json / val.json | `/data1/.../capspeech_stage2/jsons/` |
| manifest (train/val) | `/data1/.../capspeech_stage2/manifest/` |
| vocab.txt | `/data1/.../capspeech_stage2/vocab.txt` |
| g2p/ | symlink → `/tmp/capspeech_data/vn_capspeech/g2p` |
| t5/ | symlink → `/tmp/capspeech_data/vn_capspeech/t5` |
| clap_embs/ | symlink → `/tmp/capspeech_data/vn_capspeech/clap_embs` |

---

## 5. Lệnh training

```bash
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --config_file accelerate_config.yaml \
    finetune.py \
    --config-name configs/finetune_vn_stage2.yaml \
    --pretrained-ckpt /tmp/capspeech_train/ckpts/finetune_captts_vn/30000.pt \
    --epochs 10 \
    --num-workers 8 \
    --eval-every-step 500 \
    --save-every-step 500 \
    --max-ckpts 3 \
    --log-dir /tmp/capspeech_train/logs_stage2/ \
    --save-dir /tmp/capspeech_train/ckpts_stage2/ \
    --amp fp16
```

> **Lưu ý:** Lần này **KHÔNG dùng `--resume-from`** — bắt đầu fresh từ Stage 1 checkpoint.

---

## 6. Scripts liên quan

| Script | Mục đích | Vị trí |
|:--|:--|:--|
| `build_stage2_data.py` | Build mixed dataset (v2, dùng kw_age/kw_emotion) | `data_preprocessing/` |
| `process_stage2.sh` | Preprocessing (g2p + ViT5 encoding) | `data_preprocessing/` |
| `finetune.py` | Training script | `capspeech/nar/` |
| `finetune_vn_stage2.yaml` | Stage 2 v2 config | `capspeech/nar/configs/` |

---

## 7. Timeline

| Thời gian (UTC+7) | Sự kiện |
|:--|:--|
| ~19/03 22:00 | Stage 1 training hoàn thành (4 epochs, loss ~0.48) |
| 19/03 22:xx | Stage 2 v1 auto-start |
| 20/03 ~05:30 | Stage 2 v1 epoch 1 done, NCCL timeout (eval bug → fixed) |
| 20/03 ~09:19 | Stage 2 v1 resume |
| 21/03 | ❌ Stage 2 v1 kết quả kém: không thấy thay đổi age/emotion |
| 21/03 14:50 | Phân tích root cause: data strategy sai (regex, task-based grouping) |
| 21/03 14:54 | ✅ Stage 2 v2 data build + preprocess hoàn tất |
| 21/03 ??:?? | 🔜 Stage 2 v2 training bắt đầu |
