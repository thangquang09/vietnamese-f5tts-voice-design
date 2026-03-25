# CapSpeech Stage2-v3 Train Guide

Stage2-v3 dung base model tu Hugging Face, preprocess va checkpoint deu nam o `/tmp`, log dai nam o NFS `/data1`.

## Assumptions

- `ROOT_DIR=/data1/speech/nhandt23/06_thang`
- server train: A100, `4xA100 40GB`
- mount data: `/data1`
- token Hugging Face co san trong env hoac `.env`

## 1. Build data

```bash
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar

python data_preprocessing/build_stage2_data_v3.py \
  --csv-dir /data1/speech/nhandt23/06_thang/vn-instructiontts/results/final_dataset \
  --save-dir /tmp/capspeech_data/vn_capspeech_stage2_v3 \
  --recipe configs/stage2_v3_recipe.yaml \
  --caption-column caption_full \
  --mount-remap /mnt/speech:/data1/speech

bash data_preprocessing/process_stage2_v3.sh \
  --data-dir /tmp/capspeech_data/vn_capspeech_stage2_v3 \
  --gpus 0,1,2,3 \
  --num-workers 16
```

## 2. Launch training

```bash
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar
mkdir -p /data1/speech/nhandt23/06_thang/CapSpeech/train_logs/stage2_v3

nohup bash train_stage2_v3_a100.sh \
  --root-dir /data1/speech/nhandt23/06_thang \
  --env-file /data1/speech/nhandt23/.env \
  --gpus 0,1,2,3 \
  > /data1/speech/nhandt23/06_thang/CapSpeech/train_logs/stage2_v3/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 3. Resume

```bash
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file accelerate_config.yaml \
  finetune.py \
  --config-name configs/finetune_vn_stage2_v3.yaml \
  --pretrained-ckpt /tmp/nhandt23/capspeech/hf_models/.../checkpoint.pt \
  --train-sampler stage2_v3 \
  --sampler-recipe configs/stage2_v3_recipe.yaml \
  --resume-from /tmp/capspeech_train_v3/ckpts/finetune_captts_vn_stage2_v3/5000.pt \
  --epochs 5 \
  --num-workers 8 \
  --eval-every-step 500 \
  --save-every-step 500 \
  --max-ckpts 3 \
  --log-dir /tmp/capspeech_train_v3/logs \
  --save-dir /tmp/capspeech_train_v3/ckpts \
  --amp fp16
```

## 4. Important paths

- base snapshot: `/tmp/nhandt23/capspeech/hf_models/`
- processed data: `/tmp/capspeech_data/vn_capspeech_stage2_v3`
- checkpoints: `/tmp/capspeech_train_v3/ckpts/finetune_captts_vn_stage2_v3`
- logs: `/data1/speech/nhandt23/06_thang/CapSpeech/train_logs/stage2_v3`
