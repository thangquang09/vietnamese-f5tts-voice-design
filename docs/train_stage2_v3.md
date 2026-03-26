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
  --stage1-dir /tmp/capspeech_data/vn_capspeech \
  --gpus 0,1,2,3 \
  --num-workers 16
```

Important:
- `process_stage2_v3.sh` gio reuse `vocab.txt` tu stage1, khong rebuild vocab moi.
- Script nay cung regenerate lai `g2p/` va `t5/` de tranh reuse artifact cu.
- `segment_id` v3 da duoc namespace theo dataset, vi du `lsvsc__2`, de tranh collision giua cac source.

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

Note:
- `train_stage2_v3_a100.sh` la launcher all-in-one. Neu da build va preprocess xong, khong nen dung lai script nay vi no se build lai tu dau.
- Sau khi fix bug double-sharding trong `finetune.py`, run `stage2_v3` tren `4xA100` voi `virtual_epoch_samples=100000` va `batch_size=16` phai ra khoang `1563 steps/epoch`, khong con `391`.
- Mac dinh nen train moi tu checkpoint stage1 tu Hugging Face, khong resume tu `500.pt`/`600.pt` cu.

## 3. Direct launch when data is already ready

```bash
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar
mkdir -p /data1/speech/nhandt23/06_thang/CapSpeech/train_logs/stage2_v3

nohup bash -lc '
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file accelerate_config.yaml \
  finetune.py \
  --config-name configs/finetune_vn_stage2_v3.yaml \
  --pretrained-ckpt /tmp/nhandt23/capspeech/hf_models/models--thangquang09--capspeech-nar-vietnamese/snapshots/2354d63c53474c0e044b0a4c45113c9bc5272cb5/checkpoint.pt \
  --train-sampler stage2_v3 \
  --sampler-recipe configs/stage2_v3_recipe.yaml \
  --epochs 5 \
  --num-workers 8 \
  --eval-every-step 1000000 \
  --save-every-step 500 \
  --max-ckpts 3 \
  --log-dir /tmp/capspeech_train_v3/logs \
  --save-dir /tmp/capspeech_train_v3/ckpts \
  --amp fp16
' > /data1/speech/nhandt23/06_thang/CapSpeech/train_logs/stage2_v3/train_direct_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 4. Resume only if really needed

```bash
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar

nohup bash -lc '
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file accelerate_config.yaml \
  finetune.py \
  --config-name configs/finetune_vn_stage2_v3.yaml \
  --pretrained-ckpt /tmp/nhandt23/capspeech/hf_models/models--thangquang09--capspeech-nar-vietnamese/snapshots/2354d63c53474c0e044b0a4c45113c9bc5272cb5/checkpoint.pt \
  --train-sampler stage2_v3 \
  --sampler-recipe configs/stage2_v3_recipe.yaml \
  --resume-from /tmp/capspeech_train_v3/ckpts/finetune_captts_vn_stage2_v3/500.pt \
  --epochs 5 \
  --num-workers 4 \
  --eval-every-step 1000000 \
  --save-every-step 200 \
  --max-ckpts 3 \
  --log-dir /tmp/capspeech_train_v3/logs \
  --save-dir /tmp/capspeech_train_v3/ckpts \
  --amp fp16
' > /data1/speech/nhandt23/06_thang/CapSpeech/train_logs/stage2_v3/train_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 5. Important paths

- base snapshot: `/tmp/nhandt23/capspeech/hf_models/`
- processed data: `/tmp/capspeech_data/vn_capspeech_stage2_v3`
- checkpoints: `/tmp/capspeech_train_v3/ckpts/finetune_captts_vn_stage2_v3`
- logs: `/data1/speech/nhandt23/06_thang/CapSpeech/train_logs/stage2_v3`

## 6. Issues seen on A100

- `prepare_clap_none.py` co the fail voi loi `flash_attn_2_cuda ... undefined symbol`. Day la loi ABI cua env, khong phai loi data.
- Workaround nhanh da verify:
  - copy `/tmp/capspeech_data/vn_capspeech/clap_embs/none.npz`
  - vao `/tmp/capspeech_data/vn_capspeech_stage2_v3/clap_embs/none.npz`
- Root cause ky thuat cua run loi truoc do:
  - custom `stage2_v3` sampler da shard theo rank
  - `accelerator.prepare(train_loader)` shard them lan nua
  - ket qua la `391 steps/epoch` thay vi `~1563`
- Root cause data quan trong:
  - v3 cu da rebuild `vocab.txt`, dan den text token id co the lech voi stage1 checkpoint
  - v3 cu cung co nguy co reuse `g2p/t5` neu build lai trong cung save dir
- Truoc khi train, can verify:
  - `train missing_g2p=0`
  - `train missing_t5=0`
  - `val missing_g2p=0`
  - `val missing_t5=0`
- Neu data da san sang, dung `finetune.py` truc tiep thay vi `train_stage2_v3_a100.sh`.
- `eval` trong luc DDP train co the gay NCCL watchdog timeout:
  - rank 0 chay `eval_model(...)`
  - cac rank con lai dung o `accelerator.wait_for_everyone()`
  - neu eval vuot `600s` thi job se bi giet voi loi `Watchdog caught collective operation timeout`
- Practical workaround cho run 4xA100 hien tai:
  - tat in-train eval bang cach dat `--eval-every-step` rat lon, vi du `1000000`
  - van giu `--save-every-step 200`
  - danh gia checkpoint offline sau
