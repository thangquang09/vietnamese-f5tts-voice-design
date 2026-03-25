# CapSpeech Stage2-v3 HF Release

Repo dich:

- `thangquang09/capspeech-nar-vietnamese-stage2-v3`

Base repo dung de pull:

- `thangquang09/capspeech-nar-vietnamese`

## Token

Script doc `HF_TOKEN` theo uu tien:

1. env hien tai
2. `.env` truyen vao
3. `.env` tai `/data1/speech/nhandt23/.env`
4. `.env` tai `/data1/speech/nhandt23/06_thang/.env`

## Push

```bash
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar

python push_to_hf_v3.py \
  --ckpt /tmp/capspeech_train_v3/ckpts/finetune_captts_vn_stage2_v3/5000.pt \
  --repo-name capspeech-nar-vietnamese-stage2-v3 \
  --username thangquang09 \
  --config /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar/configs/finetune_vn_stage2_v3.yaml \
  --vocab /tmp/capspeech_data/vn_capspeech_stage2_v3/vocab.txt \
  --duration-predictor-dir /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar/phobert_duration_predictor \
  --env-file /data1/speech/nhandt23/.env
```

Bundle upload:

- `checkpoint.pt`
- `finetune_vn_stage2_v3.yaml`
- `finetune_vn_stage2.yaml` (compat copy)
- `vocab.txt`
- `duration_predictor/`
- `README.md`
