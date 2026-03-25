# CapSpeech Stage2-v3 Data

`stage2-v3` dung hop nhieu pool nhung giu `unique rows` trong JSON, sau do sampler o train-time enforce ti le:

- `50%` `replay_general`
- `20%` `accent_vimd_full`
- `10%` `accent_lsvsc_bal`
- `15%` `age_main`
- `5%` `emotion_gt`

## Pools

- `replay_general`: `50k` Vivoice + `50k` Dolly cho train, `2.5k + 2.5k` cho val.
- `accent_vimd_full`: giu full ViMD.
- `accent_lsvsc_bal`: LSVSC can bang theo vung mien.
- `age_main`: `teen/adult/senior` + `child_aux` low-weight.
- `emotion_gt`: toan bo emotion GT tu `ViSEC + VNEMOS`.

## Build

```bash
cd /data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar

python data_preprocessing/build_stage2_data_v3.py \
  --csv-dir /data1/speech/nhandt23/06_thang/vn-instructiontts/results/final_dataset \
  --save-dir /tmp/capspeech_data/vn_capspeech_stage2_v3 \
  --recipe configs/stage2_v3_recipe.yaml \
  --caption-column caption_full \
  --mount-remap /mnt/speech:/data1/speech \
  --dry-run
```

Build that:

```bash
python data_preprocessing/build_stage2_data_v3.py \
  --csv-dir /data1/speech/nhandt23/06_thang/vn-instructiontts/results/final_dataset \
  --save-dir /tmp/capspeech_data/vn_capspeech_stage2_v3 \
  --recipe configs/stage2_v3_recipe.yaml \
  --caption-column caption_full \
  --mount-remap /mnt/speech:/data1/speech
```

Outputs:

- `/tmp/capspeech_data/vn_capspeech_stage2_v3/jsons/{train,val}.json`
- `/tmp/capspeech_data/vn_capspeech_stage2_v3/manifest/{train,val}.txt`
- `/tmp/capspeech_data/vn_capspeech_stage2_v3/sampling_metadata/*`
- `/tmp/capspeech_data/vn_capspeech_stage2_v3/summary_v3.json`
