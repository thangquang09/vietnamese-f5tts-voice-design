#!/usr/bin/env python3
"""Push CapSpeech NAR Vietnamese Stage 2 checkpoint to a NEW HuggingFace Hub repo.

Preserves the original repo (capspeech-nar-vietnamese) untouched.

Usage (from any machine with NFS access):
    # First copy checkpoint to NFS on DGX:
    #   cp /tmp/capspeech_train/ckpts_stage2/finetune_captts_vn_stage2/1500.pt \
    #      /data1/speech/nhandt23/06_thang/capspeech_stage2/1500.pt

    python push_to_hf_stage2.py \
        --ckpt /mnt/speech/nhandt23/06_thang/capspeech_stage2/1500.pt \
        --private
"""

import argparse
import os
import shutil
from huggingface_hub import HfApi, create_repo, login


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Defaults — adjust paths for your machine
DEFAULT_CKPT = "/mnt/speech/nhandt23/06_thang/capspeech_stage2/1500.pt"
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "configs", "finetune_vn_stage2.yaml")
DEFAULT_VOCAB = "/mnt/speech/nhandt23/06_thang/capspeech_stage2/vocab.txt"
DEFAULT_DURATION_PREDICTOR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--thangquang09--capspeech-nar-vietnamese/"
    "snapshots/2354d63c53474c0e044b0a4c45113c9bc5272cb5/duration_predictor"
)


def create_model_card(save_dir, args):
    """Create a README.md model card for Stage 2."""
    card = f"""---
language: vi
license: mit
tags:
  - tts
  - text-to-speech
  - vietnamese
  - capspeech
  - flow-matching
  - instruction-tts
  - emotion
  - accent
base_model: thangquang09/capspeech-nar-vietnamese
---

# CapSpeech NAR — Vietnamese Instruction TTS (Stage 2)

Vietnamese-adapted CapSpeech NAR model, **Stage 2** fine-tuned for emotion, accent, and age-group control.

## Model Details

- **Stage 1 base**: [thangquang09/capspeech-nar-vietnamese](https://huggingface.co/thangquang09/capspeech-nar-vietnamese)
- **Architecture**: CrossDiT (dim=1024, depth=24, heads=16, ff_mult=4)
- **Vocoder**: BigVGAN v2 24kHz 100-band
- **Text encoder**: Character-level Vietnamese (176 chars)
- **Caption encoder**: ViT5-large (VietAI/vit5-large, dim=1024)
- **Parameters**: 614.10M trainable
- **Stage 2 training steps**: {args.steps}
- **Stage 2 data**: ~202K mixed samples (emotion, accent, age groups)
- **Loss**: Flow Matching (MSE)

## Stage 2 Training

| Parameter | Value |
|:----------|:------|
| Learning rate | 5e-6 (1/4 of Stage 1) |
| Warmup | 500 steps |
| Batch size | 32 × 2 GPUs |
| Gradient accumulation | 2 |
| Effective batch | 128 |
| Mixed precision | fp16 |
| GPU | 2× NVIDIA A100 40GB |

### Data Mix

| Source | Samples |
|:-------|:--------|
| Emotion (×10 upsample) | ~21K |
| Accent | ~47K |
| General → Senior (×2) | ~60K |
| General → Youth (×5) | ~15K |
| General → Children (×10) | ~9K |
| General → Adult (rehearsal) | ~50K |
| **Total** | **~202K** |

## Files

- `checkpoint.pt` — CrossDiT model weights (Stage 2, step {args.steps})
- `finetune_vn_stage2.yaml` — Stage 2 training config
- `vocab.txt` — Vietnamese character vocabulary (176 chars)
- `duration_predictor/` — PhoBERT duration predictor

## Usage

```python
from api import InstructVoiceAPI

tts = InstructVoiceAPI.from_local(
    ckpt_path="checkpoint.pt",
    config_path="finetune_vn_stage2.yaml",
    vocab_path="vocab.txt",
)

tts.synthesize(
    text="Xin chào, tôi rất vui được gặp bạn!",
    caption="Giọng nữ trẻ, vui vẻ, phấn khích, nhịp nói nhanh",
    output_path="output.wav",
)
```

## Citation

Based on [CapSpeech](https://github.com/WangHelin1997/CapSpeech).
"""
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(card)


def main():
    parser = argparse.ArgumentParser(description="Push Stage 2 checkpoint to NEW HF repo")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="Path to Stage 2 checkpoint .pt file")
    parser.add_argument("--repo-name", type=str, default="capspeech-nar-vietnamese-stage2",
                        help="HF repo name (default: capspeech-nar-vietnamese-stage2)")
    parser.add_argument("--username", type=str, default="thangquang09")
    parser.add_argument("--steps", type=int, default=1500,
                        help="Training steps for model card")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB)
    parser.add_argument("--duration-predictor", type=str, default=DEFAULT_DURATION_PREDICTOR)
    parser.add_argument("--private", action="store_true", help="Create private repo")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.repo_name}"

    # Verify checkpoint exists
    if not os.path.exists(args.ckpt):
        print(f"❌ Checkpoint not found: {args.ckpt}")
        print(f"   If on dev-playground, first copy on DGX:")
        print(f"   cp /tmp/capspeech_train/ckpts_stage2/finetune_captts_vn_stage2/1500.pt \\")
        print(f"      /data1/speech/nhandt23/06_thang/capspeech_stage2/1500.pt")
        return

    print("=" * 60)
    print(f"  Pushing Stage 2 to NEW repo: {repo_id}")
    print(f"  (Original repo capspeech-nar-vietnamese is untouched)")
    print("=" * 60)

    login()
    api = HfApi()

    # Create NEW repo
    print(f"\n[1/5] Creating repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)

    # Staging directory
    upload_dir = "/tmp/capspeech_hf_upload_stage2"
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir)

    # Copy checkpoint
    print(f"[2/5] Copying checkpoint: {args.ckpt}")
    shutil.copy2(args.ckpt, os.path.join(upload_dir, "checkpoint.pt"))

    # Copy config
    if os.path.exists(args.config):
        print(f"[3/5] Copying config: {args.config}")
        shutil.copy2(args.config, os.path.join(upload_dir, "finetune_vn_stage2.yaml"))
    else:
        print(f"[3/5] ⚠️ Config not found: {args.config}")

    # Copy vocab
    if os.path.exists(args.vocab):
        print(f"       Copying vocab: {args.vocab}")
        shutil.copy2(args.vocab, os.path.join(upload_dir, "vocab.txt"))
    else:
        print(f"       ⚠️ Vocab not found: {args.vocab}")

    # Copy duration predictor
    dp_dest = os.path.join(upload_dir, "duration_predictor")
    if os.path.isdir(args.duration_predictor):
        print(f"[4/5] Copying duration predictor: {args.duration_predictor}")
        shutil.copytree(args.duration_predictor, dp_dest)
    else:
        print(f"[4/5] ⚠️ Duration predictor not found: {args.duration_predictor}")

    # Model card
    create_model_card(upload_dir, args)
    print(f"       Created model card")

    # List files to upload
    print(f"\n📦 Files to upload:")
    for root, dirs, files in os.walk(upload_dir):
        for f in files:
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            rel = os.path.relpath(fpath, upload_dir)
            print(f"   {rel} ({size_mb:.1f} MB)")

    # Upload
    print(f"\n[5/5] Uploading to {repo_id}...")
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload CapSpeech NAR Vietnamese Stage 2 (step {args.steps})",
    )

    print(f"\n{'='*60}")
    print(f"✅ Done! View at: https://huggingface.co/{repo_id}")
    print(f"   Original repo untouched: https://huggingface.co/{args.username}/capspeech-nar-vietnamese")
    print(f"{'='*60}")

    # Cleanup
    shutil.rmtree(upload_dir)


if __name__ == "__main__":
    main()
