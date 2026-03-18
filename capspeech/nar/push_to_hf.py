#!/usr/bin/env python3
"""Push CapSpeech NAR Vietnamese checkpoint to Hugging Face Hub."""

import argparse
import os
import json
import shutil
from huggingface_hub import HfApi, create_repo

def create_model_card(save_dir, args):
    """Create a README.md model card."""
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
base_model: OpenSound/CapSpeech-models
---

# CapSpeech NAR — Vietnamese Instruction TTS

Vietnamese-adapted CapSpeech NAR (Non-Autoregressive) model for instruction-guided text-to-speech synthesis.

## Model Details

- **Base model**: [OpenSound/CapSpeech-models](https://huggingface.co/OpenSound/CapSpeech-models) (`nar_CapTTS.pt`)
- **Architecture**: CrossDiT (dim=1024, depth=24, heads=16, ff_mult=4)
- **Vocoder**: BigVGAN v2 24kHz 100-band
- **Text encoder**: Character-level Vietnamese (176 chars)
- **Caption encoder**: ViT5-large (VietAI/vit5-large, dim=1024)
- **Parameters**: 614.10M trainable
- **Training steps**: {args.steps} optimizer updates
- **Training data**: ~1.05M Vietnamese speech samples (general task)
- **Loss**: Flow Matching (MSE)

## Training Config

| Parameter | Value |
|:----------|:------|
| Learning rate | 2e-5 |
| Batch size | 32 × 2 GPUs |
| Gradient accumulation | 2 |
| Effective batch | 128 |
| Mixed precision | fp16 |
| Optimizer | AdamW |
| GPU | 2× NVIDIA A100 40GB |

## Usage

```python
# Load checkpoint
import torch
from capspeech.nar.network.crossdit import CrossDiT

model = CrossDiT(
    dim=1024, depth=24, heads=16, ff_mult=4,
    text_dim=512, conv_layers=0,
    text_num_embeds=176, mel_dim=100,
    t5_dim=1024, clap_dim=512,
    use_checkpoint=False, qk_norm=True, skip=True
)

ckpt = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
```

## Files

- `checkpoint.pt` — Model + optimizer state dict
- `finetune_vn.yaml` — Training config
- `vocab.txt` — Vietnamese character vocabulary (176 chars)

## Citation

Based on [CapSpeech](https://github.com/WangHelin1997/CapSpeech).
"""
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(card)


def main():
    parser = argparse.ArgumentParser(description="Push CapSpeech NAR checkpoint to HF Hub")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--repo-name", type=str, default="capspeech-nar-vietnamese",
                        help="HF repo name (without username)")
    parser.add_argument("--username", type=str, default="thangquang09")
    parser.add_argument("--steps", type=int, default=4000, help="Training steps for model card")
    parser.add_argument("--config", type=str, 
                        default="/data1/speech/nhandt23/06_thang/CapSpeech/capspeech/nar/configs/finetune_vn.yaml")
    parser.add_argument("--vocab", type=str,
                        default="/tmp/capspeech_data/vn_capspeech/vocab.txt")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.repo_name}"
    
    # Login
    print("=" * 50)
    print(f"Pushing to: https://huggingface.co/{repo_id}")
    print("=" * 50)
    
    from huggingface_hub import login
    login()  # Will prompt for token
    
    api = HfApi()
    
    # Create repo
    print(f"\n[1/4] Creating repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)
    
    # Prepare upload dir
    upload_dir = "/tmp/capspeech_hf_upload"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Copy checkpoint
    print(f"[2/4] Copying checkpoint: {args.ckpt}")
    ckpt_name = "checkpoint.pt"
    shutil.copy2(args.ckpt, os.path.join(upload_dir, ckpt_name))
    
    # Copy config
    if os.path.exists(args.config):
        print(f"[3/4] Copying config: {args.config}")
        shutil.copy2(args.config, os.path.join(upload_dir, "finetune_vn.yaml"))
    
    # Copy vocab
    if os.path.exists(args.vocab):
        print(f"       Copying vocab: {args.vocab}")
        shutil.copy2(args.vocab, os.path.join(upload_dir, "vocab.txt"))
    
    # Create model card
    create_model_card(upload_dir, args)
    print(f"[3/4] Created model card")
    
    # Upload
    print(f"[4/4] Uploading to {repo_id}...")
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload CapSpeech NAR Vietnamese checkpoint (step {args.steps})",
    )
    
    print(f"\n✅ Done! View at: https://huggingface.co/{repo_id}")
    
    # Cleanup
    shutil.rmtree(upload_dir)


if __name__ == "__main__":
    main()
