#!/usr/bin/env python3
"""
Standalone inference script for CapSpeech NAR Vietnamese model.

Usage:
    python inference_vn.py \
        --ckpt /tmp/capspeech_train/ckpts/finetune_captts_vn/2000.pt \
        --config configs/finetune_vn.yaml \
        --output-dir ./inference_output \
        --num-samples 5
"""

import os
import sys
import argparse
import torch
import soundfile as sf
from tqdm import tqdm
from einops import rearrange

import bigvgan

from model.modules import MelSpec
from network.crossdit import CrossDiT
from dataset.capspeech import CapSpeech
from utils import load_yaml_with_includes, make_pad_mask
from inference import sample

from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="CapSpeech NAR Vietnamese Inference")
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--config', type=str, default='configs/finetune_vn.yaml',
                        help='Path to config yaml')
    parser.add_argument('--output-dir', type=str, default='./inference_output',
                        help='Directory to save generated wav files')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=25,
                        help='ODE solver steps (more = better quality, slower)')
    parser.add_argument('--cfg', type=float, default=2.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to use: val or train')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    params = load_yaml_with_includes(args.config)
    print(f"Config loaded: {args.config}")
    print(f"Model: {params['model_name']}")

    # Load model
    model = CrossDiT(**params['model'])
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f"  → Epoch: {ckpt.get('epoch', '?')}, Global step: {ckpt.get('global_step', '?')}")
    
    model = model.to(args.device).eval()

    # Load vocoder (BigVGAN)
    print("Loading BigVGAN vocoder...")
    vocoder = bigvgan.BigVGAN.from_pretrained(
        'nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False
    )
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(args.device)

    # Mel spectrogram
    mel = MelSpec(**params['mel']).to(args.device)
    latent_sr = params['mel']['target_sample_rate'] / params['mel']['hop_length']

    # Load validation dataset
    dataset_params = params['data']['valset'].copy()
    dataset_params['split'] = args.split
    val_set = CapSpeech(**dataset_params)
    val_loader = DataLoader(
        val_set, num_workers=0, batch_size=1,
        shuffle=False, collate_fn=val_set.collate
    )

    print(f"\nGenerating {args.num_samples} samples from '{args.split}' split...")
    print(f"  Steps: {args.steps}, CFG: {args.cfg}")
    print(f"  Output: {args.output_dir}/")
    print()

    generated = 0
    for step, batch in enumerate(tqdm(val_loader, total=args.num_samples)):
        if generated >= args.num_samples:
            break

        # Prepare batch (same as in finetune.py)
        x, x_lens = batch["x"].to(args.device), batch["x_lens"].to(args.device)
        y, y_lens = batch["y"].to(args.device), batch["y_lens"].to(args.device)
        c, c_lens = batch["c"].to(args.device), batch["c_lens"].to(args.device)
        tag = batch["tag"].to(args.device)

        # Add len for clap embedding
        x_lens = x_lens + 1

        with torch.no_grad():
            audio_clip = mel(y)
            audio_clip = rearrange(audio_clip, 'b d n -> b n d')
            y_lens = (y_lens * latent_sr).long()

        cond = None

        prompt = c  # T5 caption embeddings as "prompt"
        prompt_lens = c_lens
        clap = tag  # CLAP embeddings
        text = x    # Phoneme sequence

        seq_len_prompt = prompt.shape[1]
        prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)

        # Generate
        gen_wav = sample(
            model, vocoder,
            audio_clip, cond, text, prompt, clap, prompt_mask,
            steps=args.steps, cfg=args.cfg,
            sway_sampling_coef=-1.0, device=args.device
        )

        # Save generated audio
        out_path = os.path.join(args.output_dir, f'gen_{step:04d}.wav')
        sf.write(out_path, gen_wav, samplerate=params['mel']['target_sample_rate'])

        # Also save ground truth for comparison
        gt_wav = y.squeeze().cpu().numpy()
        gt_path = os.path.join(args.output_dir, f'gt_{step:04d}.wav')
        sf.write(gt_path, gt_wav, samplerate=params['mel']['target_sample_rate'])

        print(f"  [{step}] Generated: {out_path} | GT: {gt_path}")
        generated += 1

    print(f"\nDone! Generated {generated} samples in {args.output_dir}/")


if __name__ == '__main__':
    main()
