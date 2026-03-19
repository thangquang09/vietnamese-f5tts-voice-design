#!/usr/bin/env python3
"""
Benchmark: compare single-sample loop vs true batch inference.

Measures:
1. RTF (Real Time Factor) across batch_size=1 (loop), 4, 8, 16
2. MSE (Signal Integrity) between batch=1 and batch>1 mel outputs

Usage:
    python benchmark_performance.py \
        --num_samples 8 \
        --batch_sizes 1 4 8 \
        --steps 10 \
        --device cuda:4

Checkpoint is loaded from HuggingFace (same as api.py).
"""

import os
import sys
import math
import time
import json
import random
import base64
import argparse
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

# Ensure repo root is on sys.path so 'capspeech' package is importable
import sys, pathlib
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import T5EncoderModel, T5Tokenizer

from capspeech.nar import bigvgan
from capspeech.nar.network.crossdit import CrossDiT
from capspeech.nar.inference import sample as single_sample
from capspeech.nar.batch_inference import batch_sample, batch_sample_mel
from capspeech.nar.utils import load_yaml_with_includes, make_pad_mask


# ──────────────────────── Constants ──────────────────────── #

HF_MODEL_REPO = "thangquang09/capspeech-nar-vietnamese"
BIGVGAN_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"
CAPTION_MODEL = "VietAI/vit5-large"
SAMPLE_RATE = 24000
HOP_LENGTH = 256

_CLAP_NONE_B64 = "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"


# ──────────────────────── Dataset Loading ──────────────────────── #

DEFAULT_DATASET_CSV = "/mnt/speech/nhandt23/06_thang/vn-instructiontts/results/final_dataset/test.csv"


def load_samples_from_csv(csv_path: str, num_samples: int, seed: int = 42) -> list:
    """Load real samples from vn-instructiontts CSV dataset.
    
    Reads transcript and caption_partial columns.
    Duration will be predicted by the model (not from WAV ground truth).
    Randomly samples N items (with seed for reproducibility).
    """
    import csv
    
    print(f"Loading samples from {csv_path}...")
    
    all_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcript = row.get("transcript", "")
            caption = row.get("caption_partial", row.get("caption_full", ""))
            
            if not transcript or not caption:
                continue
                
            all_rows.append({
                "text": transcript,
                "caption": caption,
                "id": row.get("id", ""),
                "dataset": row.get("dataset", ""),
            })
    
    print(f"  Found {len(all_rows)} valid samples in CSV")
    
    # Randomly sample
    rng = random.Random(seed)
    if num_samples < len(all_rows):
        sampled = rng.sample(all_rows, num_samples)
    else:
        sampled = all_rows[:num_samples]
    
    print(f"  Selected {len(sampled)} samples")
    return sampled


def text_to_chars(text: str) -> list:
    chars = []
    for ch in text.lower():
        if ch == " ":
            chars.append("<BLK>")
        else:
            chars.append(ch)
    return chars


# ──────────────────────── Model Loading ──────────────────────── #

def load_models(device: str):
    """Load models from HuggingFace."""
    print("Loading models from HuggingFace...")

    ckpt_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="checkpoint.pt")
    config_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="finetune_vn.yaml")
    vocab_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="vocab.txt")

    params = load_yaml_with_includes(config_path)
    model = CrossDiT(**params['model'])
    checkpoint = torch.load(ckpt_path, map_location=device)['model']
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device).eval()
    print("  ✓ CrossDiT loaded")

    with open(vocab_path, "r", encoding="utf-8") as f:
        lines = [l.strip().split(" ", 1) for l in f if l.strip()]
        phn2num = {item[1]: int(item[0]) for item in lines}
    print(f"  ✓ Vocab loaded ({len(phn2num)} chars)")

    try:
        vocoder = bigvgan.BigVGAN.from_pretrained(BIGVGAN_REPO, use_cuda_kernel=False)
    except TypeError:
        vocoder_dir = snapshot_download(repo_id=BIGVGAN_REPO)
        h = bigvgan.load_hparams_from_json(os.path.join(vocoder_dir, "config.json"))
        vocoder = bigvgan.BigVGAN(h, use_cuda_kernel=False)
        ckpt = torch.load(os.path.join(vocoder_dir, "bigvgan_generator.pt"), map_location="cpu")
        vocoder.load_state_dict(ckpt["generator"])
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    print("  ✓ BigVGAN loaded")

    caption_tokenizer = T5Tokenizer.from_pretrained(CAPTION_MODEL)
    caption_encoder = T5EncoderModel.from_pretrained(CAPTION_MODEL).to(device).eval()
    print("  ✓ ViT5-large loaded")

    clap_bytes = base64.b64decode(_CLAP_NONE_B64)
    clap_np = np.frombuffer(clap_bytes, dtype=np.float32)
    clap_none = torch.from_numpy(clap_np.copy()).to(device)
    print("  ✓ CLAP loaded")

    # Duration predictor (PhoBERT)
    duration_model = None
    duration_tokenizer = None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        dp_dir = snapshot_download(
            repo_id=HF_MODEL_REPO, allow_patterns="duration_predictor/*")
        dp_subdir = os.path.join(dp_dir, "duration_predictor")
        if os.path.isdir(dp_subdir):
            duration_tokenizer = AutoTokenizer.from_pretrained(dp_subdir)
            duration_model = AutoModelForSequenceClassification.from_pretrained(
                dp_subdir, num_labels=1).to(device).eval()
            print("  ✓ Duration predictor loaded")
    except Exception as e:
        print(f"  ⚠️ Duration predictor not available: {e}")
        print("      → Using heuristic estimation")

    return model, vocoder, phn2num, caption_tokenizer, caption_encoder, clap_none, \
           duration_model, duration_tokenizer


# ──────────────────────── Prepare Inputs ──────────────────────── #

def prepare_inputs(items, phn2num, caption_tokenizer, caption_encoder,
                   clap_none, device, duration_model=None, duration_tokenizer=None):
    """Prepare all inputs for both single and batch inference.
    
    Duration is predicted by PhoBERT duration predictor (if available)
    or estimated by word-count heuristic. NOT from ground truth audio.
    """
    prepared = []
    
    with torch.no_grad():
        for item in items:
            text = item['text']
            caption = item['caption']
            
            # Text encoding
            chars = text_to_chars(text)
            chars = [ch for ch in chars if ch in phn2num]
            text_tokens = [phn2num[ch] for ch in chars]
            text_tensor = torch.LongTensor(text_tokens).to(device)
            
            # Caption encoding
            batch_enc = caption_tokenizer(caption, return_tensors="pt")
            input_ids = batch_enc["input_ids"].to(device)
            prompt = caption_encoder(input_ids=input_ids).last_hidden_state.squeeze(0)
            prompt_len = prompt.shape[0]
            
            # Duration — use predictor, NOT ground truth
            if duration_model is not None and duration_tokenizer is not None:
                combined = f"{caption} [SEP] {text}"
                inputs = duration_tokenizer(
                    combined, return_tensors="pt",
                    padding="max_length", truncation=True, max_length=256
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = duration_model(**inputs)
                duration = output.logits.squeeze().item()
                duration = max(duration, 0.5)
                duration = min(duration, 30.0)
            else:
                # Heuristic fallback: ~3 words/sec
                words = text.strip().split()
                duration = max(len(words) / 3.0, 1.0)
            
            n_frames = math.ceil(duration * SAMPLE_RATE / HOP_LENGTH)
            
            # Mel target placeholder
            x = torch.zeros(n_frames, 100, device=device)
            
            prepared.append({
                'text_tensor': text_tensor,
                'prompt': prompt,
                'prompt_len': prompt_len,
                'clap': clap_none.clone(),
                'x': x,
                'n_frames': n_frames,
                'duration_sec': duration,
                'text': text,
                'caption': caption,
                'id': item.get('id', ''),
                'dataset': item.get('dataset', ''),
            })
    
    return prepared


# ──────────────────────── Single-sample Loop Baseline ──────────────────────── #

def run_loop_baseline(prepared, model, vocoder, device, steps, cfg, seed):
    """Run inference one sample at a time using the original sample() function."""
    results = []
    total_time = 0.0
    total_audio_sec = 0.0
    
    for item in prepared:
        x = item['x'].unsqueeze(0)        # [1, n, d]
        text = item['text_tensor'].unsqueeze(0)  # [1, nt]
        prompt = item['prompt'].unsqueeze(0)     # [1, n_prompt, t5_dim]
        clap = item['clap'].unsqueeze(0)         # [1, clap_dim]
        prompt_len = item['prompt_len']
        
        prompt_lens = torch.Tensor([prompt_len])
        prompt_mask = make_pad_mask(prompt_lens, prompt.shape[1]).to(device)
        
        if seed is not None:
            torch.manual_seed(seed)
        
        start = time.time()
        wav = single_sample(
            model, vocoder,
            x, None, text, prompt, clap, prompt_mask,
            steps=steps, cfg=cfg,
            sway_sampling_coef=-1.0, device=device
        )
        elapsed = time.time() - start
        
        total_time += elapsed
        audio_sec = len(wav) / SAMPLE_RATE
        total_audio_sec += audio_sec
        results.append(wav)
    
    rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0
    return results, total_time, total_audio_sec, rtf


# ──────────────────────── Batch Inference ──────────────────────── #

def run_batch(prepared, model, vocoder, device, batch_size, steps, cfg, seed,
              sort_by_duration=True, verbose=True):
    """Run true batch inference with optional duration sorting."""
    total_time = 0.0
    total_audio_sec = 0.0
    
    n = len(prepared)
    
    # Sort by duration to minimize padding waste within each batch
    indexed = list(enumerate(prepared))
    if sort_by_duration:
        indexed.sort(key=lambda x: x[1]['n_frames'])
    
    results_ordered = [None] * n
    num_batches = math.ceil(n / batch_size)
    
    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, n)
        batch_indexed = indexed[start_i:end_i]
        
        orig_indices = [idx for idx, _ in batch_indexed]
        batch = [item for _, item in batch_indexed]
        
        x_list = [item['x'] for item in batch]
        text_list = [item['text_tensor'] for item in batch]
        prompt_list = [item['prompt'] for item in batch]
        clap_list = [item['clap'] for item in batch]
        prompt_lens_list = [item['prompt_len'] for item in batch]
        duration_frames_list = [item['n_frames'] for item in batch]
        
        # Padding efficiency stats
        max_frames = max(duration_frames_list)
        total_frames = sum(duration_frames_list)
        padded_total = max_frames * len(batch)
        efficiency = total_frames / padded_total * 100 if padded_total > 0 else 100
        
        if verbose:
            frame_range = f"{min(duration_frames_list)}–{max_frames}"
            print(f"    Batch {batch_idx+1}/{num_batches}: "
                  f"{len(batch)} items, frames {frame_range}, "
                  f"pad efficiency {efficiency:.0f}%")
        
        start = time.time()
        wavs = batch_sample(
            model=model,
            vocoder=vocoder,
            x_list=x_list,
            cond=None,
            text_list=text_list,
            prompt_list=prompt_list,
            clap_list=clap_list,
            prompt_lens_list=prompt_lens_list,
            duration_frames_list=duration_frames_list,
            steps=steps,
            cfg=cfg,
            sway_sampling_coef=-1.0,
            device=device,
            seed=seed,
        )
        elapsed = time.time() - start
        
        total_time += elapsed
        batch_audio = sum(len(w) / SAMPLE_RATE for w in wavs)
        total_audio_sec += batch_audio
        
        for j, wav in enumerate(wavs):
            results_ordered[orig_indices[j]] = wav
    
    rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0
    return results_ordered, total_time, total_audio_sec, rtf


# ──────────────────────── MSE Comparison ──────────────────────── #

def compute_mse(wavs_a, wavs_b):
    """Compute MSE between two lists of waveforms (on overlapping region)."""
    mses = []
    for wa, wb in zip(wavs_a, wavs_b):
        min_len = min(len(wa), len(wb))
        if min_len == 0:
            continue
        mse = np.mean((wa[:min_len] - wb[:min_len]) ** 2)
        mses.append(mse)
    return mses


# ──────────────────────── Main ──────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Benchmark: single vs batch inference")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of test samples to randomly select")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4, 8],
                        help="Batch sizes to test")
    parser.add_argument("--steps", type=int, default=10,
                        help="ODE steps (lower for faster benchmark)")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup_runs", type=int, default=1,
                        help="Warmup runs before timing")
    parser.add_argument("--save_audio", action="store_true",
                        help="Save generated audio for manual comparison")
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument("--dataset_csv", type=str, default=DEFAULT_DATASET_CSV,
                        help="Path to vn-instructiontts CSV (test.csv or val.csv)")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    device = args.device
    
    # Load real samples from dataset
    test_items = load_samples_from_csv(args.dataset_csv, args.num_samples, seed=args.seed)
    num_samples = len(test_items)
    
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: Single vs Batch Inference")
    print(f"  Samples: {num_samples}, Steps: {args.steps}, Device: {device}")
    print(f"  Batch sizes to test: {args.batch_sizes}")
    print(f"{'='*60}\n")

    # Load models
    model, vocoder, phn2num, cap_tok, cap_enc, clap_none, \
        dur_model, dur_tok = load_models(device)
    
    # Prepare inputs
    print("\nPreparing inputs...")
    prepared = prepare_inputs(test_items, phn2num, cap_tok, cap_enc, clap_none, device,
                              duration_model=dur_model, duration_tokenizer=dur_tok)
    
    for i, p in enumerate(prepared):
        print(f"  [{i}] {p['n_frames']:>5} frames ({p['duration_sec']:>5.2f}s) "
              f"[{p.get('dataset','')}] | {p['text'][:40]}...")
    
    # ── Duration distribution ──
    durations = sorted([p['duration_sec'] for p in prepared])
    print(f"\n📊 Duration Distribution (N={len(durations)})")
    print(f"  Min: {durations[0]:.2f}s | Q1: {durations[len(durations)//4]:.2f}s "
          f"| Median: {durations[len(durations)//2]:.2f}s "
          f"| Q3: {durations[3*len(durations)//4]:.2f}s | Max: {durations[-1]:.2f}s")
    print(f"  Mean: {np.mean(durations):.2f}s | Std: {np.std(durations):.2f}s "
          f"| Total: {sum(durations):.1f}s")
    
    # Histogram
    bins = [0, 2, 4, 6, 8, 10, 15, 20, 999]
    labels = ['0-2s', '2-4s', '4-6s', '6-8s', '8-10s', '10-15s', '15-20s', '20s+']
    hist = [0] * len(labels)
    for d in durations:
        for j in range(len(bins) - 1):
            if bins[j] <= d < bins[j+1]:
                hist[j] += 1
                break
    print(f"  Histogram:")
    max_bar = max(hist) if max(hist) > 0 else 1
    for label, count in zip(labels, hist):
        bar = '█' * int(count / max_bar * 20) if count > 0 else ''
        print(f"    {label:>7}: {bar} {count}")
    
    # Warmup
    print(f"\n{'─'*60}")
    print(f"Warmup ({args.warmup_runs} run)...")
    for _ in range(args.warmup_runs):
        _ = run_loop_baseline(prepared[:2], model, vocoder, device,
                              steps=args.steps, cfg=args.cfg, seed=args.seed)
    torch.cuda.synchronize(device) if 'cuda' in device else None
    print("  ✓ Warmup complete")
    
    # ── Baseline: single-sample loop ──
    print(f"\n{'─'*60}")
    print(f"Running BASELINE (for-loop, batch_size=1)...")
    baseline_wavs, base_time, base_audio, base_rtf = run_loop_baseline(
        prepared, model, vocoder, device,
        steps=args.steps, cfg=args.cfg, seed=args.seed
    )
    torch.cuda.synchronize(device) if 'cuda' in device else None
    
    print(f"  Time: {base_time:.2f}s | Audio: {base_audio:.1f}s | RTF: {base_rtf:.4f}")
    
    # ── Batch inference tests ──
    results_table = [
        {"batch_size": 1, "time": base_time, "audio_sec": base_audio,
         "rtf": base_rtf, "speedup": 1.0}
    ]
    
    mse_results = []
    
    for bs in args.batch_sizes:
        if bs <= 1:
            continue
        
        print(f"\n{'─'*60}")
        print(f"Running BATCH (batch_size={bs})...")
        
        batch_wavs, batch_time, batch_audio, batch_rtf = run_batch(
            prepared, model, vocoder, device,
            batch_size=bs, steps=args.steps, cfg=args.cfg, seed=args.seed
        )
        torch.cuda.synchronize(device) if 'cuda' in device else None
        
        speedup = base_time / batch_time if batch_time > 0 else 0
        print(f"  Time: {batch_time:.2f}s | Audio: {batch_audio:.1f}s | "
              f"RTF: {batch_rtf:.4f} | Speedup: {speedup:.2f}x")
        
        results_table.append({
            "batch_size": bs,
            "time": batch_time,
            "audio_sec": batch_audio,
            "rtf": batch_rtf,
            "speedup": speedup,
        })
        
        # MSE comparison
        mses = compute_mse(baseline_wavs, batch_wavs)
        mean_mse = np.mean(mses) if mses else 0
        max_mse = np.max(mses) if mses else 0
        mse_results.append({
            "batch_size": bs,
            "mean_mse": mean_mse,
            "max_mse": max_mse,
            "per_sample_mse": mses,
        })
        print(f"  MSE vs baseline: mean={mean_mse:.6f}, max={max_mse:.6f}")
        
        # Save audio if requested
        if args.save_audio:
            out_dir = os.path.join(args.output_dir, f"batch_{bs}")
            os.makedirs(out_dir, exist_ok=True)
            for j, wav in enumerate(batch_wavs):
                sf.write(os.path.join(out_dir, f"sample_{j:04d}.wav"), wav, SAMPLE_RATE)
    
    if args.save_audio:
        base_dir = os.path.join(args.output_dir, "baseline")
        os.makedirs(base_dir, exist_ok=True)
        for j, wav in enumerate(baseline_wavs):
            sf.write(os.path.join(base_dir, f"sample_{j:04d}.wav"), wav, SAMPLE_RATE)
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Batch Size':>12} | {'Time (s)':>10} | {'RTF':>8} | {'Speedup':>10}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
    for r in results_table:
        print(f"{r['batch_size']:>12} | {r['time']:>10.2f} | {r['rtf']:>8.4f} | {r['speedup']:>9.2f}x")
    
    if mse_results:
        print(f"\n{'Batch Size':>12} | {'Mean MSE':>12} | {'Max MSE':>12} | {'Status':>10}")
        print(f"{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
        for m in mse_results:
            status = "✅ PASS" if m['max_mse'] < 0.1 else "⚠️ CHECK"
            print(f"{m['batch_size']:>12} | {m['mean_mse']:>12.6f} | {m['max_mse']:>12.6f} | {status:>10}")
    
    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'num_samples': num_samples,
                'steps': args.steps,
                'cfg': args.cfg,
                'device': device,
            },
            'rtf_results': results_table,
            'mse_results': [{k: v if not isinstance(v, np.floating) else float(v)
                            for k, v in m.items() if k != 'per_sample_mse'}
                           for m in mse_results],
        }, f, indent=2, default=str)
    print(f"\n📊 Detailed results saved to {results_path}")


if __name__ == "__main__":
    main()
