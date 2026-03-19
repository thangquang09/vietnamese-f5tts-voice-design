#!/usr/bin/env python3
"""
Batch generation script for CapSpeech NAR Vietnamese model.

Reads multiple (text, caption) pairs from a JSON file and generates
audio in parallel batches using true GPU batch inference.

Usage:
    python batch_generate_vn.py \
        --input_json samples.json \
        --output_dir output_batch/ \
        --batch_size 8 \
        --steps 25 --cfg 2.0

Input JSON format:
    [
        {"text": "Xin chào", "caption": "Giọng nữ trẻ", "duration": 2.5},
        {"text": "Tạm biệt", "caption": "Giọng nam", "duration": null}
    ]

This script does NOT modify any existing code. It imports from existing
modules and the new batch_inference.py.
"""

import os
import sys
import json
import math
import time
import random
import base64
import argparse
from typing import Optional, List, Dict

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from capspeech.nar import bigvgan
from capspeech.nar.network.crossdit import CrossDiT
from capspeech.nar.inference import sample  # original single-sample
from capspeech.nar.batch_inference import batch_sample
from capspeech.nar.utils import load_yaml_with_includes, make_pad_mask


# ──────────────────────── Defaults ──────────────────────── #

HF_MODEL_REPO = "thangquang09/capspeech-nar-vietnamese"
BIGVGAN_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"
CAPTION_MODEL = "VietAI/vit5-large"
SAMPLE_RATE = 24000
HOP_LENGTH = 256

# Pre-computed CLAP "none" embedding (base64)
_CLAP_NONE_B64 = "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"


# ──────────────────────── Helper Functions ──────────────────────── #

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def text_to_chars(text: str) -> list[str]:
    chars = []
    for ch in text.lower():
        if ch == " ":
            chars.append("<BLK>")
        else:
            chars.append(ch)
    return chars


def estimate_duration(text: str) -> float:
    words = text.strip().split()
    num_words = len(words)
    ref_min = num_words / 4.0
    ref_max = num_words / 2.5
    return max(round(random.uniform(ref_min, ref_max), 2), 1.0)


def get_duration(text: str, predicted_duration: float) -> float:
    words = text.strip().split()
    num_words = len(words)
    min_dur = num_words / 4.0
    max_dur = num_words / 1.5
    if predicted_duration < min_dur or predicted_duration > max_dur:
        return estimate_duration(text)
    return predicted_duration


# ──────────────────────── Model Loading ──────────────────────── #

def load_models(device: str, hf_repo: str = HF_MODEL_REPO,
                caption_model: str = CAPTION_MODEL,
                use_fp16: bool = False,
                use_compile: bool = False):
    """Load all models needed for batch inference."""
    print("=" * 60)
    print("  CapSpeech NAR — Batch Inference Loader")
    print("=" * 60)

    target_dtype = torch.float16 if use_fp16 else torch.float32

    # 1. Download from HuggingFace
    print(f"\n[1/5] Downloading model from {hf_repo}...")
    ckpt_path = hf_hub_download(repo_id=hf_repo, filename="checkpoint.pt")
    config_path = hf_hub_download(repo_id=hf_repo, filename="finetune_vn.yaml")
    vocab_path = hf_hub_download(repo_id=hf_repo, filename="vocab.txt")

    # 2. Load CrossDiT
    print("[2/5] Loading CrossDiT model...")
    params = load_yaml_with_includes(config_path)
    model = CrossDiT(**params['model'])
    checkpoint = torch.load(ckpt_path, map_location=device)['model']
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device).eval()

    if use_fp16:
        model = model.half()
        print("      → Applied fp16")

    if use_compile:
        print("      → Applying torch.compile (this may take a minute)...")
        model = torch.compile(model)

    print(f"      ✓ CrossDiT loaded")

    # 3. Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        lines = [l.strip().split(" ", 1) for l in f if l.strip()]
        phn2num = {item[1]: int(item[0]) for item in lines}
    print(f"      ✓ Vocab loaded ({len(phn2num)} characters)")

    # 4. Load BigVGAN
    print("[3/5] Loading BigVGAN vocoder...")
    try:
        vocoder = bigvgan.BigVGAN.from_pretrained(
            BIGVGAN_REPO, use_cuda_kernel=False)
    except TypeError:
        vocoder_dir = snapshot_download(repo_id=BIGVGAN_REPO)
        h = bigvgan.load_hparams_from_json(os.path.join(vocoder_dir, "config.json"))
        vocoder = bigvgan.BigVGAN(h, use_cuda_kernel=False)
        ckpt = torch.load(os.path.join(vocoder_dir, "bigvgan_generator.pt"), map_location="cpu")
        vocoder.load_state_dict(ckpt["generator"])
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    if use_fp16:
        vocoder = vocoder.half()
    print("      ✓ BigVGAN loaded")

    # 5. Load ViT5 caption encoder
    print(f"[4/5] Loading caption encoder ({caption_model})...")
    caption_tokenizer = T5Tokenizer.from_pretrained(caption_model)
    caption_encoder = T5EncoderModel.from_pretrained(caption_model).to(device).eval()
    if use_fp16:
        caption_encoder = caption_encoder.half()
    print("      ✓ ViT5-large loaded")

    # 6. Load CLAP "none" embedding
    print("[5/5] Loading CLAP embedding...")
    clap_bytes = base64.b64decode(_CLAP_NONE_B64)
    clap_np = np.frombuffer(clap_bytes, dtype=np.float32)
    clap_none = torch.from_numpy(clap_np.copy()).to(device)
    if use_fp16:
        clap_none = clap_none.half()
    print(f"      ✓ CLAP 'none' embedding loaded")

    # 7. Load duration predictor (optional)
    duration_predictor_path = os.path.join(
        os.path.dirname(__file__), "capspeech", "nar", "phobert_duration_predictor"
    )
    duration_model = None
    duration_tokenizer = None
    
    if os.path.isdir(duration_predictor_path):
        try:
            duration_tokenizer = AutoTokenizer.from_pretrained(duration_predictor_path)
            duration_model = AutoModelForSequenceClassification.from_pretrained(
                duration_predictor_path, num_labels=1
            ).to(device).eval()
            print("      ✓ Duration predictor loaded")
        except Exception as e:
            print(f"      ⚠️ Duration predictor failed: {e}")
    else:
        # Try HuggingFace
        try:
            dp_dir = snapshot_download(
                repo_id=hf_repo, allow_patterns="duration_predictor/*")
            dp_subdir = os.path.join(dp_dir, "duration_predictor")
            if os.path.isdir(dp_subdir):
                duration_tokenizer = AutoTokenizer.from_pretrained(dp_subdir)
                duration_model = AutoModelForSequenceClassification.from_pretrained(
                    dp_subdir, num_labels=1
                ).to(device).eval()
                print("      ✓ Duration predictor loaded from HF")
        except Exception:
            print("      ⚠️ Duration predictor not available, using heuristic")

    print("\n✅ All models loaded!\n")

    return {
        'model': model,
        'vocoder': vocoder,
        'phn2num': phn2num,
        'caption_tokenizer': caption_tokenizer,
        'caption_encoder': caption_encoder,
        'clap_none': clap_none,
        'duration_model': duration_model,
        'duration_tokenizer': duration_tokenizer,
        'params': params,
        'dtype': target_dtype,
    }


# ──────────────────────── Batch Encoding ──────────────────────── #

def encode_batch(
    items: List[Dict],
    models: dict,
    device: str,
) -> tuple:
    """Encode a list of items into tensors ready for batch_sample().
    
    Each item: {"text": str, "caption": str, "duration": float|None}
    
    Returns:
        (x_list, text_list, prompt_list, clap_list,
         prompt_lens_list, duration_frames_list, durations_sec)
    """
    phn2num = models['phn2num']
    caption_tokenizer = models['caption_tokenizer']
    caption_encoder = models['caption_encoder']
    clap_none = models['clap_none']
    duration_model = models['duration_model']
    duration_tokenizer = models['duration_tokenizer']
    target_dtype = models['dtype']
    
    x_list = []
    text_list = []
    prompt_list = []
    clap_list = []
    prompt_lens_list = []
    duration_frames_list = []
    durations_sec = []
    
    # ── Batch encode captions with ViT5 ──
    # Detect unique captions for caching
    captions = [item['caption'] for item in items]
    unique_captions = list(set(captions))
    
    with torch.no_grad():
        # Tokenize all unique captions at once
        batch_enc = caption_tokenizer(
            unique_captions, return_tensors="pt",
            padding=True, truncation=True
        )
        input_ids = batch_enc["input_ids"].to(device)
        attention_mask = batch_enc["attention_mask"].to(device)
        
        # Encode all unique captions in one forward pass
        outputs = caption_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        caption_embeds = outputs.last_hidden_state  # [n_unique, max_tokens, t5_dim]
        
        # Build cache: caption_text -> (embedding, actual_length)
        caption_cache = {}
        for i, cap in enumerate(unique_captions):
            actual_len = attention_mask[i].sum().item()
            emb = caption_embeds[i, :int(actual_len), :]  # trim padding
            caption_cache[cap] = (emb, int(actual_len))
    
    # ── Process each item ──
    for item in items:
        text = item['text']
        caption = item['caption']
        duration = item.get('duration', None)
        speed = item.get('speed', 1.0)
        
        # Text encoding
        chars = text_to_chars(text)
        chars = [ch for ch in chars if ch in phn2num]
        text_tokens = [phn2num[ch] for ch in chars]
        text_tensor = torch.LongTensor(text_tokens).to(device)
        text_list.append(text_tensor)
        
        # Caption embedding (from cache)
        prompt_emb, prompt_len = caption_cache[caption]
        if target_dtype == torch.float16:
            prompt_emb = prompt_emb.half()
        prompt_list.append(prompt_emb)
        prompt_lens_list.append(prompt_len)
        
        # CLAP
        clap_list.append(clap_none.clone())
        
        # Duration
        if duration is None:
            if duration_model is not None and duration_tokenizer is not None:
                with torch.no_grad():
                    combined = f"{caption} [SEP] {text}"
                    inputs = duration_tokenizer(
                        combined, return_tensors="pt",
                        padding="max_length", truncation=True, max_length=256
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    output = duration_model(**inputs)
                    duration = max(output.logits.squeeze().item(), 0.5)
                    duration = min(duration, 30.0)
            else:
                duration = estimate_duration(text)
        
        if speed and speed > 0:
            duration = duration / speed
        
        n_frames = math.ceil(duration * SAMPLE_RATE / HOP_LENGTH)
        duration_frames_list.append(n_frames)
        durations_sec.append(duration)
        
        # Mel target (zeros placeholder)
        x = torch.zeros(n_frames, 100, device=device)
        if target_dtype == torch.float16:
            x = x.half()
        x_list.append(x)
    
    return (x_list, text_list, prompt_list, clap_list,
            prompt_lens_list, duration_frames_list, durations_sec)


# ──────────────────────── Main Generation ──────────────────────── #

def batch_generate(
    items: List[Dict],
    models: dict,
    device: str,
    batch_size: int = 8,
    steps: int = 25,
    cfg: float = 2.0,
    seed: int = None,
) -> List[np.ndarray]:
    """Generate audio for multiple items using true batch inference.
    
    Args:
        items: List of dicts with keys: text, caption, duration (optional)
        models: Model dict from load_models()
        device: CUDA device string
        batch_size: Number of items per batch
        steps: ODE solver steps
        cfg: Classifier-free guidance
        seed: Random seed
    
    Returns:
        List of numpy arrays (one per item)
    """
    all_results = []
    total_items = len(items)
    target_dtype = models['dtype']
    
    # Sort by estimated duration for more efficient padding
    indexed_items = [(i, item) for i, item in enumerate(items)]
    # Pre-estimate frame counts for sorting
    def est_frames(item):
        dur = item.get('duration')
        if dur is None:
            dur = estimate_duration(item['text'])
        return math.ceil(dur * SAMPLE_RATE / HOP_LENGTH)
    
    indexed_items.sort(key=lambda x: est_frames(x[1]))
    
    # Result array indexed by original position
    results_ordered = [None] * total_items
    
    # Process in batches
    num_batches = math.ceil(total_items / batch_size)
    overall_start = time.time()
    
    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, total_items)
        batch_items = indexed_items[start_i:end_i]
        
        orig_indices = [idx for idx, _ in batch_items]
        batch_data = [item for _, item in batch_items]
        
        print(f"\n{'─'*50}")
        print(f"Batch {batch_idx + 1}/{num_batches} ({len(batch_data)} samples)")
        
        # Encode inputs
        (x_list, text_list, prompt_list, clap_list,
         prompt_lens_list, duration_frames_list, durations_sec) = encode_batch(
            batch_data, models, device
        )
        
        for j, (item, dur) in enumerate(zip(batch_data, durations_sec)):
            print(f"  [{orig_indices[j]}] dur={dur:.2f}s | {item['text'][:50]}...")
        
        # Run batch inference
        batch_start = time.time()
        wavs = batch_sample(
            model=models['model'],
            vocoder=models['vocoder'],
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
            dtype=target_dtype if target_dtype != torch.float32 else None,
        )
        batch_elapsed = time.time() - batch_start
        
        total_audio = sum(len(w) / SAMPLE_RATE for w in wavs)
        rtf = batch_elapsed / total_audio if total_audio > 0 else 0
        print(f"  → {total_audio:.1f}s audio in {batch_elapsed:.2f}s (RTF={rtf:.3f})")
        
        # Store results in original order
        for j, wav in enumerate(wavs):
            results_ordered[orig_indices[j]] = wav
    
    overall_elapsed = time.time() - overall_start
    total_audio = sum(len(w) / SAMPLE_RATE for w in results_ordered if w is not None)
    overall_rtf = overall_elapsed / total_audio if total_audio > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Total: {total_audio:.1f}s audio in {overall_elapsed:.2f}s (RTF={overall_rtf:.3f})")
    print(f"{'='*60}")
    
    return results_ordered


# ──────────────────────── CLI ──────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="CapSpeech NAR — Batch Vietnamese TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_generate_vn.py \\
      --input_json samples.json \\
      --output_dir output_batch/ \\
      --batch_size 8

  # With fp16 and torch.compile
  python batch_generate_vn.py \\
      --input_json samples.json \\
      --output_dir output_batch/ \\
      --batch_size 4 --fp16 --compile
        """,
    )
    parser.add_argument("--input_json", type=str, required=True,
                        help="JSON file with list of {text, caption, duration}")
    parser.add_argument("--output_dir", type=str, default="output_batch",
                        help="Output directory for WAV files")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--steps", type=int, default=25,
                        help="ODE solver steps")
    parser.add_argument("--cfg", type=float, default=2.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 for inference")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile to CrossDiT")
    parser.add_argument("--hf_repo", type=str, default=HF_MODEL_REPO,
                        help="HuggingFace model repo")
    args = parser.parse_args()

    # Device
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    # Seed
    if args.seed is not None:
        seed_everything(args.seed)

    # Load input
    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from {args.input_json}")

    # Load models
    models = load_models(
        device=args.device,
        hf_repo=args.hf_repo,
        use_fp16=args.fp16,
        use_compile=args.compile,
    )

    # Generate
    wavs = batch_generate(
        items=items,
        models=models,
        device=args.device,
        batch_size=args.batch_size,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    for i, wav in enumerate(wavs):
        if wav is not None:
            out_path = os.path.join(args.output_dir, f"sample_{i:04d}.wav")
            sf.write(out_path, wav, SAMPLE_RATE)
    
    print(f"\n💾 Saved {len(wavs)} files to {args.output_dir}/")


if __name__ == "__main__":
    main()
