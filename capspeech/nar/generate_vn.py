"""
Vietnamese inference for CapSpeech NAR model.

Changes from original generate.py:
1. Uses ViT5-large instead of Flan-T5-large for caption encoding
2. Uses character-level tokenizer instead of English G2P
3. All CLAP tags default to "none"
"""

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from capspeech.nar import bigvgan
import librosa
from capspeech.nar.utils import make_pad_mask
from capspeech.nar.model.modules import MelSpec
from capspeech.nar.network.crossdit import CrossDiT
from capspeech.nar.inference import sample
from capspeech.nar.utils import load_yaml_with_includes
import soundfile as sf
from transformers import T5EncoderModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
import laion_clap
import re
import time


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def text_to_chars(text):
    """Convert Vietnamese text to character-level tokens."""
    chars = []
    for ch in text.lower():
        if ch == ' ':
            chars.append('<BLK>')
        else:
            chars.append(ch)
    return chars


def encode(text, phn2num):
    """Encode Vietnamese text to token IDs using character-level vocab."""
    chars = text_to_chars(text)
    # Filter only tokens that exist in vocab
    chars = [ch for ch in chars if ch in phn2num]
    return chars


def estimate_duration_range(text):
    words = text.strip().split()
    num_words = len(words)
    min_duration = num_words / 4.0
    max_duration = num_words / 1.5
    ref_min = num_words / 3.0
    ref_max = num_words / 1.5
    return min_duration, max_duration, ref_min, ref_max


def get_duration(text, predicted_duration):
    min_dur, max_dur, ref_min, ref_max = estimate_duration_range(text)
    if predicted_duration < min_dur or predicted_duration > max_dur:
        return round(random.uniform(ref_min, ref_max), 2)
    return predicted_duration


def run(
        model_list,
        device,
        duration,
        transcript,
        caption,
        speed=1.0,
        steps=25,
        cfg=2.0
    ):
    model, vocoder, phn2num, clap_model, duration_tokenizer, duration_model, caption_tokenizer, caption_encoder = model_list
    print("Start Generation...")
    start_time = time.time()

    # Vietnamese: always "none" tag (no sound events)
    tag = "none"

    chars = encode(transcript, phn2num)
    text_tokens = [phn2num[item] for item in chars]
    text = torch.LongTensor(text_tokens).unsqueeze(0).to(device)

    if duration is None:
        duration_inputs = caption + " <NEW_SEP> " + transcript
        duration_inputs = duration_tokenizer(
            duration_inputs, return_tensors="pt",
            padding="max_length", truncation=True, max_length=400
        )

    with torch.no_grad():
        batch_encoding = caption_tokenizer(caption, return_tensors="pt")
        ori_tokens = batch_encoding["input_ids"].to(device)
        prompt = caption_encoder(input_ids=ori_tokens).last_hidden_state.squeeze().unsqueeze(0).to(device)
        
        tag_data = [tag]
        tag_embed = clap_model.get_text_embedding(tag_data, use_tensor=True)
        clap = tag_embed.squeeze().unsqueeze(0).to(device)

        if duration is None:
            duration_outputs = duration_model(**duration_inputs)
            predicted_duration = duration_outputs.logits.squeeze().item()
            duration = get_duration(transcript, predicted_duration)

    if speed == 0:
        speed = 1
    duration = duration / speed
    audio_clips = torch.zeros([1, math.ceil(duration * 24000 / 256), 100]).to(device)
    cond = None
    seq_len_prompt = prompt.shape[1]
    prompt_lens = torch.Tensor([prompt.shape[1]])
    prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)
    gen = sample(model, vocoder,
                 audio_clips, cond, text, prompt, clap, prompt_mask,
                 steps=steps, cfg=cfg,
                 sway_sampling_coef=-1.0, device=device)

    end_time = time.time()
    audio_len = gen.shape[-1] / 24000
    rtf = (end_time - start_time) / audio_len
    print(f"RTF: {rtf:.4f}")
    return gen


def load_model(device, model_path, config_path, vocab_path,
               caption_model_name="VietAI/vit5-large"):
    """Load Vietnamese CapSpeech NAR model.
    
    Args:
        device: torch device
        model_path: path to .pt checkpoint
        config_path: path to YAML config
        vocab_path: path to Vietnamese vocab.txt
        caption_model_name: HuggingFace model name for caption encoder
    """
    print("Loading models...")
    params = load_yaml_with_includes(config_path)
    model = CrossDiT(**params['model']).to(device)
    checkpoint = torch.load(model_path, map_location=device)['model']
    model.load_state_dict(checkpoint, strict=True)

    # mel spectrogram
    mel = MelSpec(**params['mel']).to(device)
    latent_sr = params['mel']['target_sample_rate'] / params['mel']['hop_length']

    # load Vietnamese character vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        temp = [l.strip().split(" ", 1) for l in f.readlines() if len(l.strip()) > 0]
        phn2num = {item[1]: int(item[0]) for item in temp}

    # load vocoder
    vocoder = bigvgan.BigVGAN.from_pretrained(
        'nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    # load ViT5 caption encoder (Vietnamese)
    print(f"Loading caption encoder: {caption_model_name}")
    caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_name)
    caption_encoder = T5EncoderModel.from_pretrained(caption_model_name).to(device).eval()

    # load CLAP
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    # Note: CLAP checkpoint needed for generating "none" embedding
    # If not available, the model will use default initialization

    # load duration predictor (optional — can also use fixed duration)
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id="OpenSound/CapSpeech-models")
        duration_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(local_dir, "nar_duration_predictor"))
        duration_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(local_dir, "nar_duration_predictor"))
        duration_model.eval()
        clap_model.load_ckpt(os.path.join(local_dir, "clap-630k-best.pt"))
    except Exception as e:
        print(f"Warning: Could not load duration predictor: {e}")
        print("You must specify --duration manually")
        duration_tokenizer = None
        duration_model = None

    model_list = [
        model, vocoder, phn2num, clap_model,
        duration_tokenizer, duration_model,
        caption_tokenizer, caption_encoder
    ]

    return model_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument('--config_path', type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument('--vocab_path', type=str, required=True,
                        help="Path to Vietnamese vocab.txt")
    parser.add_argument('--caption_model', type=str, default='VietAI/vit5-large',
                        help="Caption encoder model name")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--random", action="store_true")
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--transcript', type=str, required=True)
    parser.add_argument('--caption', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not args.random:
        seed_everything(args.seed)

    model_list = load_model(
        device, args.model_path, args.config_path,
        args.vocab_path, args.caption_model
    )
    audio_arr = run(model_list, device, args.duration,
                    args.transcript, args.caption)
    sf.write(args.output_path, audio_arr, 24000)
    print(f"Saved to: {args.output_path}")
