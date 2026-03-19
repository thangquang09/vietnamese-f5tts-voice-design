#!/usr/bin/env python3
"""
Batch inference for CapSpeech NAR Vietnamese model.

Provides `batch_sample()` — a true GPU-parallelized batch inference function
that processes multiple samples simultaneously via matrix multiplication,
using padding + attention masking + post-processing slicing.

This file does NOT modify any existing code. The original `sample()` in
`inference.py` remains untouched.
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from einops import rearrange


def lens_to_mask(lengths: torch.Tensor, length: int = None) -> torch.Tensor:
    """Create boolean mask [b, n] from lengths tensor [b].
    
    mask[i, j] = True if j < lengths[i]
    """
    if length is None:
        length = lengths.max().item()
    seq = torch.arange(length, device=lengths.device)
    return seq.unsqueeze(0) < lengths.unsqueeze(1)  # [b, n]


@torch.no_grad()
def batch_sample(
    model,
    vocoder,
    x_list: list[torch.Tensor],       # list of [1, n_frames_i, mel_dim] or [n_frames_i, mel_dim]
    cond,                              # None (no ref audio conditioning)
    text_list: list[torch.Tensor],     # list of [1, nt_i] or [nt_i]
    prompt_list: list[torch.Tensor],   # list of [1, n_prompt_i, t5_dim] or [n_prompt_i, t5_dim]
    clap_list: list[torch.Tensor],     # list of [1, clap_dim] or [clap_dim]
    prompt_lens_list: list[int],       # list of prompt lengths
    duration_frames_list: list[int],   # actual frame count for each sample
    steps: int = 25,
    cfg: float = 2.0,
    sway_sampling_coef: float = -1.0,
    device: str = 'cuda',
    seed: int = None,
    dtype: torch.dtype = None,         # None = model's own dtype; torch.float16 for half
):
    """True batch inference with padding, masking and post-processing slicing.
    
    Key differences from `sample()` in inference.py:
    1. Accepts lists of tensors (variable length per sample)
    2. Pads all tensors to max_duration within the batch
    3. Creates attention mask from actual durations → model ignores padding
    4. After ODE, slices each sample back to its original duration
    
    Args:
        model: CrossDiT model (eval mode)
        vocoder: BigVGAN vocoder (eval mode)
        x_list: List of mel-spectrogram tensors (one per sample)
        cond: Conditioning audio (None for unconditional)
        text_list: List of text token tensors
        prompt_list: List of ViT5 caption embedding tensors
        clap_list: List of CLAP embedding tensors
        prompt_lens_list: List of prompt sequence lengths
        duration_frames_list: Actual mel frame count per sample
        steps: ODE solver steps
        cfg: Classifier-free guidance strength
        sway_sampling_coef: Sway sampling coefficient
        device: Target device
        seed: Random seed (None = no seeding)
        dtype: Optional dtype override for half-precision inference
        
    Returns:
        List of numpy arrays, each containing the generated waveform for one sample.
    """
    model.eval()
    vocoder.eval()
    
    batch_size = len(x_list)
    mel_dim = x_list[0].shape[-1] if x_list[0].ndim >= 2 else 100
    
    # ── Parse duration info ──
    duration_tensor = torch.tensor(duration_frames_list, device=device, dtype=torch.long)
    max_duration = duration_tensor.max().item()
    
    # ── Pad mel targets (x) to max_duration ──
    x_padded = []
    for x_i in x_list:
        if x_i.ndim == 3:
            x_i = x_i.squeeze(0)  # [n_frames, mel_dim]
        n = x_i.shape[0]
        if n < max_duration:
            x_i = F.pad(x_i, (0, 0, 0, max_duration - n), value=0.0)
        x_padded.append(x_i)
    x_batch = torch.stack(x_padded, dim=0).to(device)  # [b, max_dur, mel_dim]
    
    # ── Pad text tokens ──
    text_tensors = []
    for t_i in text_list:
        if t_i.ndim == 2:
            t_i = t_i.squeeze(0)  # [nt]
        text_tensors.append(t_i)
    # pad_sequence with padding_value=-1 (filler token)
    text_batch = pad_sequence(text_tensors, batch_first=True, padding_value=-1).to(device)  # [b, max_nt]
    
    # ── Pad prompt (ViT5 caption embeddings) ──
    prompt_tensors = []
    for p_i in prompt_list:
        if p_i.ndim == 3:
            p_i = p_i.squeeze(0)  # [n_prompt, t5_dim]
        prompt_tensors.append(p_i)
    prompt_batch = pad_sequence(prompt_tensors, batch_first=True, padding_value=0.0).to(device)  # [b, max_prompt, t5_dim]
    
    # ── Stack CLAP embeddings ──
    clap_tensors = []
    for c_i in clap_list:
        if c_i.ndim == 2:
            c_i = c_i.squeeze(0)  # [clap_dim]
        clap_tensors.append(c_i)
    clap_batch = torch.stack(clap_tensors, dim=0).to(device)  # [b, clap_dim]
    
    # ── Create prompt mask ──
    prompt_lens_tensor = torch.tensor(prompt_lens_list, device=device, dtype=torch.long)
    max_prompt_len = prompt_batch.shape[1]
    prompt_mask = lens_to_mask(prompt_lens_tensor, length=max_prompt_len)  # [b, max_prompt]
    
    # ── Create attention mask from actual durations ──
    # This mask tells the model which positions are real vs padding
    mask = lens_to_mask(duration_tensor, length=max_duration)  # [b, max_dur]
    
    # ── Optional: half-precision casting ──
    if dtype is not None:
        x_batch = x_batch.to(dtype)
        prompt_batch = prompt_batch.to(dtype)
        clap_batch = clap_batch.to(dtype)
    
    # ── Build negative (unconditional) inputs for CFG ──
    neg_text = torch.ones_like(text_batch) * -1
    neg_clap = torch.zeros_like(clap_batch)
    neg_prompt = torch.zeros_like(prompt_batch)
    neg_prompt_mask = torch.zeros_like(prompt_mask)
    neg_prompt_mask[:, 0] = True  # at least one position to avoid NaN
    
    # ── Generate noise ──
    # Per-sample noise generation for reproducibility (matches cfm.py logic)
    y0_list = []
    for dur in duration_frames_list:
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn(dur, mel_dim, device=device)
        if dtype is not None:
            noise = noise.to(dtype)
        y0_list.append(noise)
    y0 = pad_sequence(y0_list, batch_first=True, padding_value=0.0)  # [b, max_dur, mel_dim]
    
    # ── ODE solver with masking ──
    cond_input = torch.zeros_like(x_batch) if cond is None else cond

    def fn(t, x):
        # Conditional prediction (with real text, prompt, clap)
        pred = model(
            x=x, cond=cond_input, text=text_batch, time=t,
            prompt=prompt_batch, clap=clap_batch,
            mask=mask, prompt_mask=prompt_mask
        )
        
        if cfg < 1e-5:
            return pred
        
        # Unconditional prediction (for CFG)
        null_pred = model(
            x=x, cond=cond_input, text=neg_text, time=t,
            prompt=neg_prompt, clap=neg_clap,
            mask=mask, prompt_mask=neg_prompt_mask
        )
        
        return pred + (pred - null_pred) * cfg
    
    # Time steps with optional sway sampling
    t_start = 0
    t = torch.linspace(t_start, 1, steps, device=device)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
    
    # Solve ODE
    trajectory = odeint(fn, y0, t, method="euler")
    out = trajectory[-1]  # [b, max_dur, mel_dim]
    
    # ── Post-processing: slice off padding and vocode per-sample ──
    results = []
    for i, dur in enumerate(duration_frames_list):
        out_i = out[i, :dur, :]  # [dur, mel_dim] — remove padding
        out_i = rearrange(out_i, 'n d -> 1 d n')  # [1, mel_dim, dur]
        
        # Cast back to float32 for vocoder if needed
        if out_i.dtype != torch.float32:
            out_i = out_i.float()
        
        with torch.inference_mode():
            wav_i = vocoder(out_i)
        
        wav_np = wav_i.squeeze().float().cpu().numpy()
        results.append(wav_np)
    
    return results


@torch.no_grad()
def batch_sample_mel(
    model,
    x_list: list[torch.Tensor],
    cond,
    text_list: list[torch.Tensor],
    prompt_list: list[torch.Tensor],
    clap_list: list[torch.Tensor],
    prompt_lens_list: list[int],
    duration_frames_list: list[int],
    steps: int = 25,
    cfg: float = 2.0,
    sway_sampling_coef: float = -1.0,
    device: str = 'cuda',
    seed: int = None,
    dtype: torch.dtype = None,
):
    """Same as batch_sample but returns mel spectrograms instead of waveforms.
    
    Useful for benchmarking (MSE comparison) without vocoder overhead.
    
    Returns:
        List of mel tensors, each [dur_i, mel_dim] (padding removed).
    """
    model.eval()
    
    batch_size = len(x_list)
    mel_dim = x_list[0].shape[-1] if x_list[0].ndim >= 2 else 100
    
    duration_tensor = torch.tensor(duration_frames_list, device=device, dtype=torch.long)
    max_duration = duration_tensor.max().item()
    
    # Pad mel targets
    x_padded = []
    for x_i in x_list:
        if x_i.ndim == 3:
            x_i = x_i.squeeze(0)
        n = x_i.shape[0]
        if n < max_duration:
            x_i = F.pad(x_i, (0, 0, 0, max_duration - n), value=0.0)
        x_padded.append(x_i)
    x_batch = torch.stack(x_padded, dim=0).to(device)
    
    # Pad text
    text_tensors = []
    for t_i in text_list:
        if t_i.ndim == 2:
            t_i = t_i.squeeze(0)
        text_tensors.append(t_i)
    text_batch = pad_sequence(text_tensors, batch_first=True, padding_value=-1).to(device)
    
    # Pad prompts
    prompt_tensors = []
    for p_i in prompt_list:
        if p_i.ndim == 3:
            p_i = p_i.squeeze(0)
        prompt_tensors.append(p_i)
    prompt_batch = pad_sequence(prompt_tensors, batch_first=True, padding_value=0.0).to(device)
    
    # Stack CLAP
    clap_tensors = []
    for c_i in clap_list:
        if c_i.ndim == 2:
            c_i = c_i.squeeze(0)
        clap_tensors.append(c_i)
    clap_batch = torch.stack(clap_tensors, dim=0).to(device)
    
    # Masks
    prompt_lens_tensor = torch.tensor(prompt_lens_list, device=device, dtype=torch.long)
    prompt_mask = lens_to_mask(prompt_lens_tensor, length=prompt_batch.shape[1])
    mask = lens_to_mask(duration_tensor, length=max_duration)
    
    # Optional dtype
    if dtype is not None:
        x_batch = x_batch.to(dtype)
        prompt_batch = prompt_batch.to(dtype)
        clap_batch = clap_batch.to(dtype)
    
    # CFG negatives
    neg_text = torch.ones_like(text_batch) * -1
    neg_clap = torch.zeros_like(clap_batch)
    neg_prompt = torch.zeros_like(prompt_batch)
    neg_prompt_mask = torch.zeros_like(prompt_mask)
    neg_prompt_mask[:, 0] = True
    
    # Noise
    y0_list = []
    for dur in duration_frames_list:
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn(dur, mel_dim, device=device)
        if dtype is not None:
            noise = noise.to(dtype)
        y0_list.append(noise)
    y0 = pad_sequence(y0_list, batch_first=True, padding_value=0.0)
    
    cond_input = torch.zeros_like(x_batch) if cond is None else cond

    def fn(t, x):
        pred = model(
            x=x, cond=cond_input, text=text_batch, time=t,
            prompt=prompt_batch, clap=clap_batch,
            mask=mask, prompt_mask=prompt_mask
        )
        if cfg < 1e-5:
            return pred
        null_pred = model(
            x=x, cond=cond_input, text=neg_text, time=t,
            prompt=neg_prompt, clap=neg_clap,
            mask=mask, prompt_mask=neg_prompt_mask
        )
        return pred + (pred - null_pred) * cfg
    
    t = torch.linspace(0, 1, steps, device=device)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
    
    trajectory = odeint(fn, y0, t, method="euler")
    out = trajectory[-1]
    
    # Slice per-sample
    results = []
    for i, dur in enumerate(duration_frames_list):
        results.append(out[i, :dur, :].float().cpu())
    
    return results
