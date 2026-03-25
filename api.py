#!/usr/bin/env python3
"""
InstructVoice — Vietnamese Instruction TTS API

Automatically downloads the model from HuggingFace and provides
a simple API for Vietnamese text-to-speech synthesis with
instruction-guided voice control.

Usage (CLI):
    python api.py \
        --text "Xin chào, hôm nay bạn có khoẻ không?" \
        --caption "Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc." \
        --output output.wav

Usage (Python):
    from api import InstructVoiceAPI
    
    tts = InstructVoiceAPI()
    tts.synthesize(
        text="Xin chào, hôm nay bạn có khoẻ không?",
        caption="Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc.",
        output_path="output.wav"
    )

Requirements:
    pip install -r requirements.txt
"""

import os
import math
import time
import random
import base64
import argparse
from typing import Optional

import numpy as np
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from capspeech.nar import bigvgan
from capspeech.nar.network.crossdit import CrossDiT
from capspeech.nar.inference import sample
from capspeech.nar.utils import load_yaml_with_includes, make_pad_mask

# Optional: sea-g2p text normalizer for handling numbers, symbols, code-switch
try:
    from sea_g2p import Normalizer as SeaNormalizer
    _HAS_SEA_G2P = True
except ImportError:
    _HAS_SEA_G2P = False


# ──────────────────────── Default HuggingFace Repos ──────────────────────── #

HF_MODEL_REPO = "thangquang09/capspeech-nar-vietnamese-stage2"
BIGVGAN_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"
CAPTION_MODEL = "VietAI/vit5-large"
SAMPLE_RATE = 24000

# Default path for the duration predictor model (local or HF repo)
DURATION_PREDICTOR_PATH = os.path.join(
    os.path.dirname(__file__), "capspeech", "nar", "phobert_duration_predictor"
)

# Pre-computed CLAP text embedding for the "none" tag.
# Vietnamese model always uses "none" (no sound effects), so we embed it
# directly to avoid downloading the entire OpenSound/CapSpeech-models (~15GB).
_CLAP_NONE_B64 = "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"


# ──────────────────────── Helper Functions ──────────────────────── #

def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def text_to_chars(text: str) -> list[str]:
    """Convert Vietnamese text to character-level tokens."""
    chars = []
    for ch in text.lower():
        if ch == " ":
            chars.append("<BLK>")
        else:
            chars.append(ch)
    return chars


def estimate_duration(text: str) -> float:
    """Estimate speech duration (seconds) from text length.

    Natural Vietnamese speaking rate: ~3.5 words/second.
    We target a tight range around that to avoid overly long pauses.
    """
    words = text.strip().split()
    num_words = len(words)
    # ~3-4 words/sec for natural Vietnamese speech
    ref_min = num_words / 4.0
    ref_max = num_words / 2.5
    duration = round(random.uniform(ref_min, ref_max), 2)
    return max(duration, 1.0)  # at least 1 second


def get_duration(text: str, predicted_duration: float) -> float:
    """Clamp predicted duration to a reasonable range."""
    words = text.strip().split()
    num_words = len(words)
    min_dur = num_words / 4.0
    max_dur = num_words / 1.5
    if predicted_duration < min_dur or predicted_duration > max_dur:
        return estimate_duration(text)
    return predicted_duration


# ──────────────────────── Main API Class ──────────────────────── #

class InstructVoiceAPI:
    """Vietnamese Instruction TTS — end-to-end API.

    Based on CapSpeech NAR model, fine-tuned for Vietnamese.

    Automatically downloads all required models from HuggingFace Hub
    on first use and caches them locally.

    Args:
        device: torch device string, e.g. "cuda:0" or "cpu"
        hf_model_repo: HuggingFace repo ID for the CapSpeech checkpoint
        caption_model: HuggingFace model ID for the caption encoder
        cache_dir: directory to cache downloaded models (None = default HF cache)
        seed: random seed for reproducibility (None = random)
    """

    def __init__(
        self,
        device: str = None,
        hf_model_repo: str = HF_MODEL_REPO,
        caption_model: str = CAPTION_MODEL,
        duration_predictor_path: Optional[str] = DURATION_PREDICTOR_PATH,
        cache_dir: Optional[str] = None,
        seed: Optional[int] = 42,
        normalize: bool = True,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.sample_rate = SAMPLE_RATE

        if seed is not None:
            seed_everything(seed)

        # Initialize text normalizer (sea-g2p)
        self.normalizer = None
        if normalize and _HAS_SEA_G2P:
            try:
                self.normalizer = SeaNormalizer(lang="vi")
                print("✓ sea-g2p Normalizer loaded (Vietnamese)")
            except Exception as e:
                print(f"⚠️ Failed to init sea-g2p Normalizer: {e}")
        elif normalize and not _HAS_SEA_G2P:
            print("⚠️ sea-g2p not installed — text normalization disabled")
            print("   Install with: pip install sea-g2p")

        self._load_all(hf_model_repo, caption_model, duration_predictor_path, cache_dir)

    # ─────────────── Model Loading ─────────────── #

    def _load_all(self, hf_model_repo: str, caption_model: str,
                  duration_predictor_path: Optional[str], cache_dir: Optional[str]):
        """Download and load all model components."""
        print("============================================================")
        print("  InstructVoice — Vietnamese Instruction TTS")
        print("=" * 60)

        # 1. Download checkpoint, config, and vocab from HuggingFace
        print(f"\n[1/6] Downloading model from {hf_model_repo}...")
        ckpt_path = hf_hub_download(
            repo_id=hf_model_repo,
            filename="checkpoint.pt",
            cache_dir=cache_dir,
        )
        # Try stage2 config first, then fallback to stage1
        config_path = None
        for config_name in ["finetune_vn_stage2.yaml", "finetune_vn.yaml"]:
            try:
                config_path = hf_hub_download(
                    repo_id=hf_model_repo,
                    filename=config_name,
                    cache_dir=cache_dir,
                )
                break
            except Exception:
                continue
        if config_path is None:
            raise FileNotFoundError(
                f"No config file found in {hf_model_repo}. "
                "Expected finetune_vn_stage2.yaml or finetune_vn.yaml"
            )
        vocab_path = hf_hub_download(
            repo_id=hf_model_repo,
            filename="vocab.txt",
            cache_dir=cache_dir,
        )

        # 2. Load config and build model
        print("[2/6] Loading CrossDiT model...")
        self.params = load_yaml_with_includes(config_path)
        model = CrossDiT(**self.params["model"]).to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device)["model"]
        model.load_state_dict(checkpoint, strict=True)
        self.model = model.eval()
        print(f"      ✓ CrossDiT loaded (dim={self.params['model']['dim']}, "
              f"depth={self.params['model']['depth']})")

        # 3. Load Vietnamese character vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = [l.strip().split(" ", 1) for l in f if l.strip()]
            self.phn2num = {item[1]: int(item[0]) for item in lines}
        print(f"      ✓ Vocab loaded ({len(self.phn2num)} characters)")

        # 4. Load BigVGAN vocoder
        print("[3/6] Loading BigVGAN vocoder...")
        try:
            self.vocoder = bigvgan.BigVGAN.from_pretrained(
                BIGVGAN_REPO, use_cuda_kernel=False
            )
        except TypeError:
            # Workaround: bigvgan _from_pretrained requires 'proxies' and
            # 'resume_download' kwargs removed in newer huggingface_hub.
            # Replicate the _from_pretrained logic manually.
            vocoder_dir = snapshot_download(repo_id=BIGVGAN_REPO)
            h = bigvgan.load_hparams_from_json(
                os.path.join(vocoder_dir, "config.json")
            )
            self.vocoder = bigvgan.BigVGAN(h, use_cuda_kernel=False)
            ckpt = torch.load(
                os.path.join(vocoder_dir, "bigvgan_generator.pt"),
                map_location="cpu",
            )
            self.vocoder.load_state_dict(ckpt["generator"])
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.device)
        print("      ✓ BigVGAN v2 24kHz loaded")

        # 5. Load ViT5-large caption encoder
        print(f"[4/6] Loading caption encoder ({caption_model})...")
        self.caption_tokenizer = T5Tokenizer.from_pretrained(caption_model)
        self.caption_encoder = T5EncoderModel.from_pretrained(
            caption_model
        ).to(self.device).eval()
        print("      ✓ ViT5-large loaded")

        # 5. Load pre-computed CLAP "none" embedding
        # Vietnamese model always uses tag="none" (no sound effects).
        # We embed this directly to avoid downloading OpenSound/CapSpeech-models (~15GB).
        print("[5/6] Loading CLAP embedding...")
        clap_bytes = base64.b64decode(_CLAP_NONE_B64)
        clap_np = np.frombuffer(clap_bytes, dtype=np.float32)
        self.clap_none = torch.from_numpy(clap_np.copy()).unsqueeze(0).to(self.device)
        print(f"      ✓ CLAP 'none' embedding loaded (shape={self.clap_none.shape})")

        # 6. Load PhoBERT duration predictor
        print(f"[6/6] Loading duration predictor...")
        self.duration_model = None
        self.duration_tokenizer = None

        # Try loading: (1) local path → (2) HuggingFace → (3) heuristic fallback
        dp_load_path = None
        if duration_predictor_path and os.path.isdir(duration_predictor_path):
            dp_load_path = duration_predictor_path
            print(f"      Found local: {dp_load_path}")
        else:
            try:
                dp_dir = snapshot_download(
                    repo_id=hf_model_repo,
                    allow_patterns="duration_predictor/*",
                    cache_dir=cache_dir,
                )
                dp_subdir = os.path.join(dp_dir, "duration_predictor")
                if os.path.isdir(dp_subdir):
                    dp_load_path = dp_subdir
                    print(f"      Downloaded from HF: {hf_model_repo}/duration_predictor/")
            except Exception as e:
                print(f"      ⚠️ Could not download duration predictor from HF: {e}")

        if dp_load_path:
            try:
                self.duration_tokenizer = AutoTokenizer.from_pretrained(dp_load_path)
                self.duration_model = AutoModelForSequenceClassification.from_pretrained(
                    dp_load_path, num_labels=1
                ).to(self.device).eval()
                print(f"      ✓ PhoBERT duration predictor loaded")
            except Exception as e:
                print(f"      ⚠️ Failed to load duration predictor: {e}")
                print(f"      → Falling back to heuristic duration estimation")
                self.duration_model = None
                self.duration_tokenizer = None
        else:
            print(f"      ⚠️ Duration predictor not available")
            print(f"      → Using heuristic duration estimation")

        print("\n✅ All models loaded! Ready to synthesize.\n")

    # ─────────────── Duration Prediction ─────────────── #

    def predict_duration(self, text: str, caption: str) -> float:
        """Predict speech duration using PhoBERT model, with heuristic fallback."""
        if self.duration_model is not None and self.duration_tokenizer is not None:
            combined = f"{caption} [SEP] {text}"
            with torch.no_grad():
                inputs = self.duration_tokenizer(
                    combined, return_tensors="pt",
                    padding="max_length", truncation=True, max_length=256,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = self.duration_model(**inputs)
                predicted = output.logits.squeeze().item()

            # Clamp to reasonable range
            predicted = max(predicted, 0.5)
            predicted = min(predicted, 30.0)
            return predicted
        else:
            dur = estimate_duration(text)
            print(f"      ⚠️ Fallback: heuristic duration estimation → {dur:.2f}s")
            return dur

    # ─────────────── Text Normalization ─────────────── #

    def normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text (numbers, symbols, code-switch) using sea-g2p.

        Examples:
            "Giá SP500 hôm nay là 4.200,5 điểm"
            → "giá ét pê năm trăm hôm nay là bốn nghìn hai trăm phẩy năm điểm"
        """
        if self.normalizer is None:
            return text
        try:
            result = self.normalizer.normalize(text)
            # sea-g2p wraps English words in <en>...</en> tags — strip them
            import re
            result = re.sub(r'</?en>', '', result)
            return result
        except Exception as e:
            print(f"⚠️ Normalization failed, using raw text: {e}")
            return text

    # ─────────────── Core Synthesis ─────────────── #

    def synthesize(
        self,
        text: str,
        caption: str,
        output_path: str = "output.wav",
        duration: Optional[float] = None,
        speed: float = 1.0,
        steps: int = 25,
        cfg: float = 2.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Vietnamese speech from text with instruction-guided voice control.

        Args:
            text: Vietnamese text to synthesize.
            caption: Voice description / instruction in Vietnamese.
                     Example: "Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc."
            output_path: Path to save the output WAV file.
            duration: Audio duration in seconds (None = auto-estimate).
            speed: Speed factor (1.0 = normal, > 1.0 = faster, < 1.0 = slower).
            steps: ODE solver steps (more = better quality, slower). Default 25.
            cfg: Classifier-free guidance scale. Default 2.0.
            seed: Random seed for this generation (None = keep current state).

        Returns:
            numpy array of audio samples (float32, 24kHz).
        """
        if seed is not None:
            seed_everything(seed)

        # Normalize text (numbers, symbols, code-switch)
        original_text = text
        text = self.normalize_text(text)

        print(f"📝 Text:    {original_text}")
        if text != original_text:
            print(f"📝 Normalized: {text}")
        print(f"🎙️  Caption: {caption}")
        start_time = time.time()

        # Encode text to character tokens
        chars = text_to_chars(text)
        chars = [ch for ch in chars if ch in self.phn2num]
        text_tokens = [self.phn2num[ch] for ch in chars]
        text_tensor = torch.LongTensor(text_tokens).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Encode caption with ViT5-large
            batch_enc = self.caption_tokenizer(caption, return_tensors="pt")
            ori_tokens = batch_enc["input_ids"].to(self.device)
            prompt = self.caption_encoder(
                input_ids=ori_tokens
            ).last_hidden_state.squeeze().unsqueeze(0).to(self.device)

            # Use pre-computed CLAP "none" embedding
            clap = self.clap_none

            # Predict duration using PhoBERT model (or heuristic fallback)
            if duration is None:
                duration = self.predict_duration(text, caption)

        # Apply speed
        if speed <= 0:
            speed = 1.0
        duration = duration / speed

        print(f"⏱️  Duration: {duration:.2f}s (speed={speed}x)")

        # Generate mel spectrogram
        n_frames = math.ceil(duration * SAMPLE_RATE / 256)
        audio_clips = torch.zeros([1, n_frames, 100]).to(self.device)
        cond = None

        seq_len_prompt = prompt.shape[1]
        prompt_lens = torch.Tensor([prompt.shape[1]])
        prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)

        # Flow matching ODE sampling
        gen_wav = sample(
            self.model, self.vocoder,
            audio_clips, cond, text_tensor, prompt, clap, prompt_mask,
            steps=steps, cfg=cfg,
            sway_sampling_coef=-1.0, device=self.device,
        )

        elapsed = time.time() - start_time
        audio_len = gen_wav.shape[-1] / SAMPLE_RATE
        rtf = elapsed / audio_len if audio_len > 0 else 0
        print(f"✅ Generated {audio_len:.2f}s audio in {elapsed:.2f}s (RTF={rtf:.3f})")

        # Save WAV
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, gen_wav, SAMPLE_RATE)
        print(f"💾 Saved to: {output_path}\n")

        return gen_wav

    def synthesize_batch(
        self,
        texts: list[str],
        captions: list[str],
        output_dir: str = "outputs",
        batch_size: int = 4,
        durations: list[float] | None = None,
        speed: float = 1.0,
        steps: int = 25,
        cfg: float = 2.0,
        seed: Optional[int] = None,
    ) -> list[np.ndarray]:
        """True GPU-parallelized batch synthesis using batch_sample().

        Processes items in parallel via padding + masking + matrix multiplication.
        Sorts by duration to minimize padding waste within each sub-batch.

        Args:
            texts: List of Vietnamese texts.
            captions: List of voice descriptions (same length as texts).
            output_dir: Directory to save WAV files.
            batch_size: Number of items per GPU batch (default: 4).
            durations: Optional list of durations (None = auto per item).
            speed: Speed factor.
            steps: ODE solver steps.
            cfg: Classifier-free guidance scale.
            seed: Random seed.

        Returns:
            List of numpy audio arrays.
        """
        from capspeech.nar.batch_inference import batch_sample

        assert len(texts) == len(captions), \
            f"texts ({len(texts)}) and captions ({len(captions)}) must have same length"

        os.makedirs(output_dir, exist_ok=True)

        if seed is not None:
            seed_everything(seed)

        n = len(texts)
        if durations is None:
            durations = [None] * n

        start_time = time.time()

        # ── Batch encode captions with ViT5 ──
        unique_captions = list(set(captions))
        with torch.no_grad():
            batch_enc = self.caption_tokenizer(
                unique_captions, return_tensors="pt",
                padding=True, truncation=True
            )
            input_ids = batch_enc["input_ids"].to(self.device)
            attention_mask = batch_enc["attention_mask"].to(self.device)
            outputs = self.caption_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            caption_embeds = outputs.last_hidden_state
            caption_cache = {}
            for i, cap in enumerate(unique_captions):
                actual_len = int(attention_mask[i].sum().item())
                caption_cache[cap] = (caption_embeds[i, :actual_len, :], actual_len)

        # ── Prepare per-item tensors ──
        x_list, text_list, prompt_list, clap_list = [], [], [], []
        prompt_lens_list, duration_frames_list = [], []

        for i in range(n):
            text = texts[i]
            caption = captions[i]
            dur = durations[i]

            # Normalize + Text encoding
            text = self.normalize_text(text)
            chars = text_to_chars(text)
            chars = [ch for ch in chars if ch in self.phn2num]
            text_tokens = [self.phn2num[ch] for ch in chars]
            text_list.append(torch.LongTensor(text_tokens).to(self.device))

            # Caption from cache
            prompt_emb, prompt_len = caption_cache[caption]
            prompt_list.append(prompt_emb)
            prompt_lens_list.append(prompt_len)

            # CLAP
            clap_list.append(self.clap_none.squeeze(0).clone())

            # Duration
            if dur is None:
                dur = self.predict_duration(text, caption)
            if speed > 0:
                dur = dur / speed
            n_frames = math.ceil(dur * SAMPLE_RATE / 256)
            duration_frames_list.append(n_frames)

            x_list.append(torch.zeros(n_frames, 100, device=self.device))

        # ── Sort by duration for efficient padding ──
        sorted_indices = sorted(range(n), key=lambda i: duration_frames_list[i])

        # ── Process in batches ──
        all_wavs = [None] * n
        num_batches = math.ceil(n / batch_size)

        for batch_idx in range(num_batches):
            si = batch_idx * batch_size
            ei = min(si + batch_size, n)
            batch_indices = sorted_indices[si:ei]

            wavs = batch_sample(
                model=self.model,
                vocoder=self.vocoder,
                x_list=[x_list[j] for j in batch_indices],
                cond=None,
                text_list=[text_list[j] for j in batch_indices],
                prompt_list=[prompt_list[j] for j in batch_indices],
                clap_list=[clap_list[j] for j in batch_indices],
                prompt_lens_list=[prompt_lens_list[j] for j in batch_indices],
                duration_frames_list=[duration_frames_list[j] for j in batch_indices],
                steps=steps,
                cfg=cfg,
                sway_sampling_coef=-1.0,
                device=str(self.device),
                seed=seed,
            )

            for k, wav in enumerate(wavs):
                idx = batch_indices[k]
                all_wavs[idx] = wav
                out_path = os.path.join(output_dir, f"sample_{idx:04d}.wav")
                sf.write(out_path, wav, SAMPLE_RATE)

        elapsed = time.time() - start_time
        total_audio = sum(len(w) / SAMPLE_RATE for w in all_wavs if w is not None)
        rtf = elapsed / total_audio if total_audio > 0 else 0
        print(f"✅ Batch: {n} samples, {total_audio:.1f}s audio "
              f"in {elapsed:.2f}s (RTF={rtf:.3f})")
        print(f"📁 Saved to {output_dir}/")

        return all_wavs


# ──────────────────────── CLI Entry Point ──────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="CapSpeech NAR — Vietnamese Instruction TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple generation
  python api.py \\
      --text "Xin chào, hôm nay bạn có khoẻ không?" \\
      --caption "Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc."

  # With custom settings
  python api.py \\
      --text "Thời tiết hôm nay rất đẹp." \\
      --caption "Giọng nam trung niên, nhịp nói vừa phải." \\
      --duration 4.0 --steps 50 --cfg 3.0 \\
      --output output/test.wav

  # Using CPU
  python api.py \\
      --text "Xin chào" \\
      --caption "Giọng nữ" \\
      --device cpu
        """,
    )
    parser.add_argument("--text", type=str, required=True,
                        help="Vietnamese text to synthesize")
    parser.add_argument("--caption", type=str, required=True,
                        help="Voice description / instruction in Vietnamese")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output WAV file path (default: output.wav)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Audio duration in seconds (default: auto)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed factor (default: 1.0)")
    parser.add_argument("--steps", type=int, default=25,
                        help="ODE solver steps (default: 25)")
    parser.add_argument("--cfg", type=float, default=2.0,
                        help="Classifier-free guidance scale (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda:0, cpu, etc. (default: auto)")
    parser.add_argument("--hf-repo", type=str, default=HF_MODEL_REPO,
                        help=f"HuggingFace model repo (default: {HF_MODEL_REPO})")
    args = parser.parse_args()

    tts = InstructVoiceAPI(
        device=args.device,
        hf_model_repo=args.hf_repo,
        seed=args.seed,
    )

    tts.synthesize(
        text=args.text,
        caption=args.caption,
        output_path=args.output,
        duration=args.duration,
        speed=args.speed,
        steps=args.steps,
        cfg=args.cfg,
    )


if __name__ == "__main__":
    main()
