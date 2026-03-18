<h1 align="center">🇻🇳 Vietnamese F5-TTS Voice Design</h1>

<p align="center">
  <em>Instruction-guided Vietnamese Text-to-Speech with voice style control</em>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/thangquang09/capspeech-nar-vietnamese"><strong>Model</strong></a> &nbsp;|&nbsp;
  📄 <a href="https://arxiv.org/abs/2506.02863"><strong>Original Paper</strong></a> &nbsp;|&nbsp;
  🔧 <a href="https://github.com/WangHelin1997/CapSpeech"><strong>Upstream Repo</strong></a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg" />
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg" />
  <img alt="License" src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg" />
</p>

---

## 🔊 What is this?

This is a **Vietnamese adaptation** of [CapSpeech NAR](https://github.com/WangHelin1997/CapSpeech) — a non-autoregressive, flow-matching TTS model that supports **voice style control through natural language instructions**.

You describe the voice you want (e.g., *"Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc"*) and the model generates speech that matches your description.

### Key Changes from Original CapSpeech

| Component | Original (English) | This Fork (Vietnamese) |
|:----------|:-------------------|:-----------------------|
| Text tokenization | G2P phonemes | **Character-level** (176 chars) |
| Caption encoder | Flan-T5-large | **ViT5-large** (`VietAI/vit5-large`) |
| CLAP tags | Multiple sound events | `none` only |
| Vocab size | 256 | **176** Vietnamese characters |

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/thangquang09/vietnamese-f5tts-voice-design.git
cd vietnamese-f5tts-voice-design
pip install -r requirements.txt
```

### Inference (CLI)

```bash
python api.py \
    --text "Xin chào, hôm nay bạn có khoẻ không?" \
    --caption "Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc." \
    --output output.wav
```

The model checkpoint, config, and vocab will be **automatically downloaded** from [HuggingFace](https://huggingface.co/thangquang09/capspeech-nar-vietnamese) on first run.

### Inference (Python API)

```python
from api import InstructVoiceAPI

# Initialize — downloads model from HuggingFace automatically
tts = InstructVoiceAPI(device="cuda:0")

# Generate speech
tts.synthesize(
    text="Thời tiết hôm nay rất đẹp.",
    caption="Giọng nam trung niên, nhịp nói vừa phải, giọng miền Nam.",
    output_path="output.wav",
)
```

### Web UI (Gradio)

```bash
python app.py --device cuda:0 --port 7860
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

Features:
- 🎚️ Adjustable parameters (speed, duration, ODE steps, CFG)
- 📋 Pre-built examples with different voice styles
- 🔊 Real-time audio playback
- 🌐 Optional public link with `--share`

### Advanced Options

```python
tts.synthesize(
    text="Xin chào thế giới!",
    caption="Giọng nữ, trẻ tuổi, nói nhanh, giọng cao.",
    output_path="fast_speech.wav",
    duration=3.0,     # Fixed duration (seconds), None = auto
    speed=1.2,        # Speed factor (> 1.0 = faster)
    steps=50,         # ODE solver steps (more = better quality)
    cfg=3.0,          # Classifier-free guidance scale
    seed=123,         # Reproducibility
)

# Batch synthesis
tts.synthesize_batch(
    texts=["Câu một.", "Câu hai.", "Câu ba."],
    captions=["Giọng nữ trẻ."] * 3,
    output_dir="batch_output/",
)
```

### Caption Examples

| Caption | Voice Style |
|:--------|:------------|
| `Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc.` | Young female, slow, Northern accent |
| `Giọng nam trung niên, nhịp nói vừa phải, giọng miền Nam.` | Middle-aged male, moderate, Southern accent |
| `Giọng nữ, trẻ tuổi, nói nhanh, giọng cao.` | Young female, fast, high pitch |
| `Giọng nam, già, giọng trầm, nói rõ ràng.` | Old male, deep voice, clear speech |

## 🏗️ Model Architecture

- **Architecture**: CrossDiT (dim=1024, depth=24, heads=16, ff_mult=4)
- **Text embedding**: `nn.Embedding(177, 512)` — character-level Vietnamese
- **Caption encoder**: ViT5-large (VietAI/vit5-large, dim=1024)
- **CLAP embedding**: `Linear(512→512)` for sound event tags
- **Vocoder**: BigVGAN v2 24kHz 100-band
- **Loss**: Flow Matching (MSE)
- **Parameters**: 614.10M trainable

## 📁 Project Structure

```
├── api.py                              # 🔥 Main inference API
├── app.py                              # 🌐 Gradio web UI
├── capspeech/nar/
│   ├── configs/
│   │   ├── finetune_vn.yaml            # Vietnamese finetune config
│   │   └── pretrain_vn.yaml            # Vietnamese pretrain config
│   ├── data_preprocessing/
│   │   ├── preprocess_vn.py            # CSV → JSON + manifest
│   │   ├── build_vocab_vn.py           # Character-level vocab builder
│   │   ├── phonemize_vn.py             # Character tokenization
│   │   ├── caption_vn.py               # ViT5-large caption encoding
│   │   ├── prepare_clap_none.py        # CLAP "none" embedding
│   │   ├── process_vn.sh              # Single-GPU preprocessing
│   │   └── run_preprocess_4gpu.sh     # Multi-GPU preprocessing
│   ├── generate_vn.py                  # Vietnamese text-to-speech generation
│   ├── finetune.py                     # Training script
│   ├── push_to_hf.py                   # Push checkpoints to HuggingFace
│   ├── accelerate_config.yaml          # DDP config (FSDP2 fix)
│   └── network/crossdit.py             # CrossDiT model architecture
├── setup.py
└── requirements.txt
```

## 🏋️ Training

### Data Preprocessing

```bash
cd capspeech/nar/data_preprocessing
bash run_preprocess_4gpu.sh
```

The pipeline has 5 stages:

| Stage | Description | GPU? |
|:------|:-----------|:-----|
| 0 | CSV → JSON + Manifest | ❌ |
| 1 | Build vocabulary (176 chars) | ❌ |
| 2 | Character tokenization | ❌ |
| 3 | ViT5-large caption encoding | ✅ (4 GPU parallel) |
| 4 | CLAP "none" embedding | ❌ |

### Finetune from Pretrained

```bash
cd capspeech/nar

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config.yaml \
    finetune.py \
    --config-name configs/finetune_vn.yaml \
    --pretrained-ckpt <path_to_nar_CapTTS.pt> \
    --epochs 5 \
    --save-every-step 2000 \
    --max-ckpts 3 \
    --amp fp16
```

### Training Details

| Parameter | Value |
|:----------|:------|
| Base model | [OpenSound/CapSpeech-models](https://huggingface.co/OpenSound/CapSpeech-models) |
| Training data | ~1.05M Vietnamese speech samples |
| Batch size | 32 × 2 GPUs |
| Gradient accumulation | 2 |
| Effective batch | 128 |
| Mixed precision | fp16 |
| Hardware | 2× NVIDIA A100 40GB |

## 📦 HuggingFace Model

The trained checkpoint is available at: [**thangquang09/capspeech-nar-vietnamese**](https://huggingface.co/thangquang09/capspeech-nar-vietnamese)

Files included:
- `checkpoint.pt` — Model + optimizer state dict
- `finetune_vn.yaml` — Training configuration
- `vocab.txt` — Vietnamese character vocabulary (176 chars)

## 🔮 Future Work

- **Vietnamese Duration Predictor** — The original CapSpeech uses a BERT-based duration predictor trained on English data (`OpenSound/CapSpeech`). We currently estimate duration from text length. A PhoBERT-based duration predictor trained on Vietnamese speech data would improve timing accuracy.
- **More Training Epochs** — Current checkpoint was trained for ~0.5 epoch. More epochs should improve voice quality and instruction following.
- **Gradio Spaces** — Deploy the model to HuggingFace Spaces for web-based demo.
- **Streaming Inference** — Support chunk-by-chunk audio generation for real-time applications.
- **Voice Cloning Integration** — Combine with F5-TTS-Vietnamese for voice cloning + instruction control.

## 📝 Citation

This work is based on [CapSpeech](https://github.com/WangHelin1997/CapSpeech):

```bibtex
@misc{wang2025capspeech,
    title={CapSpeech: Enabling Downstream Applications in Style-Captioned Text-to-Speech}, 
    author={Helin Wang and Jiarui Hai and Dading Chong and Karan Thakkar and Tiantian Feng and Dongchao Yang and Junhyeok Lee and Laureano Moro Velazquez and Jesus Villalba and Zengyi Qin and Shrikanth Narayanan and Mounya Elhiali and Najim Dehak},
    year={2025},
    eprint={2506.02863},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
}
```

## 📜 License

Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](./LICENSE).

## 🙏 Acknowledgements

- [CapSpeech](https://github.com/WangHelin1997/CapSpeech) — Base model and architecture
- [VietAI/vit5-large](https://huggingface.co/VietAI/vit5-large) — Vietnamese T5 encoder
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — Neural vocoder
- [F5-TTS](https://github.com/SWivid/F5-TTS) — Flow matching TTS inspiration
