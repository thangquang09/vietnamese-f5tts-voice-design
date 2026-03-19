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

You describe the voice you want (e.g., *"Giọng nữ miền Nam cao vút, to và đầy cảm xúc."*) and the model generates speech that matches your description.

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
    --caption "Giọng nữ miền Nam cao vút, to và đầy cảm xúc." \
    --output output.wav
```

The model checkpoint, config, and vocab will be **automatically downloaded** from [HuggingFace](https://huggingface.co/thangquang09/capspeech-nar-vietnamese) on first run.

### Inference (Python API)

```python
from api import InstructVoiceAPI

# Initialize — downloads model from HuggingFace automatically
tts = InstructVoiceAPI(device="cuda:0")

# Single synthesis
tts.synthesize(
    text="Thời tiết hôm nay rất đẹp.",
    caption="Giọng trầm, sinh động của một người đàn ông miền Nam.",
    output_path="output.wav",
)

# Batch synthesis (true GPU-parallelized)
tts.synthesize_batch(
    texts=["Câu một.", "Câu hai.", "Câu ba."],
    captions=["Giọng nữ miền Bắc cao vút, nhanh nhẹn và đầy sinh khí dù nói rất nhẹ."] * 3,
    output_dir="batch_output/",
    batch_size=4,  # optimal on V100
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

### ☁️ Cloud Deployment (Modal)

Deploy the TTS model as a serverless GPU API using [Modal](https://modal.com/):

```bash
# Install Modal CLI
pip install modal
modal setup   # one-time auth

# Deploy
modal deploy modal_app.py
```

This will give you a public HTTPS endpoint. First request triggers a cold start (~1–2 min to load models), subsequent requests are fast.

#### API Usage

```bash
# cURL
curl -X POST https://YOUR_MODAL_ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chào", "caption": "Giọng nữ miền Trung nhỏ nhẹ, đều đều và trầm ấm."}' \
  --output output.wav
```

```python
# Python
import requests

resp = requests.post(
    "https://YOUR_MODAL_ENDPOINT",
    json={
        "text": "Hôm nay trời đẹp quá.",
        "caption": "Giọng nam miền Bắc nhanh, trầm và nhỏ nhẹ.",
        "speed": 1.0,
    },
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

#### Request Body

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `text` | `string` | *required* | Vietnamese text to synthesize |
| `caption` | `string` | *required* | Voice style instruction |
| `speed` | `float` | `1.0` | Speed factor (>1.0 = faster) |
| `duration` | `float\|null` | `null` | Fixed duration in seconds (null = auto) |

**Response**: `audio/wav` binary

### Advanced Options

```python
tts.synthesize(
    text="Xin chào thế giới!",
    caption="Giọng nữ miền Nam cao vút, to và đầy cảm xúc.",
    output_path="fast_speech.wav",
    duration=3.0,     # Fixed duration (seconds), None = auto
    speed=1.2,        # Speed factor (> 1.0 = faster)
    steps=50,         # ODE solver steps (more = better quality)
    cfg=3.0,          # Classifier-free guidance scale
    seed=123,         # Reproducibility
)
```

### Caption Examples

The model accepts natural language voice descriptions. Here are some examples:

| Caption (Vietnamese) | Voice Style |
|:---------------------|:------------|
| `Giọng nữ miền Nam cao vút, to và đầy cảm xúc.` | Female, Southern, high pitch, loud, expressive |
| `Giọng nam miền Bắc nhanh, trầm và nhỏ nhẹ.` | Male, Northern, fast, deep, soft |
| `Giọng nữ miền Trung nhỏ nhẹ, đều đều và trầm ấm.` | Female, Central, soft, steady, warm |
| `Giọng trầm, sinh động của một người đàn ông miền Nam.` | Male, Southern, deep, vivid |
| `Giọng nữ miền Bắc cao vút, nhanh nhẹn và đầy sinh khí dù nói rất nhẹ.` | Female, Northern, high, energetic, very soft |

> **Tip**: Captions can describe gender, age, accent (miền Bắc/Nam/Trung), speaking rate, pitch, loudness, and expressiveness.

## 🏗️ Model Architecture

- **Architecture**: CrossDiT (dim=1024, depth=24, heads=16, ff_mult=4)
- **Text embedding**: `nn.Embedding(177, 512)` — character-level Vietnamese
- **Caption encoder**: ViT5-large (VietAI/vit5-large, dim=1024)
- **CLAP embedding**: `Linear(512→512)` for sound event tags
- **Vocoder**: BigVGAN v2 24kHz 100-band
- **Loss**: Flow Matching (MSE)
- **Parameters**: 614.10M trainable (CrossDiT)
- **Duration predictor**: PhoBERT-base + regression head (~135M params, MAE=0.43s)

### Batch Inference

The model supports true GPU-parallelized batch inference via padding + attention masking:

| Batch Size | RTF | Speedup | Notes |
|:-----------|:----|:--------|:------|
| 1 (loop) | 0.42 | 1.00x | Baseline |
| **4** | **0.23** | **1.87x** | Recommended |
| 8 | 0.25 | 1.71x | Higher padding overhead |

> Benchmarked on V100 32GB, 64 samples from test set, 10 ODE steps.

## 📁 Project Structure

```
├── api.py                              # 🔥 Main inference API
├── app.py                              # 🌐 Gradio web UI
├── modal_app.py                        # ☁️ Modal serverless GPU deployment
├── capspeech/nar/
│   ├── batch_inference.py              # 🚀 GPU batch inference (padding + masking)
│   ├── batch_generate_vn.py            # Batch CLI generation
│   ├── benchmark_performance.py        # Performance benchmark
│   ├── generate_vn.py                  # Single-sample generation
│   ├── network/crossdit.py             # CrossDiT model architecture
│   └── configs/
│       ├── finetune_vn.yaml            # Vietnamese finetune config
│       └── pretrain_vn.yaml            # Vietnamese pretrain config
├── setup.py
└── requirements.txt
```

## 📦 HuggingFace Model

The trained checkpoint is available at: [**thangquang09/capspeech-nar-vietnamese**](https://huggingface.co/thangquang09/capspeech-nar-vietnamese)

Files included:
- `checkpoint.pt` — Model + optimizer state dict
- `finetune_vn.yaml` — Training configuration
- `vocab.txt` — Vietnamese character vocabulary (176 chars)
- `duration_predictor/` — PhoBERT-base duration predictor (config, model, tokenizer)

## 🔮 Future Work

- ~~**Vietnamese Duration Predictor**~~ ✅ Done — PhoBERT-base regression, MAE=0.43s, Pearson r=0.98.
- ~~**Cloud Deployment**~~ ✅ Done — Serverless GPU API via Modal.
- ~~**Batch Inference**~~ ✅ Done — 1.87x speedup at batch_size=4 on V100.
- **Training Code** — Clean, well-documented training pipeline and data preprocessing scripts will be released soon.
- **More Training Epochs** — Current checkpoint was trained for ~0.5 epoch. More epochs should improve voice quality and instruction following.
- **Gradio Spaces** — Deploy to HuggingFace Spaces for web-based demo.
- **Streaming Inference** — Chunk-by-chunk audio generation for real-time applications.
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
