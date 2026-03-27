#!/usr/bin/env python3
"""
InstructVoice — Vietnamese Instruction TTS — Gradio Web UI

Launch:
    python app.py [--device cuda:0] [--port 7860] [--share]
"""

import argparse
import tempfile
import os
import json

import gradio as gr
from api import InstructVoiceAPI

# ──────────────────────── Global TTS instance ──────────────────────── #
tts: InstructVoiceAPI = None  # type: ignore


def load_model(hf_model_repo: str, device: str):
    """Load the TTS model once at startup."""
    global tts
    tts = InstructVoiceAPI(device=device, hf_model_repo=hf_model_repo)


def synthesize(
    text: str,
    caption: str,
    duration: float,
    speed: float,
    steps: int,
    cfg: float,
    seed: int,
):
    """Generate speech and return audio file path."""
    if not text.strip():
        raise gr.Error("Vui lòng nhập văn bản cần tổng hợp!")
    if not caption.strip():
        raise gr.Error("Vui lòng nhập mô tả giọng nói!")

    # Duration: 0 means auto
    dur = None if duration <= 0 else duration

    # Create temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    tmp.close()

    tts.synthesize(
        text=text,
        caption=caption,
        output_path=out_path,
        duration=dur,
        speed=speed,
        steps=steps,
        cfg=cfg,
        seed=seed if seed >= 0 else None,
    )

    return out_path



# ──────────────────────── Demo Samples (Ground Truth) ──────────────────────── #

DEMO_SAMPLES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demo_samples.json",
)

def _load_demo_samples():
    """Load curated demo samples from JSON."""
    if not os.path.exists(DEMO_SAMPLES_PATH):
        print(f"⚠️  Demo samples not found at {DEMO_SAMPLES_PATH}")
        return []
    with open(DEMO_SAMPLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

DEMO_SAMPLES = _load_demo_samples()


def _attr_badges_html(attrs: dict) -> str:
    """Render voice attributes as colored badge HTML."""
    badge_colors = {
        "gender":         ("#f48fb1", "👤"),
        "age":            ("#ce93d8", "🎂"),
        "accent":         ("#90caf9", "🗺️"),
        "speaking_rate":  ("#ffcc80", "🏃"),
        "pitch":          ("#a5d6a7", "🎵"),
        "loudness":       ("#ef9a9a", "🔊"),
        "expressiveness": ("#80deea", "🎭"),
    }
    html = '<div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:6px;">'
    for key, (color, icon) in badge_colors.items():
        val = attrs.get(key)
        if val:
            html += (
                f'<span style="background:{color}18; color:{color}; '
                f'border:1px solid {color}44; border-radius:12px; '
                f'padding:2px 10px; font-size:0.82em; white-space:nowrap;">'
                f'{icon} {val}</span>'
            )
    html += '</div>'
    return html


# ──────────────────────── Build Gradio UI ──────────────────────── #

def build_ui():
    """Build and return the Gradio interface."""

    css = """
    .main-title {
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 1.5em;
    }
    .demo-card {
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: #1a1a2e;
        transition: box-shadow 0.2s;
    }
    .demo-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        border-color: #555;
    }
    .demo-card-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .demo-card-header .badge {
        background: #1976d2;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .demo-card-header .sample-id {
        color: #888;
        font-size: 0.78em;
        font-family: monospace;
    }
    .caption-box {
        background: #0d2137;
        border-left: 3px solid #42a5f5;
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-style: italic;
        color: #90caf9;
    }
    .caption-partial-box {
        background: #2a1a00;
        border-left: 3px solid #ffa726;
        padding: 6px 12px;
        border-radius: 0 8px 8px 0;
        margin: 4px 0 8px 0;
        font-size: 0.9em;
        color: #ffcc80;
    }
    .transcript-box {
        background: #111;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.95em;
        line-height: 1.5;
        color: #ccc;
    }
    .section-divider {
        border: none;
        border-top: 2px solid #333;
        margin: 2em 0;
    }
    """

    with gr.Blocks(
        title="🇻🇳 Vietnamese Instruction TTS",
        css=css,
        theme=gr.themes.Soft(),
    ) as demo:

        gr.HTML("""
            <h1 class="main-title">🇻🇳 Vietnamese F5-TTS Voice Design</h1>
            <p class="subtitle">
                Instruction-guided Vietnamese Text-to-Speech — 
                Describe the voice you want, and the model generates matching speech.
            </p>
        """)

        with gr.Row():
            # ─── Left column: Inputs ───
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="📝 Văn bản (Text)",
                    placeholder="Nhập văn bản tiếng Việt cần tổng hợp...",
                    lines=3,
                )
                caption_input = gr.Textbox(
                    label="🎙️ Mô tả giọng nói (Voice Description)",
                    placeholder="Ví dụ: Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc.",
                    lines=2,
                )

                with gr.Accordion("⚙️ Tham số nâng cao", open=False):
                    with gr.Row():
                        duration_slider = gr.Slider(
                            minimum=0, maximum=20, value=0, step=0.5,
                            label="⏱️ Thời lượng (s)",
                            info="0 = tự động ước lượng",
                        )
                        speed_slider = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                            label="🏃 Tốc độ",
                            info="1.0 = bình thường",
                        )
                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=10, maximum=100, value=25, step=5,
                            label="🔄 ODE Steps",
                            info="Càng nhiều càng chất lượng",
                        )
                        cfg_slider = gr.Slider(
                            minimum=1.0, maximum=5.0, value=2.0, step=0.5,
                            label="🎯 CFG Scale",
                            info="Classifier-free guidance",
                        )
                    seed_input = gr.Number(
                        value=42, label="🎲 Seed",
                        info="-1 = ngẫu nhiên",
                        precision=0,
                    )

                generate_btn = gr.Button(
                    "🔊 Tổng hợp giọng nói",
                    variant="primary",
                    size="lg",
                )

            # ─── Right column: Output ───
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="🔈 Kết quả",
                    type="filepath",
                )

        # ─── Event handler ───
        generate_btn.click(
            fn=synthesize,
            inputs=[
                text_input, caption_input,
                duration_slider, speed_slider,
                steps_slider, cfg_slider, seed_input,
            ],
            outputs=audio_output,
        )

        # ─── Demo Samples from Dataset (Ground Truth) ───
        if DEMO_SAMPLES:
            gr.HTML('<hr class="section-divider">')
            gr.HTML("""
                <h2 style="text-align:center; margin-bottom:0.3em;">
                    🎧 Demo Samples từ Dataset
                </h2>
                <p style="text-align:center; color:#666; font-size:0.9em; margin-bottom:1.5em;">
                    Các mẫu ground-truth được chọn lọc từ tập test — thể hiện sự đa dạng
                    về giới tính, vùng miền, tốc độ nói, cao độ và mức biểu cảm.
                    <br>Bấm <strong>"Dùng mẫu này"</strong> để điền text & caption vào form tổng hợp phía trên.
                </p>
            """)

            for idx, sample in enumerate(DEMO_SAMPLES):
                attrs = sample["attributes"]
                badges_html = _attr_badges_html(attrs)

                with gr.Row():
                    with gr.Column(scale=4):
                        gr.HTML(f"""
                        <div class="demo-card">
                            <div class="demo-card-header">
                                <span class="badge">{sample['dataset'].upper()}</span>
                                <span class="badge" style="background:#43a047;">#{idx+1}</span>
                                <span class="sample-id">{sample['id']}</span>
                            </div>
                            <div class="caption-box">
                                📝 <strong>Caption (full):</strong> {sample['caption_full']}
                            </div>
                            <div class="caption-partial-box">
                                ✂️ <strong>Caption (partial):</strong> {sample['caption_partial']}
                            </div>
                            <div class="transcript-box">
                                💬 <strong>Transcript:</strong> {sample['transcript']}
                            </div>
                            {badges_html}
                        </div>
                        """)

                    with gr.Column(scale=1, min_width=120):
                        use_btn = gr.Button(
                            "✨ Dùng mẫu này",
                            variant="secondary",
                            size="sm",
                        )
                        use_btn.click(
                            fn=lambda t=sample["transcript"], c=sample["caption_full"]: (t, c),
                            inputs=[],
                            outputs=[text_input, caption_input],
                        )

        gr.Markdown("""
        ---
        <p style="text-align:center; color:#999; font-size:0.85em;">
            🇻🇳 Vietnamese F5-TTS Voice Design • Based on 
            <a href="https://github.com/WangHelin1997/CapSpeech">CapSpeech</a> •
            Model: <a href="https://huggingface.co/thangquang09/capspeech-nar-vietnamese">thangquang09/capspeech-nar-vietnamese</a>
        </p>
        """)

    return demo


# ──────────────────────── Main ──────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstructVoice Vietnamese TTS — Gradio UI")
    parser.add_argument("--hf_model_repo", type=str, default="thangquang09/capspeech-nar-vietnamese-stage2-v3",
                        help="HuggingFace model repository")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda:0, cpu, etc. (default: auto)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server (default: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    load_model(args.hf_model_repo, device)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
