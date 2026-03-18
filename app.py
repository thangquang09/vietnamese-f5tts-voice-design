#!/usr/bin/env python3
"""
InstructVoice — Vietnamese Instruction TTS — Gradio Web UI

Launch:
    python app.py [--device cuda:0] [--port 7860] [--share]
"""

import argparse
import tempfile
import os

import gradio as gr
from api import InstructVoiceAPI

# ──────────────────────── Global TTS instance ──────────────────────── #
tts: InstructVoiceAPI = None  # type: ignore


def load_model(device: str):
    """Load the TTS model once at startup."""
    global tts
    print(f"\n🚀 Loading InstructVoice model on {device}...")
    tts = InstructVoiceAPI(device=device)


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


# ──────────────────────── Example data ──────────────────────── #

EXAMPLES = [
    [
        "Xin chào, hôm nay bạn có khoẻ không?",
        "Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc.",
        0, 1.0, 25, 2.0, 42,
    ],
    [
        "Thời tiết hôm nay rất đẹp, chúng ta ra ngoài chơi đi.",
        "Giọng nam trung niên, nhịp nói vừa phải, giọng miền Nam.",
        0, 1.0, 25, 2.0, 42,
    ],
    [
        "Tin tức mới nhất cho thấy nền kinh tế đang phục hồi mạnh mẽ.",
        "Giọng nam, già, giọng trầm, nói rõ ràng.",
        0, 1.0, 32, 2.0, 42,
    ],
    [
        "Xin mời quý vị hành khách chuẩn bị lên tàu.",
        "Giọng nữ, trẻ tuổi, nói nhanh, giọng cao.",
        0, 1.2, 25, 2.0, 42,
    ],
    [
        "Chúc mừng năm mới! Kính chúc quý vị một năm an khang thịnh vượng.",
        "Giọng nữ trẻ, nói chậm, giọng miền Trung, ấm áp.",
        0, 0.9, 25, 2.5, 42,
    ],
]


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

                gr.Markdown("""
                ### 💡 Gợi ý mô tả giọng nói

                | Mô tả | Phong cách |
                |:------|:-----------|
                | `Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc.` | Nữ trẻ, chậm, Bắc |
                | `Giọng nam trung niên, nhịp nói vừa phải, giọng miền Nam.` | Nam trung niên, vừa, Nam |
                | `Giọng nữ, trẻ tuổi, nói nhanh, giọng cao.` | Nữ trẻ, nhanh, cao |
                | `Giọng nam, già, giọng trầm, nói rõ ràng.` | Nam già, trầm, rõ |
                """)

        # ─── Examples ───
        gr.Examples(
            examples=EXAMPLES,
            inputs=[
                text_input, caption_input,
                duration_slider, speed_slider,
                steps_slider, cfg_slider, seed_input,
            ],
            outputs=audio_output,
            fn=synthesize,
            cache_examples=False,
            label="📋 Ví dụ",
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
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda:0, cpu, etc. (default: auto)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server (default: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    load_model(device)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
