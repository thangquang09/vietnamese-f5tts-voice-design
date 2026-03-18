import modal
import os
import fastapi

# ── Modal Image ──────────────────────────────────────────────────────────── #

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0",
        "torchaudio",
        "transformers==4.46.1",
        "huggingface_hub",
        "soundfile",
        "einops",
        "torchdiffeq",
        "x_transformers",
        "sentencepiece",
        "protobuf>=4.0.0",
        "librosa",
        "numpy",
        "pypinyin",
        "matplotlib",
        "pandas",
        "tqdm",
        "pyyaml",
        "scipy",
        "jieba",
        "datasets",
        "fastapi",
    )
    .run_commands(
        "huggingface-cli download thangquang09/capspeech-nar-vietnamese",
        "huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x",
        "huggingface-cli download VietAI/vit5-large",
    )
    .add_local_dir(".", remote_path="/root/app")
)

app = modal.App("vietnamese-instruction-tts-api", image=image)


# ── TTS API Service ──────────────────────────────────────────────────────── #


@app.cls(
    gpu="T4",
    timeout=300,
    keep_warm=0,
)
class TTSModalAPI:
    @modal.enter()
    def setup(self):
        """Load all models into VRAM once on container start."""
        import sys

        sys.path.append("/root/app")
        os.chdir("/root/app")

        from api import InstructVoiceAPI

        print("Loading InstructVoice model...")
        self.tts = InstructVoiceAPI(device="cuda:0")
        print("Server ready.")

    @modal.web_endpoint(method="POST")
    async def generate(self, raw_request: fastapi.Request):
        """Receive a POST request and return synthesized WAV audio."""
        import uuid
        from fastapi.responses import JSONResponse

        data = await raw_request.json()

        text = data.get("text", "Xin chào, đây là hệ thống thử nghiệm.")
        caption = data.get("caption", "Giọng nữ trẻ, nói chậm rãi, giọng miền Bắc.")
        speed = float(data.get("speed", 1.0))
        duration = data.get("duration", None)
        if duration is not None:
            duration = float(duration)

        tmp_filename = f"/tmp/{uuid.uuid4()}.wav"

        try:
            self.tts.synthesize(
                text=text,
                caption=caption,
                output_path=tmp_filename,
                duration=duration,
                speed=speed,
            )

            with open(tmp_filename, "rb") as f:
                audio_bytes = f.read()

            return fastapi.Response(content=audio_bytes, media_type="audio/wav")

        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )

        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)