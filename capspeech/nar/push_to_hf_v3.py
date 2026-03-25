#!/usr/bin/env python3
"""Push CapSpeech stage2-v3 checkpoint to Hugging Face Hub non-interactively."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi, create_repo

from hf_utils import load_hf_token


def create_model_card(save_dir: str, repo_id: str, base_model_repo: str) -> None:
    card = f"""---
language: vi
license: mit
tags:
  - tts
  - text-to-speech
  - vietnamese
  - capspeech
  - instruction-tts
  - stage2-v3
base_model: {base_model_repo}
---

# CapSpeech NAR Vietnamese Stage2-v3

Stage2-v3 release trained from `{base_model_repo}` with replay-aware sampling, full ViMD coverage, balanced LSVSC accents, and curated age/emotion pools.
"""
    with open(os.path.join(save_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def main():
    parser = argparse.ArgumentParser(description="Push CapSpeech stage2-v3 checkpoint to HF")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--repo-name", default="capspeech-nar-vietnamese-stage2-v3")
    parser.add_argument("--username", default="thangquang09")
    parser.add_argument("--config", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--duration-predictor-dir", required=True)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--token-env-var", default="HF_TOKEN")
    parser.add_argument("--base-model-repo", default="thangquang09/capspeech-nar-vietnamese")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = load_hf_token(env_file=args.env_file, token_env_var=args.token_env_var)
    repo_id = f"{args.username}/{args.repo_name}"
    create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private, token=token)

    upload_dir = tempfile.mkdtemp(prefix="capspeech_stage2_v3_")
    try:
        shutil.copy2(args.ckpt, os.path.join(upload_dir, "checkpoint.pt"))
        shutil.copy2(args.config, os.path.join(upload_dir, "finetune_vn_stage2_v3.yaml"))
        shutil.copy2(args.config, os.path.join(upload_dir, "finetune_vn_stage2.yaml"))
        shutil.copy2(args.vocab, os.path.join(upload_dir, "vocab.txt"))
        if os.path.isdir(args.duration_predictor_dir):
            shutil.copytree(args.duration_predictor_dir, os.path.join(upload_dir, "duration_predictor"))
        create_model_card(upload_dir, repo_id, args.base_model_repo)

        print("Files prepared for upload:")
        for root, _, files in os.walk(upload_dir):
            for file_name in sorted(files):
                print(" -", os.path.relpath(os.path.join(root, file_name), upload_dir))

        api = HfApi(token=token)
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload CapSpeech stage2-v3",
        )
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
