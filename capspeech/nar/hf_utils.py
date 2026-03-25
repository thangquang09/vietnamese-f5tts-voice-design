#!/usr/bin/env python3
"""Helpers for non-interactive Hugging Face auth and downloads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_ENV_CANDIDATES = (
    "/data1/speech/nhandt23/.env",
    "/data1/speech/nhandt23/06_thang/.env",
    "/mnt/speech/nhandt23/.env",
    "/mnt/speech/nhandt23/06_thang/.env",
)


def _load_env_file(env_path: str) -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_hf_token(env_file: Optional[str] = None, token_env_var: str = "HF_TOKEN") -> str:
    token = os.environ.get(token_env_var)
    if token:
        return token

    candidates = [env_file] if env_file else []
    candidates.extend(DEFAULT_ENV_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        _load_env_file(candidate)
        token = os.environ.get(token_env_var)
        if token:
            return token

    raise RuntimeError(
        f"Missing {token_env_var}. Set it in the environment or provide an .env file containing {token_env_var}=..."
    )


def snapshot_repo_to_tmp(
    repo_id: str,
    cache_dir: str,
    token: str,
    allow_patterns: Optional[Iterable[str]] = None,
) -> str:
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        local_dir_use_symlinks=False,
    )


def hf_hub_download_to_tmp(repo_id: str, filename: str, cache_dir: str, token: str) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        token=token,
        local_dir_use_symlinks=False,
    )
