#!/usr/bin/env python3
"""Build CapSpeech stage2-v3 data with unique rows and pool memberships."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import soundfile as sf
import yaml


KNOWN_MOUNT_PREFIXES = ("/mnt/speech", "/data1/speech")


def parse_args():
    parser = argparse.ArgumentParser(description="Build CapSpeech stage2-v3 dataset.")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--save-dir", default="/tmp/capspeech_data/vn_capspeech_stage2_v3")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--caption-column", default="caption_full")
    parser.add_argument("--mount-remap", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def detect_mount_remap(sample_path: str) -> Optional[tuple[str, str]]:
    if os.path.exists(sample_path):
        return None
    for known_prefix in KNOWN_MOUNT_PREFIXES:
        if sample_path.startswith(known_prefix):
            for alt_prefix in KNOWN_MOUNT_PREFIXES:
                if alt_prefix == known_prefix:
                    continue
                remapped = sample_path.replace(known_prefix, alt_prefix, 1)
                if os.path.exists(remapped):
                    return known_prefix, alt_prefix
    return None


def remap_audio_paths(df: pd.DataFrame, mount_remap: Optional[str]) -> pd.DataFrame:
    if mount_remap:
        old_prefix, new_prefix = mount_remap.split(":", 1)
        df["audio_path"] = df["audio_path"].astype(str).str.replace(old_prefix, new_prefix, n=1)
        return df
    sample_paths = df["audio_path"].dropna().astype(str).head(10).tolist()
    for sample_path in sample_paths:
        remap = detect_mount_remap(sample_path)
        if remap is not None:
            old_prefix, new_prefix = remap
            df["audio_path"] = df["audio_path"].astype(str).str.replace(old_prefix, new_prefix, n=1)
            return df
        if os.path.exists(sample_path):
            return df
    return df


def load_split_df(csv_dir: str, split: str, mount_remap: Optional[str]) -> pd.DataFrame:
    csv_path = Path(csv_dir) / f"{split}.csv"
    df = pd.read_csv(csv_path)
    df["dataset_norm"] = df["dataset"].astype(str).str.strip().str.lower()
    df["task_norm"] = df["task"].astype(str).str.strip().str.lower()
    for col in ("kw_age", "kw_emotion", "kw_accent"):
        df[f"{col}_norm"] = df[col].map(normalize_text)
    df["audio_path"] = df["audio_path"].astype(str)
    df = remap_audio_paths(df, mount_remap)
    return df


def sample_rows(df: pd.DataFrame, count: int, rng: random.Random) -> pd.DataFrame:
    if count >= len(df):
        return df.copy()
    indices = list(df.index)
    rng.shuffle(indices)
    return df.loc[indices[:count]].copy()


def add_membership(unique_rows: Dict[str, dict], row: pd.Series, pool: str, subpool: str, label: str, caption_column: str) -> None:
    segment_id = row["id"]
    entry = unique_rows.get(segment_id)
    if entry is None:
        entry = {
            "segment_id": segment_id,
            "audio_path": row["audio_path"],
            "text": row["transcript"],
            "caption": row[caption_column],
            "source": row["dataset_norm"],
            "pool": pool,
            "subpool": subpool,
            "label": label,
            "clap_tag": "none",
            "pool_memberships": [],
            "subpool_memberships": [],
            "label_memberships": [],
        }
        unique_rows[segment_id] = entry

    if pool not in entry["pool_memberships"]:
        entry["pool_memberships"].append(pool)
    if subpool and subpool not in entry["subpool_memberships"]:
        entry["subpool_memberships"].append(subpool)
    if label and label not in entry["label_memberships"]:
        entry["label_memberships"].append(label)


def region_sample(df: pd.DataFrame, split_cfg: dict, rng: random.Random) -> pd.DataFrame:
    regions = split_cfg["regions"]
    frames = []
    if split_cfg["strategy"] == "split_local_min":
        per_region = min(len(df[df["kw_accent_norm"] == region]) for region in regions)
    else:
        per_region = int(split_cfg["per_region"])
    for region in regions:
        frames.append(sample_rows(df[df["kw_accent_norm"] == region], per_region, rng))
    return pd.concat(frames, ignore_index=False)


def validate_entry(item: dict, min_duration: float, max_duration: float):
    audio_path = item["audio_path"]
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        duration = float(sf.info(audio_path).duration)
    except Exception:
        return None
    if duration < min_duration or duration > max_duration:
        return None
    item = dict(item)
    item["duration"] = round(duration, 5)
    return item


def select_split(df: pd.DataFrame, recipe: dict, split: str, seed: int, caption_column: str) -> tuple[List[dict], dict]:
    rng = random.Random(seed)
    unique_rows: Dict[str, dict] = {}
    summary = defaultdict(dict)

    replay_cfg = recipe["selection"]["replay"][split]["per_source"]
    for source, count in replay_cfg.items():
        chosen = sample_rows(df[(df["dataset_norm"] == source) & (df["task_norm"] == "general")], int(count), rng)
        summary["replay_general"][source] = int(len(chosen))
        for _, row in chosen.iterrows():
            add_membership(unique_rows, row, "replay_general", f"replay:{source}", source, caption_column)

    vimd_rows = df[df["dataset_norm"] == "vimd"]
    summary["accent_vimd_full"]["unique"] = int(len(vimd_rows))
    for _, row in vimd_rows.iterrows():
        province = normalize_text(row.get("kw_accent_province", "")) or "unknown"
        add_membership(unique_rows, row, "accent_vimd_full", f"province:{province}", province, caption_column)

    lsvsc_cfg = recipe["selection"]["accent_lsvsc_bal"][split]
    lsvsc_rows = region_sample(df[df["dataset_norm"] == "lsvsc"], lsvsc_cfg, rng)
    summary["accent_lsvsc_bal"]["unique"] = int(len(lsvsc_rows))
    for _, row in lsvsc_rows.iterrows():
        region = row["kw_accent_norm"] or "unknown"
        add_membership(unique_rows, row, "accent_lsvsc_bal", f"region:{region}", region, caption_column)

    age_sources = set(recipe["selection"]["age_main"]["allowed_datasets"])
    age_rows = df[df["dataset_norm"].isin(age_sources)]
    teen_rows = age_rows[age_rows["kw_age_norm"] == "thanh thiếu niên"]
    adult_rows = age_rows[age_rows["kw_age_norm"] == "người trưởng thành"]
    senior_rows = age_rows[age_rows["kw_age_norm"] == "cao tuổi"]
    child_rows = df[(df["dataset_norm"] == "lsvsc") & (df["kw_age_norm"] == "trẻ em")]
    adult_cap = int(recipe["selection"]["age_main"][split]["adult_cap"])
    child_cap = int(recipe["selection"]["age_child_aux"][split])
    adult_rows = sample_rows(adult_rows, adult_cap, rng)
    child_rows = sample_rows(child_rows, child_cap, rng)
    summary["age_main"]["teen"] = int(len(teen_rows))
    summary["age_main"]["adult"] = int(len(adult_rows))
    summary["age_main"]["senior"] = int(len(senior_rows))
    summary["age_main"]["child_aux"] = int(len(child_rows))
    for label, frame in (
        ("teen", teen_rows),
        ("adult", adult_rows),
        ("senior", senior_rows),
        ("child_aux", child_rows),
    ):
        for _, row in frame.iterrows():
            add_membership(unique_rows, row, "age_main", f"age:{label}", label, caption_column)

    emotion_sources = set(recipe["selection"]["emotion_gt"]["allowed_datasets"])
    emotion_rows = df[df["dataset_norm"].isin(emotion_sources) & (df["kw_emotion_norm"] != "")]
    summary["emotion_gt"] = {emotion: int(count) for emotion, count in emotion_rows["kw_emotion_norm"].value_counts().items()}
    for _, row in emotion_rows.iterrows():
        emotion = row["kw_emotion_norm"]
        add_membership(unique_rows, row, "emotion_gt", f"emotion:{emotion}", emotion, caption_column)

    return list(unique_rows.values()), summary


def write_outputs(entries: List[dict], save_dir: Path, split: str) -> None:
    json_dir = save_dir / "jsons"
    manifest_dir = save_dir / "manifest"
    json_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda item: item["segment_id"])
    with (json_dir / f"{split}.json").open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    with (manifest_dir / f"{split}.txt").open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(f"{entry['segment_id']}\tnone\n")


def main():
    args = parse_args()
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    summary_path = Path(args.summary_json or (Path(args.save_dir) / "summary_v3.json"))
    save_dir = Path(args.save_dir)
    pool_stats_dir = save_dir / "sampling_metadata"
    pool_stats_dir.mkdir(parents=True, exist_ok=True)

    all_summary = {"recipe": args.recipe, "splits": {}}
    selected_cache = {}
    for split in ("train", "val"):
        df = load_split_df(args.csv_dir, split, args.mount_remap)
        selected, stats = select_split(df, recipe, split, args.seed + (0 if split == "train" else 1), args.caption_column)
        all_summary["splits"][split] = {
            "selected_unique_rows": len(selected),
            "pool_stats": stats,
        }
        selected_cache[split] = selected
        (pool_stats_dir / f"{split}_pool_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{split}] selected unique rows: {len(selected)}")
        for pool_name, pool_stats in stats.items():
            print(f"  - {pool_name}: {pool_stats}")

    summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"\nDry run complete. Summary written to {summary_path}")
        return

    min_duration = float(recipe["filters"]["min_duration"])
    max_duration = float(recipe["filters"]["max_duration"])
    for split, entries in selected_cache.items():
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            validated = list(executor.map(validate_entry, entries, [min_duration] * len(entries), [max_duration] * len(entries)))
        validated = [item for item in validated if item is not None]
        write_outputs(validated, save_dir, split)
        all_summary["splits"][split]["validated_unique_rows"] = len(validated)
        print(f"[{split}] wrote {len(validated)} validated rows to {save_dir}")

    summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nStage2-v3 data saved to {save_dir}")


if __name__ == "__main__":
    main()
