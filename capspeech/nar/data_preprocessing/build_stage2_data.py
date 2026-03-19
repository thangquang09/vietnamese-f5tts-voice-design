#!/usr/bin/env python3
"""
Build Stage 2 mixed dataset for CapSpeech NAR Vietnamese.

Combines:
- Emotion task (upsampled ×10)
- Accent task (×1)
- General → Senior (×2)
- General → Youth (×5)
- General → Children (×10)
- General → Adult rehearsal (~50K random, ×1)

Output: jsons/{train,val}.json + manifest/{train,val}.txt in save_dir
"""

import os
import csv
import json
import random
import argparse
import re
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# Age group patterns in caption_full
AGE_PATTERNS = {
    "senior": re.compile(r'cao tuổi|già|lớn tuổi', re.IGNORECASE),
    "youth": re.compile(r'thanh thiếu niên|trẻ tuổi', re.IGNORECASE),
    "children": re.compile(r'trẻ em|bé trai|bé gái|em bé|cậu bé|bé nhỏ', re.IGNORECASE),
}

UPSAMPLE_FACTORS = {
    "emotion": 10,
    "accent": 1,
    "senior": 2,
    "youth": 5,
    "children": 100,
    "adult_rehearsal": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage 2 mixed dataset")
    parser.add_argument('--csv_dir', type=str,
                        default='/data1/speech/nhandt23/06_thang/vn-instructiontts/results/final_dataset',
                        help="Directory containing train.csv, val.csv")
    parser.add_argument('--save_dir', type=str,
                        default='/data1/speech/nhandt23/06_thang/capspeech_stage2',
                        help="Output directory (NFS)")
    parser.add_argument('--caption_column', type=str, default='caption_full')
    parser.add_argument('--adult_rehearsal_train', type=int, default=50000,
                        help="Number of adult rehearsal samples for train split")
    parser.add_argument('--adult_rehearsal_val', type=int, default=5000,
                        help="Number of adult rehearsal samples for val split")
    parser.add_argument('--max_duration', type=float, default=18.0)
    parser.add_argument('--min_duration', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true',
                        help="Only print counts, don't write files")
    return parser.parse_args()


def get_audio_duration(audio_path):
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return None


def classify_age_group(caption):
    """Classify age group from caption text."""
    for group, pattern in AGE_PATTERNS.items():
        if pattern.search(caption):
            return group
    return "adult"


def process_row(row, caption_column):
    """Process a single CSV row, return dict or None."""
    segment_id = row['id']
    audio_path = row['audio_path'].replace('/mnt/', '/data1/')
    transcript = row['transcript']
    caption = row.get(caption_column, '')

    if not transcript or not caption or not audio_path:
        return None
    if not os.path.exists(audio_path):
        return None

    duration = get_audio_duration(audio_path)
    if duration is None:
        return None

    return {
        "segment_id": segment_id,
        "audio_path": audio_path,
        "text": transcript,
        "caption": caption,
        "duration": round(duration, 5),
        "source": row.get('dataset', 'unknown'),
        "task": row.get('task', 'general'),
        "age_group": classify_age_group(caption),
    }


def build_stage2_split(csv_path, args, split):
    """Build Stage 2 data for one split."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Classify rows
    buckets = {
        "emotion": [],
        "accent": [],
        "senior": [],
        "youth": [],
        "children": [],
        "adult": [],
    }

    for row in rows:
        task = row.get('task', 'general')
        if task == 'emotion':
            buckets['emotion'].append(row)
        elif task == 'accent':
            buckets['accent'].append(row)
        elif task == 'general':
            caption = row.get(args.caption_column, '')
            age = classify_age_group(caption)
            if age in buckets:
                buckets[age].append(row)
            else:
                buckets['adult'].append(row)

    # Print stats
    print(f"\n{'='*60}")
    print(f"Split: {split}")
    print(f"{'='*60}")
    for k, v in buckets.items():
        print(f"  {k:20s}: {len(v):>8d} raw samples")

    if args.dry_run:
        # Estimate upsampled counts
        total = 0
        for k, v in buckets.items():
            if k == 'adult':
                factor_key = 'adult_rehearsal'
                count = min(len(v), args.adult_rehearsal_train if split == 'train' else args.adult_rehearsal_val)
            else:
                factor_key = k
                count = len(v)
            upsampled = count * UPSAMPLE_FACTORS.get(factor_key, 1)
            total += upsampled
            print(f"  {k:20s}: {upsampled:>8d} after upsample (×{UPSAMPLE_FACTORS.get(factor_key, 1)})")
        print(f"  {'TOTAL':20s}: {total:>8d}")
        return []

    # Process and validate entries in parallel
    all_rows_to_process = []
    row_tags = []  # track which bucket each row belongs to

    for bucket_name, bucket_rows in buckets.items():
        if bucket_name == 'adult':
            # Random sample for rehearsal
            random.seed(args.seed)
            n_rehearsal = args.adult_rehearsal_train if split == 'train' else args.adult_rehearsal_val
            n = min(len(bucket_rows), n_rehearsal)
            selected = random.sample(bucket_rows, n)
            all_rows_to_process.extend(selected)
            row_tags.extend(['adult_rehearsal'] * len(selected))
        else:
            all_rows_to_process.extend(bucket_rows)
            row_tags.extend([bucket_name] * len(bucket_rows))

    print(f"\n  Processing {len(all_rows_to_process)} rows...")

    # Process with workers
    valid_entries = {}  # tag -> list of entries
    for tag in set(row_tags):
        valid_entries[tag] = []

    BATCH_SIZE = 500
    for batch_start in range(0, len(all_rows_to_process), BATCH_SIZE):
        batch_rows = all_rows_to_process[batch_start:batch_start + BATCH_SIZE]
        batch_tags = row_tags[batch_start:batch_start + BATCH_SIZE]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for i, row in enumerate(batch_rows):
                fut = executor.submit(process_row, row, args.caption_column)
                futures[fut] = batch_tags[i]

            for future in tqdm(as_completed(futures, timeout=60), total=len(futures),
                               desc=f"Batch {batch_start}", leave=False):
                tag = futures[future]
                try:
                    entry = future.result(timeout=10)
                except Exception:
                    continue
                if entry is None:
                    continue
                if entry['duration'] < args.min_duration or entry['duration'] > args.max_duration:
                    continue
                valid_entries[tag].append(entry)

    # Upsample
    final_entries = []
    for tag, entries in valid_entries.items():
        factor = UPSAMPLE_FACTORS.get(tag, 1)
        upsampled = entries * factor
        final_entries.extend(upsampled)
        print(f"  {tag:20s}: {len(entries):>6d} valid → ×{factor} = {len(upsampled):>8d}")

    # Shuffle
    random.seed(args.seed)
    random.shuffle(final_entries)

    print(f"  {'TOTAL':20s}: {len(final_entries):>8d}")
    return final_entries


def save_split(entries, save_dir, split):
    """Save JSON and manifest for a split."""
    json_dir = os.path.join(save_dir, "jsons")
    manifest_dir = os.path.join(save_dir, "manifest")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    # Remove internal fields before saving
    clean_entries = []
    for e in entries:
        clean = {k: v for k, v in e.items() if k not in ('task', 'age_group')}
        clean_entries.append(clean)

    json_path = os.path.join(json_dir, f"{split}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_entries, f, ensure_ascii=False, indent=2)

    manifest_path = os.path.join(manifest_dir, f"{split}.txt")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in clean_entries:
            f.write(f"{entry['segment_id']}\tnone\n")

    print(f"  → JSON: {json_path} ({len(clean_entries)} entries)")
    print(f"  → Manifest: {manifest_path}")


def main():
    args = parse_args()

    for split in ['train', 'val']:
        csv_path = os.path.join(args.csv_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping")
            continue

        entries = build_stage2_split(csv_path, args, split)

        if not args.dry_run and entries:
            save_split(entries, args.save_dir, split)

    if not args.dry_run:
        # Copy vocab.txt from stage1
        vocab_src = "/tmp/capspeech_data/vn_capspeech/vocab.txt"
        vocab_dst = os.path.join(args.save_dir, "vocab.txt")
        if os.path.exists(vocab_src):
            import shutil
            shutil.copy2(vocab_src, vocab_dst)
            print(f"\n  → Vocab: {vocab_dst}")
        else:
            print(f"\n  ⚠ vocab.txt not found at {vocab_src}, copy manually")

        print(f"\n✅ Stage 2 data saved to: {args.save_dir}")
        print("Next: run process_stage2.sh to preprocess g2p/t5/clap for new samples")


if __name__ == "__main__":
    main()
