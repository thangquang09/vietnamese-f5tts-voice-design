#!/usr/bin/env python3
"""
Build Stage 2 mixed dataset for CapSpeech NAR Vietnamese (v2).

Strategy:
- Use `kw_age` and `kw_emotion` columns directly (no regex on caption)
- Priority: emotion > age (if sample has kw_emotion, it goes to emotion bucket)
- Ignore task column — group purely by kw_age / kw_emotion
- Adult rehearsal = random sample from "người trưởng thành" with no emotion

Groups:
  1. emotion     (kw_emotion non-empty, any age)        ×15
  2. children    (kw_age = "trẻ em", no emotion)         ×2
  3. teen        (kw_age = "thanh thiếu niên", no emo)   ×3
  4. senior      (kw_age = "cao tuổi", no emotion)       ×8
  5. adult       (kw_age = "người trưởng thành", no emo) ×1, capped

Output: jsons/{train,val}.json + manifest/{train,val}.txt in save_dir
"""

import os
import csv
import json
import random
import argparse
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# Age keywords in kw_age column
AGE_MAP = {
    "trẻ em": "children",
    "thanh thiếu niên": "teen",
    "cao tuổi": "senior",
    "người trưởng thành": "adult",
}

UPSAMPLE_FACTORS = {
    "emotion": 15,
    "children": 2,
    "teen": 3,
    "senior": 8,
    "adult_rehearsal": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage 2 mixed dataset (v2)")
    parser.add_argument('--csv_dir', type=str,
                        default='/data1/speech/nhandt23/06_thang/vn-instructiontts/results/final_dataset',
                        help="Directory containing train.csv, val.csv")
    parser.add_argument('--save_dir', type=str,
                        default='/data1/speech/nhandt23/06_thang/capspeech_stage2',
                        help="Output directory (NFS)")
    parser.add_argument('--caption_column', type=str, default='caption_full')
    parser.add_argument('--adult_rehearsal_train', type=int, default=30000,
                        help="Number of adult rehearsal samples for train split")
    parser.add_argument('--adult_rehearsal_val', type=int, default=3000,
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


def classify_row(row):
    """Classify a row into a bucket using kw_emotion and kw_age columns.
    
    Priority: emotion > age group.
    Returns bucket name: 'emotion', 'children', 'teen', 'senior', or 'adult'.
    """
    emotion = row.get('kw_emotion', '').strip()
    if emotion:
        return 'emotion'
    
    age_kw = row.get('kw_age', '').strip()
    return AGE_MAP.get(age_kw, 'adult')


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
    }


def build_stage2_split(csv_path, args, split):
    """Build Stage 2 data for one split."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Classify rows by emotion > age priority
    buckets = {
        "emotion": [],
        "children": [],
        "teen": [],
        "senior": [],
        "adult": [],
    }

    for row in rows:
        bucket = classify_row(row)
        if bucket in buckets:
            buckets[bucket].append(row)
        else:
            buckets['adult'].append(row)

    # Print raw stats
    print(f"\n{'='*60}")
    print(f"Split: {split}")
    print(f"{'='*60}")
    for k, v in buckets.items():
        print(f"  {k:20s}: {len(v):>8d} raw samples")

    # Print emotion breakdown
    emo_breakdown = {}
    for row in buckets['emotion']:
        emo = row.get('kw_emotion', '').strip()
        emo_breakdown[emo] = emo_breakdown.get(emo, 0) + 1
    if emo_breakdown:
        print(f"\n  Emotion breakdown:")
        for k, v in sorted(emo_breakdown.items(), key=lambda x: -x[1]):
            print(f"    {k:20s}: {v:>6d}")

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
            factor = UPSAMPLE_FACTORS.get(factor_key, 1)
            upsampled = count * factor
            total += upsampled
            print(f"  {k:20s}: {upsampled:>8d} after upsample (×{factor})")
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

    json_path = os.path.join(json_dir, f"{split}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    manifest_path = os.path.join(manifest_dir, f"{split}.txt")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(f"{entry['segment_id']}\tnone\n")

    print(f"  → JSON: {json_path} ({len(entries)} entries)")
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
