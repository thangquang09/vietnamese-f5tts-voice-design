"""
Convert Vietnamese Instruction TTS dataset (CSV) to CapSpeech NAR format.

Input:  results/final_dataset/{train,val,test}.csv
Output: 
  - jsons/{split}.json   (segment_id, audio_path, text, caption, duration, source)
  - manifest/{split}.txt (segment_id \t tag)
"""

import os
import json
import csv
import argparse
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Vietnamese CSV dataset to CapSpeech NAR format")
    parser.add_argument('--csv_dir', type=str, required=True,
                        help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory to save processed data (jsons/, manifest/)")
    parser.add_argument('--caption_column', type=str, default='caption_full',
                        choices=['caption_full', 'caption_partial'],
                        help="Which caption column to use")
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help="Splits to process")
    parser.add_argument('--task_filter', type=str, default='general',
                        help="Only include rows with this task value (e.g. general, accent, emotion). "
                             "Set to 'all' to include everything.")
    parser.add_argument('--max_duration', type=float, default=18.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument('--min_duration', type=float, default=1.0,
                        help="Minimum audio duration in seconds")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of workers for parallel duration computation")
    return parser.parse_args()


def get_audio_duration(audio_path):
    """Get audio duration in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return None


def process_row(row, caption_column):
    """Process a single CSV row, return (entry, segment_id) or None."""
    segment_id = row['id']
    audio_path = row['audio_path']
    transcript = row['transcript']
    caption = row.get(caption_column, '')
    dataset = row.get('dataset', 'unknown')

    if not transcript or not caption or not audio_path:
        return None

    # Remap /mnt → /data1 for cross-server compatibility
    audio_path = audio_path.replace('/mnt/', '/data1/')

    if not os.path.exists(audio_path):
        return None

    duration = get_audio_duration(audio_path)
    if duration is None:
        return None

    entry = {
        "segment_id": segment_id,
        "audio_path": audio_path,
        "text": transcript,
        "caption": caption,
        "duration": round(duration, 5),
        "source": dataset
    }
    return entry


def process_split(csv_path, save_dir, split, caption_column, 
                  min_duration, max_duration, num_workers, task_filter='general'):
    """Process a single split."""
    json_dir = os.path.join(save_dir, "jsons")
    manifest_dir = os.path.join(save_dir, "manifest")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_before = len(rows)
    if task_filter and task_filter != 'all':
        rows = [r for r in rows if r.get('task', '') == task_filter]
    print(f"Processing {split}: {len(rows)} rows (filtered from {total_before}, task={task_filter})")

    # Process rows in parallel with timeout to avoid NFS hangs
    entries = []
    skipped = 0
    timeout_count = 0

    # Process in batches to avoid submitting too many NFS calls at once
    BATCH_SIZE = 500
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[batch_start:batch_start + BATCH_SIZE]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_row, row, caption_column): row 
                for row in batch_rows
            }
            for future in tqdm(as_completed(futures, timeout=30), total=len(futures), 
                               desc=f"Processing {split} [{batch_start}:{batch_start+len(batch_rows)}]",
                               leave=False):
                try:
                    entry = future.result(timeout=10)
                except Exception:
                    skipped += 1
                    timeout_count += 1
                    continue
                if entry is None:
                    skipped += 1
                    continue
                if entry['duration'] < min_duration or entry['duration'] > max_duration:
                    skipped += 1
                    continue
                entries.append(entry)

    if timeout_count > 0:
        print(f"  ⚠ {timeout_count} entries timed out (NFS slow)")
    print(f"  Processed: {len(entries)}, skipped: {skipped}")

    # Sort by segment_id for reproducibility
    entries.sort(key=lambda x: x['segment_id'])

    # Save JSON
    json_path = os.path.join(json_dir, f"{split}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    # Save manifest (segment_id \t tag)
    # All Vietnamese samples use "none" as CLAP tag (no sound events)
    manifest_path = os.path.join(manifest_dir, f"{split}.txt")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(f"{entry['segment_id']}\tnone\n")

    print(f"  → Saved {len(entries)} entries ({skipped} skipped)")
    print(f"  → JSON: {json_path}")
    print(f"  → Manifest: {manifest_path}")

    return len(entries)


if __name__ == "__main__":
    args = parse_args()

    total = 0
    for split in args.splits:
        csv_path = os.path.join(args.csv_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        
        count = process_split(
            csv_path, args.save_dir, split, args.caption_column,
            args.min_duration, args.max_duration, args.num_workers,
            task_filter=args.task_filter
        )
        total += count

    print(f"\nTotal: {total} entries across {len(args.splits)} splits")
