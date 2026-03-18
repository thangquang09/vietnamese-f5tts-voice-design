"""
Vietnamese character-level tokenization for CapSpeech NAR.

Instead of English G2P phonemes, this script converts Vietnamese text
to character-level tokens (following F5-TTS-Vietnamese approach).

Input:  jsons/{split}.json
Output: g2p/{segment_id}.txt  (space-separated character tokens)

Example:
  "xin chào" → "x i n <BLK> c h à o"
"""

import os
import json
import glob
import argparse
import logging
import multiprocessing
from tqdm import tqdm
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vietnamese character-level tokenization for CapSpeech NAR")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory containing jsons/ and where g2p/ will be created")
    parser.add_argument('--num_cpus', type=int, default=16,
                        help="Number of CPUs for parallel processing")
    return parser.parse_args()


def text_to_chars(text):
    """Convert Vietnamese text to character-level tokens.
    
    Space → <BLK>
    Each character → individual token
    Lowercase everything.
    """
    chars = []
    for ch in text.lower():
        if ch == ' ':
            chars.append('<BLK>')
        else:
            chars.append(ch)
    return chars


def process_batch(args_tuple):
    """Process a batch of entries."""
    idx, batch, save_root = args_tuple
    for entry in tqdm(batch, desc=f"Worker {idx}", position=idx, leave=False):
        segment_id = entry['segment_id']
        save_fn = os.path.join(save_root, f"{segment_id}.txt")

        if os.path.exists(save_fn):
            continue

        text = entry['text']
        chars = text_to_chars(text)
        char_seq = " ".join(chars)

        with open(save_fn, "w", encoding="utf-8") as f:
            f.write(char_seq)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s",
        level=logging.INFO
    )
    args = parse_args()

    phn_save_root = os.path.join(args.save_dir, "g2p")
    os.makedirs(phn_save_root, exist_ok=True)

    stime = time.time()

    json_paths = sorted(glob.glob(os.path.join(args.save_dir, 'jsons', '*.json')))
    logging.info(f"Found {len(json_paths)} JSON files")

    for json_path in json_paths:
        logging.info(f"Processing {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            jsondata = json.load(f)

        logging.info(f"  {len(jsondata)} entries")

        # Split data across workers
        import numpy as np
        splits = np.array_split(jsondata, args.num_cpus)
        work_items = [(i, list(split), phn_save_root) for i, split in enumerate(splits)]

        with multiprocessing.Pool(processes=args.num_cpus) as pool:
            pool.map(process_batch, work_items)

    elapsed = time.time() - stime
    logging.info(f"Done in {elapsed:.1f}s")

    # Count output files
    num_files = len(glob.glob(os.path.join(phn_save_root, "*.txt")))
    logging.info(f"Total g2p files: {num_files}")
