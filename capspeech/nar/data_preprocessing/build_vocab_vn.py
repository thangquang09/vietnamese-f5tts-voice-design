"""
Build Vietnamese character-level vocabulary from processed JSON data.

Scans all transcripts and builds a vocab.txt file where each line is:
  index token

Vietnamese text is tokenized at character level (giống F5-TTS-Vietnamese).
"""

import os
import json
import glob
import argparse
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Build Vietnamese character vocabulary")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory containing jsons/ folder and where vocab.txt will be saved")
    return parser.parse_args()


def extract_chars_from_text(text):
    """Extract individual characters, replacing spaces with <BLK>."""
    chars = []
    for ch in text.lower():
        if ch == ' ':
            chars.append('<BLK>')
        else:
            chars.append(ch)
    return chars


if __name__ == "__main__":
    args = parse_args()

    json_dir = os.path.join(args.save_dir, "jsons")
    json_paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))

    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    # Collect all characters
    char_counter = Counter()
    total_texts = 0

    for json_path in json_paths:
        print(f"Scanning {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            text = item['text']
            chars = extract_chars_from_text(text)
            char_counter.update(chars)
            total_texts += 1

    print(f"\nTotal texts scanned: {total_texts}")
    print(f"Unique tokens: {len(char_counter)}")

    # Sort alphabetically, but put <BLK> first
    special_tokens = ['<BLK>']
    regular_tokens = sorted([ch for ch in char_counter.keys() if ch not in special_tokens])
    all_tokens = special_tokens + regular_tokens

    # Write vocab.txt with format: "index token"
    vocab_path = os.path.join(args.save_dir, "vocab.txt")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for idx, token in enumerate(all_tokens):
            f.write(f"{idx} {token}\n")

    print(f"\nVocab saved to {vocab_path}")
    print(f"Vocab size: {len(all_tokens)}")

    # Print top 20 most common tokens
    print("\nTop 20 most common tokens:")
    for token, count in char_counter.most_common(20):
        print(f"  '{token}': {count}")
