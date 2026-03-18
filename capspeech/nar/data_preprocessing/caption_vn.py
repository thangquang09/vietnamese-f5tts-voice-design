"""
Encode Vietnamese captions using ViT5-large model.

Replaces the original caption.py which uses Flan-T5-large (English only).
ViT5-large outputs embeddings with dim=1024, matching CapSpeech config.

Input:  jsons/{split}.json (reads 'caption' field)
Output: t5/{segment_id}.npz
"""

import argparse
import logging
import json
import os
import numpy as np
import torch
import tqdm
import time
import glob
from transformers import T5EncoderModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode Vietnamese captions using ViT5-large")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory containing jsons/ and where t5/ will be created")
    parser.add_argument('--model_name', type=str, default='VietAI/vit5-large',
                        help="HuggingFace model name for caption encoder")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument('--start', type=int, default=0,
                        help='Start index for parallel processing')
    parser.add_argument('--end', type=int, default=10000000,
                        help='End index for parallel processing')
    parser.add_argument('--split', type=str, default=None,
                        help='Process only this split (e.g., "train" or "val"). '
                             'If not set, processes all JSON files.')
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s",
        level=logging.INFO
    )
    args = parse_args()

    logging.info(f"Loading caption encoder: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    caption_encoder = T5EncoderModel.from_pretrained(args.model_name).cuda().eval()
    logging.info("Caption encoder loaded")

    t5_save_root = os.path.join(args.save_dir, "t5")
    os.makedirs(t5_save_root, exist_ok=True)

    stime = time.time()

    json_paths = sorted(glob.glob(os.path.join(args.save_dir, 'jsons', '*.json')))
    if args.split:
        json_paths = [p for p in json_paths if os.path.basename(p) == f"{args.split}.json"]
    logging.info(f"Found {len(json_paths)} JSON files")

    total_processed = 0
    total_skipped = 0

    for json_path in json_paths:
        logging.info(f"Processing {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            jsondata = json.load(f)

        jsondata = jsondata[args.start:args.end]

        for key in tqdm.tqdm(range(len(jsondata)), desc=os.path.basename(json_path)):
            segment_id = jsondata[key]['segment_id']
            save_fn = os.path.join(t5_save_root, f"{segment_id}.npz")

            if os.path.exists(save_fn):
                total_skipped += 1
                continue

            caption = jsondata[key]['caption']

            with torch.no_grad():
                batch_encoding = tokenizer(
                    caption, 
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                input_ids = batch_encoding["input_ids"].cuda()
                outputs = caption_encoder(input_ids=input_ids).last_hidden_state

            emb = outputs.cpu().numpy()
            np.savez_compressed(save_fn, emb)
            total_processed += 1

    elapsed = time.time() - stime
    logging.info(f"Done in {elapsed:.1f}s")
    logging.info(f"Processed: {total_processed}, Skipped (existing): {total_skipped}")
