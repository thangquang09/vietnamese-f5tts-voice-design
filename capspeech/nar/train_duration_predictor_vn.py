#!/usr/bin/env python3
"""Train a Vietnamese Duration Predictor using PhoBERT.

Predicts speech duration (seconds) from caption + text input.
Input format: "caption [SEP] text" → predicted duration (float).

Usage:
    python train_duration_predictor_vn.py \
        --data-dir /tmp/capspeech_data/vn_capspeech \
        --output-dir /tmp/capspeech_duration/phobert_duration_predictor \
        --batch-size 128 \
        --gradient-accumulation-steps 2 \
        --lr 1e-4 \
        --epochs 10 \
        --max-length 256

Data preparation:
    Reads train.json / val.json from --data-dir, extracts duration from
    audio files (filename parsing for vivoice, soundfile.info for dolly).
    Optionally remaps paths via --path-remap (e.g. /mnt/:/data1/).
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import soundfile
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from scipy.stats import pearsonr


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vietnamese Duration Predictor")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing train.json and val.json")
    parser.add_argument("--prepared-dir", type=str, default=None,
                        help="Directory to save/load prepared CSV data (default: <data-dir>/duration_data)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the trained model")
    parser.add_argument("--path-remap", type=str, default=None,
                        help="Path remapping in format 'old:new' (e.g. '/mnt/:/data1/')")
    parser.add_argument("--model-name", type=str, default="vinai/phobert-base",
                        help="PhoBERT model name")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max token length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def extract_duration_from_filename(audio_path: str) -> float | None:
    """Try to extract duration from vivoice-style filename.
    
    Vivoice filenames: <id>_<start_ms>_<end_ms>.wav
    """
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    parts = basename.split("_")
    if len(parts) >= 3:
        try:
            start_ms = int(parts[-2])
            end_ms = int(parts[-1])
            duration = (end_ms - start_ms) / 1000.0
            if 0.1 < duration < 60.0:
                return duration
        except ValueError:
            pass
    return None


def extract_duration_from_audio(audio_path: str) -> float | None:
    """Get duration from audio file header (fast, no full load)."""
    try:
        info = soundfile.info(audio_path)
        return info.duration
    except Exception:
        return None


def prepare_duration_data(args):
    """Prepare duration data from train.json and val.json."""
    prepared_dir = args.prepared_dir or os.path.join(args.data_dir, "duration_data")
    os.makedirs(prepared_dir, exist_ok=True)

    train_csv = os.path.join(prepared_dir, "train.csv")
    val_csv = os.path.join(prepared_dir, "val.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        print(f"Found prepared data at {prepared_dir}, loading...")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
        return train_df, val_df

    # Parse path remapping
    remap_old, remap_new = None, None
    if args.path_remap:
        parts = args.path_remap.split(":")
        if len(parts) == 2:
            remap_old, remap_new = parts[0], parts[1]
            print(f"Path remapping: '{remap_old}' → '{remap_new}'")

    for split_name, json_file, csv_file in [
        ("train", os.path.join(args.data_dir, "jsons", "train.json"), train_csv),
        ("val", os.path.join(args.data_dir, "jsons", "val.json"), val_csv),
    ]:
        print(f"\nPreparing {split_name} data from {json_file}...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        failed = 0
        for entry in tqdm(data, desc=f"Extracting durations ({split_name})"):
            audio_path = entry.get("audio_path", "")
            caption = entry.get("caption", "")
            text = entry.get("text", "")

            if not audio_path or not text:
                failed += 1
                continue

            # Apply path remapping
            if remap_old and remap_new:
                audio_path = audio_path.replace(remap_old, remap_new)

            # Try filename first (fast), then audio header
            duration = extract_duration_from_filename(audio_path)
            if duration is None:
                duration = extract_duration_from_audio(audio_path)

            if duration is None:
                failed += 1
                continue

            # Filter extremes
            if duration < 0.5 or duration > 30.0:
                failed += 1
                continue

            rows.append({
                "text": text,
                "caption": caption,
                "duration": round(duration, 3),
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"  {split_name}: {len(df)} samples saved ({failed} failed/filtered)")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    return train_df, val_df


def main():
    args = parse_args()
    print("=" * 60)
    print("  Vietnamese Duration Predictor Training")
    print("=" * 60)
    print(f"\n  Model: {args.model_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accum: {args.gradient_accumulation_steps}")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Max length: {args.max_length}")
    print()

    # 1. Prepare data
    train_df, val_df = prepare_duration_data(args)

    # 2. Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 3. Create HF Datasets
    def create_dataset(df):
        texts = []
        labels = []
        for _, row in df.iterrows():
            combined = f"{row['caption']} [SEP] {row['text']}"
            texts.append(combined)
            labels.append(float(row["duration"]))
        
        encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
        })
        dataset.set_format("torch")
        return dataset

    print("Creating datasets...")
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 4. Load model
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1, problem_type="regression"
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
    )

    # 6. Custom metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        mae = np.mean(np.abs(predictions - labels))
        return {"mae": mae}

    # 7. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60)
    trainer.train()

    # 8. Save best model
    print("\nSaving best model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 9. Final evaluation
    print("\n" + "=" * 60)
    print("  Final Evaluation")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)):
        with torch.no_grad():
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = outputs.logits.squeeze().cpu().numpy()
            labels = batch["labels"].numpy()
            
            if preds.ndim == 0:
                preds = np.array([preds])
            if labels.ndim == 0:
                labels = np.array([labels])
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae = np.mean(np.abs(all_preds - all_labels))
    r, _ = pearsonr(all_preds, all_labels)
    print(f"\n  MAE:       {mae:.4f} seconds")
    print(f"  Pearson r: {r:.4f}")

    # Sample predictions
    print(f"\n  Sample predictions (10 random):")
    indices = np.random.choice(len(all_preds), min(10, len(all_preds)), replace=False)
    for batch in tqdm(torch.utils.data.DataLoader(
        val_dataset.select(indices.tolist()), batch_size=args.batch_size
    )):
        with torch.no_grad():
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = outputs.logits.squeeze().cpu().numpy()
            labels = batch["labels"].numpy()
            
            if preds.ndim == 0:
                preds = np.array([preds])
            if labels.ndim == 0:
                labels = np.array([labels])
            
            for p, l in zip(preds, labels):
                print(f"    Predicted: {p:.2f}s | GT: {l:.2f}s | Error: {abs(p-l):.2f}s")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
