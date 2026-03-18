"""
Create CLAP embedding for the "none" tag.

Vietnamese Instruction TTS dataset does not have sound events,
so all samples use the "none" tag. This script pre-computes the
CLAP text embedding for "none" and saves it.

Output: clap_embs/none.npz
"""

import os
import argparse
import numpy as np
import laion_clap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare CLAP embedding for 'none' tag")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory where clap_embs/ will be created")
    parser.add_argument('--clap_ckpt', type=str, default=None,
                        help="Path to CLAP checkpoint (clap-630k-best.pt). "
                             "If not provided, will try to download from HuggingFace.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    clap_save_dir = os.path.join(args.save_dir, "clap_embs")
    os.makedirs(clap_save_dir, exist_ok=True)

    save_fn = os.path.join(clap_save_dir, "none.npz")

    if os.path.exists(save_fn):
        print(f"CLAP embedding already exists: {save_fn}")
        data = np.load(save_fn)
        print(f"Shape: {data['arr_0'].shape}")
    else:
        print("Loading CLAP model...")
        clap_model = laion_clap.CLAP_Module(enable_fusion=False)

        if args.clap_ckpt and os.path.exists(args.clap_ckpt):
            clap_model.load_ckpt(args.clap_ckpt)
        else:
            # Try downloading from HuggingFace
            try:
                from huggingface_hub import snapshot_download
                local_dir = snapshot_download(repo_id="OpenSound/CapSpeech-models")
                clap_ckpt = os.path.join(local_dir, "clap-630k-best.pt")
                clap_model.load_ckpt(clap_ckpt)
            except Exception as e:
                print(f"Warning: Could not load CLAP checkpoint: {e}")
                print("Using default CLAP model initialization")

        print("Computing CLAP embedding for 'none' tag...")
        tag_data = ["none"]
        tag_embed = clap_model.get_text_embedding(tag_data, use_tensor=False)
        
        np.savez_compressed(save_fn, tag_embed.squeeze())
        print(f"Saved CLAP embedding: {save_fn}")
        print(f"Shape: {tag_embed.squeeze().shape}")
