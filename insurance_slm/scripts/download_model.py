#!/usr/bin/env python3
"""
One-shot script to download Phi-3.5-mini-instruct to the fly.io volume.

Run as the fly.io release command (set in fly.toml) so the model is cached
on the persistent volume before the first request hits the inference server.

Usage:
  python scripts/download_model.py
  python scripts/download_model.py --model_id microsoft/Phi-3.5-mini-instruct
"""

import argparse
import os

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models/cache")


def main():
    parser = argparse.ArgumentParser(description="Pre-download HuggingFace model to local cache")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Downloading {args.model_id} → {args.cache_dir}")
    print("This is ~7.6 GB and will only happen once (cached on the volume).")

    snapshot_download(
        repo_id=args.model_id,
        cache_dir=args.cache_dir,
        token=args.token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
