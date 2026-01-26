"""
Model Download Script for Hitori Training.

Downloads the base model and tokenizer to a local directory for faster loading
during training and inference.

Usage:
    python download_model.py
    python download_model.py --model Qwen/Qwen2.5-3B-Instruct --output models/qwen2.5-3b
"""

import argparse
import os

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUTPUT_DIR = "models/qwen2.5-3b-instruct"


def download_model(
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    revision: str = "main",
    token: str = None,
):
    """
    Download model and tokenizer from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Local directory to save model
        revision: Model revision/branch
        token: HuggingFace token (optional, for gated models)
    """
    print("=" * 60)
    print("Hitori Model Download")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Revision: {revision}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    print("\nDownloading model files...")
    snapshot_download(
        repo_id=model_name,
        local_dir=output_dir,
        revision=revision,
        token=token,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    print(f"\nModel downloaded to: {output_dir}")
    print("\nTo use this model for training:")
    print(f"  python train_hitori_grpo.py --model_name_or_path {output_dir}")
    print("\nTo use for inference:")
    print(f"  python test_rollout.py --model-path {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download model for Hitori training")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--revision", type=str, default="main",
        help="Model revision (default: main)"
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token for gated models"
    )

    args = parser.parse_args()

    download_model(
        model_name=args.model,
        output_dir=args.output,
        revision=args.revision,
        token=args.token,
    )


if __name__ == "__main__":
    main()
