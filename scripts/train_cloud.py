#!/usr/bin/env python3
"""
Cloud Training Script for Multimodal Video Recommender

Run on GCP:
    python3 scripts/train_cloud.py --epochs 10 --batch_size 8

This script:
1. Downloads MicroLens dataset (if not present)
2. Trains the Two-Tower model with contrastive learning
3. Saves checkpoints to ./checkpoints/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_microlens(data_dir: Path, max_videos: int = 1000):
    """Download MicroLens dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset"

    # Download metadata
    metadata_files = [
        "MicroLens-50k_titles.csv",
        "MicroLens-50k_pairs.csv"
    ]

    for f in metadata_files:
        path = data_dir / f
        if not path.exists():
            print(f"Downloading {f}...")
            subprocess.run(["curl", "-o", str(path), f"{base_url}/{f}"], check=True)

    # Download videos
    video_dir = data_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    import pandas as pd
    titles = pd.read_csv(data_dir / "MicroLens-50k_titles.csv")

    downloaded = len(list(video_dir.glob("*.mp4")))
    to_download = min(max_videos, len(titles)) - downloaded

    if to_download > 0:
        print(f"Downloading {to_download} videos...")
        for _, row in titles.head(max_videos).iterrows():
            video_id = row['item']
            video_path = video_dir / f"{video_id}.mp4"
            if not video_path.exists():
                url = f"{base_url}/MicroLens-50k_videos/{video_id}.mp4"
                try:
                    subprocess.run(
                        ["curl", "-s", "-o", str(video_path), url],
                        check=True,
                        timeout=60
                    )
                    print(f"  Downloaded {video_id}.mp4")
                except:
                    print(f"  Failed: {video_id}.mp4")

    print(f"Dataset ready: {len(list(video_dir.glob('*.mp4')))} videos")


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Video Recommender")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_videos", type=int, default=500, help="Max videos to use")
    parser.add_argument("--download", action="store_true", help="Download dataset first")
    parser.add_argument("--data_dir", type=str, default="data/raw/microlens")
    args = parser.parse_args()

    # Setup
    import torch
    from src.utils.device import get_device

    device = get_device()
    print(f"\n{'='*50}")
    print("MULTIMODAL VIDEO RECOMMENDER - CLOUD TRAINING")
    print(f"{'='*50}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max videos: {args.max_videos}")
    print(f"{'='*50}\n")

    data_dir = Path(args.data_dir)

    # Download dataset if requested
    if args.download:
        download_microlens(data_dir, args.max_videos)

    # Check dataset exists
    if not (data_dir / "MicroLens-50k_titles.csv").exists():
        print("Dataset not found! Run with --download flag first.")
        sys.exit(1)

    # Create data loader
    from src.data.microlens_dataset import create_microlens_dataloader

    print("Loading dataset...")
    train_loader = create_microlens_dataloader(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        max_videos=args.max_videos,
        num_frames=8,
        audio_duration=5.0,
        shuffle=True,
        num_workers=2  # Parallel data loading
    )

    print(f"Dataset size: {len(train_loader.dataset)} videos")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create model
    from src.models.two_tower import TwoTowerModel

    print("\nInitializing model...")
    model = TwoTowerModel(
        feature_dim=512,
        share_towers=True,
        freeze_encoders=True,  # Only train fusion layers
        pretrained=False  # Set True if SSL works on GCP
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    from src.training.trainer import Trainer

    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # Could split dataset for validation
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir="checkpoints",
        log_dir="logs"
    )

    best_loss = trainer.train()

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: checkpoints/best.pt")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
