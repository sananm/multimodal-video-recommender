"""
MicroLens Dataset Loader

Loads data from the MicroLens-50K dataset with real:
- Video files (.mp4)
- Titles from CSV
- Audio extracted from videos

Dataset structure:
    data/raw/microlens/
    ├── MicroLens-50k_titles.csv      (item, title)
    ├── MicroLens-50k_pairs.csv       (user, item, timestamp)
    └── videos/
        ├── 1.mp4
        ├── 2.mp4
        └── ...
"""

import pandas as pd
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .processors import VideoProcessor, AudioProcessor


class MicroLensDataset(Dataset):
    """
    Dataset for MicroLens video recommendation data.

    Args:
        data_dir: Path to microlens data directory
        video_dir: Path to video files (default: data_dir/videos)
        num_frames: Frames to extract per video
        frame_size: Resize frames to (H, W)
        audio_duration: Seconds of audio to extract
        sample_rate: Audio sample rate
        max_videos: Limit number of videos (for testing)
    """

    def __init__(
        self,
        data_dir: str = "data/raw/microlens",
        video_dir: str = None,
        num_frames: int = 8,
        frame_size: tuple[int, int] = (224, 224),
        audio_duration: float = 5.0,
        sample_rate: int = 16000,
        max_videos: int = None
    ):
        self.data_dir = Path(data_dir)
        self.video_dir = Path(video_dir) if video_dir else self.data_dir / "videos"

        # Load titles
        titles_path = self.data_dir / "MicroLens-50k_titles.csv"
        if not titles_path.exists():
            raise FileNotFoundError(f"Titles file not found: {titles_path}")

        self.titles_df = pd.read_csv(titles_path)
        print(f"Loaded {len(self.titles_df)} video titles")

        # Filter to only videos we have downloaded
        available_videos = set(
            int(p.stem) for p in self.video_dir.glob("*.mp4")
        )
        print(f"Found {len(available_videos)} video files")

        # Filter dataset to available videos
        self.titles_df = self.titles_df[
            self.titles_df['item'].isin(available_videos)
        ].reset_index(drop=True)

        print(f"Using {len(self.titles_df)} videos with available files")

        # Limit for testing
        if max_videos and len(self.titles_df) > max_videos:
            self.titles_df = self.titles_df.head(max_videos)
            print(f"Limited to {max_videos} videos for testing")

        # Initialize processors
        self.video_processor = VideoProcessor(
            num_frames=num_frames,
            frame_size=frame_size
        )

        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            duration=audio_duration
        )

    def __len__(self) -> int:
        return len(self.titles_df)

    def __getitem__(self, idx: int) -> dict:
        """Load one video's multimodal data."""
        row = self.titles_df.iloc[idx]

        video_id = str(row['item'])
        title = row['title']
        video_path = self.video_dir / f"{video_id}.mp4"

        result = {
            "video_id": video_id,
            "title": title,
        }

        # Extract frames from video
        try:
            result["frames"] = self.video_processor.extract_frames(video_path)
        except Exception as e:
            print(f"Warning: Failed to load video {video_path}: {e}")
            result["frames"] = torch.zeros(
                self.video_processor.num_frames, 3,
                self.video_processor.frame_size[0],
                self.video_processor.frame_size[1]
            )

        # Extract audio from video
        try:
            result["audio"] = self.audio_processor.extract_audio(video_path)
        except Exception as e:
            print(f"Warning: Failed to load audio {video_path}: {e}")
            result["audio"] = torch.zeros(self.audio_processor.target_length)

        return result


def microlens_collate_fn(batch: list[dict]) -> dict:
    """Collate function for MicroLens dataset."""
    return {
        "video_id": [item["video_id"] for item in batch],
        "frames": torch.stack([item["frames"] for item in batch]),
        "audio": torch.stack([item["audio"] for item in batch]),
        "title": [item["title"] for item in batch],
    }


def create_microlens_dataloader(
    data_dir: str = "data/raw/microlens",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    max_videos: int = None,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for MicroLens dataset.

    Args:
        data_dir: Path to microlens data
        batch_size: Videos per batch
        shuffle: Randomize order
        num_workers: Parallel data loading
        max_videos: Limit videos for testing
        **dataset_kwargs: Additional args for MicroLensDataset

    Returns:
        PyTorch DataLoader
    """
    dataset = MicroLensDataset(
        data_dir=data_dir,
        max_videos=max_videos,
        **dataset_kwargs
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=microlens_collate_fn,
        pin_memory=True
    )

    return loader


# Test code
if __name__ == "__main__":
    print("Testing MicroLens Dataset...")

    # Test dataset
    dataset = MicroLensDataset(
        data_dir="data/raw/microlens",
        max_videos=5
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load one sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  video_id: {sample['video_id']}")
    print(f"  title: {sample['title'][:50]}...")
    print(f"  frames shape: {sample['frames'].shape}")
    print(f"  audio shape: {sample['audio'].shape}")

    # Test dataloader
    print("\nTesting DataLoader...")
    loader = create_microlens_dataloader(
        data_dir="data/raw/microlens",
        batch_size=2,
        max_videos=4
    )

    for batch in loader:
        print(f"\nBatch:")
        print(f"  video_ids: {batch['video_id']}")
        print(f"  frames: {batch['frames'].shape}")
        print(f"  audio: {batch['audio'].shape}")
        print(f"  titles: {[t[:30] + '...' for t in batch['title']]}")
        break

    print("\n[OK] MicroLens Dataset working!")
