"""
VideoDataset: PyTorch Dataset for loading multimodal video data

This connects video files to our encoders by:
1. Loading video paths and metadata from a CSV/JSON
2. Extracting frames and audio on-the-fly
3. Returning batches ready for the model
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from processors import VideoProcessor, AudioProcessor


class VideoDataset(Dataset):
    """
    Dataset for multimodal video recommendation.

    Expected data format (JSON):
    [
        {
            "video_id": "abc123",
            "video_path": "data/raw/videos/abc123.mp4",
            "title": "How to make pasta",
            "description": "Easy recipe for beginners",
            "tags": ["cooking", "food"]
        },
        ...
    ]

    Args:
        data_path: Path to JSON file with video metadata
        video_dir: Base directory for video files (if paths are relative)
        num_frames: Frames to extract per video
        frame_size: Resize frames to (H, W)
        audio_duration: Seconds of audio to extract
        sample_rate: Audio sample rate
    """

    def __init__(
        self,
        data_path: str,
        video_dir: str = None,
        num_frames: int = 8,
        frame_size: tuple[int, int] = (224, 224),
        audio_duration: float = 5.0,
        sample_rate: int = 16000,
        load_video: bool = True,
        load_audio: bool = True
    ):
        self.data_path = Path(data_path)
        self.video_dir = Path(video_dir) if video_dir else None
        self.load_video = load_video
        self.load_audio = load_audio

        # Load metadata
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} videos from {data_path}")

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
        """Number of videos in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Load one video's multimodal data.

        Args:
            idx: Index of video to load

        Returns:
            Dictionary with:
                - video_id: Unique identifier
                - frames: (num_frames, 3, H, W) tensor
                - audio: (num_samples,) tensor
                - title: String
                - description: String
                - tags: List of strings
        """
        item = self.data[idx]

        # Get video path
        video_path = item.get("video_path", "")
        if self.video_dir and not Path(video_path).is_absolute():
            video_path = self.video_dir / video_path

        # Build result dictionary
        result = {
            "video_id": item.get("video_id", str(idx)),
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "tags": item.get("tags", []),
        }

        # Extract frames
        if self.load_video and video_path and Path(video_path).exists():
            try:
                result["frames"] = self.video_processor.extract_frames(video_path)
            except Exception as e:
                print(f"Warning: Failed to load video {video_path}: {e}")
                result["frames"] = torch.zeros(
                    self.video_processor.num_frames, 3,
                    self.video_processor.frame_size[0],
                    self.video_processor.frame_size[1]
                )
        else:
            # Return zero tensor if no video
            result["frames"] = torch.zeros(
                self.video_processor.num_frames, 3,
                self.video_processor.frame_size[0],
                self.video_processor.frame_size[1]
            )

        # Extract audio
        if self.load_audio and video_path and Path(video_path).exists():
            try:
                result["audio"] = self.audio_processor.extract_audio(video_path)
            except Exception as e:
                print(f"Warning: Failed to load audio {video_path}: {e}")
                result["audio"] = torch.zeros(self.audio_processor.target_length)
        else:
            result["audio"] = torch.zeros(self.audio_processor.target_length)

        return result


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function to handle variable-length text.

    DataLoader calls this to combine individual items into a batch.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Batched dictionary with stacked tensors
    """
    return {
        "video_id": [item["video_id"] for item in batch],
        "frames": torch.stack([item["frames"] for item in batch]),
        "audio": torch.stack([item["audio"] for item in batch]),
        "title": [item["title"] for item in batch],
        "description": [item["description"] for item in batch],
        "tags": [item["tags"] for item in batch],
    }


def create_dataloader(
    data_path: str,
    video_dir: str = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    Convenience function to create a DataLoader.

    Args:
        data_path: Path to JSON metadata file
        video_dir: Base directory for videos
        batch_size: Videos per batch
        shuffle: Randomize order
        num_workers: Parallel data loading processes
        **dataset_kwargs: Additional args for VideoDataset

    Returns:
        PyTorch DataLoader ready for training
    """
    dataset = VideoDataset(
        data_path=data_path,
        video_dir=video_dir,
        **dataset_kwargs
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True  # Faster GPU transfer
    )

    return loader


# Test code
if __name__ == "__main__":
    print("Testing VideoDataset...")

    # Create sample metadata file for testing
    sample_data = [
        {
            "video_id": "test_001",
            "video_path": "videos/test1.mp4",
            "title": "How to cook pasta",
            "description": "Easy pasta recipe for beginners",
            "tags": ["cooking", "food", "tutorial"]
        },
        {
            "video_id": "test_002",
            "video_path": "videos/test2.mp4",
            "title": "Funny cat compilation",
            "description": "Hilarious cat moments",
            "tags": ["cats", "funny", "animals"]
        }
    ]

    # Save sample data
    sample_path = Path("data/sample_metadata.json")
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    with open(sample_path, 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"Created sample metadata at {sample_path}")

    # Test dataset (without actual video files)
    dataset = VideoDataset(
        data_path=str(sample_path),
        load_video=False,  # Skip video loading for test
        load_audio=False   # Skip audio loading for test
    )

    print(f"Dataset length: {len(dataset)}")

    # Get one item
    item = dataset[0]
    print(f"\nSample item:")
    print(f"  video_id: {item['video_id']}")
    print(f"  title: {item['title']}")
    print(f"  frames shape: {item['frames'].shape}")
    print(f"  audio shape: {item['audio'].shape}")

    print("\n✓ VideoDataset working!")
