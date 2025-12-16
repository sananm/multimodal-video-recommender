"""
Media Processors: Extract frames and audio from video files

These utilities handle the raw video → tensor conversion needed for our encoders.
"""

import torch
import numpy as np
from pathlib import Path

# Video processing
import cv2

# Audio processing
import librosa


class VideoProcessor:
    """
    Extracts frames from video files.

    Args:
        num_frames: How many frames to sample from each video
        frame_size: Resize frames to (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
    """

    def __init__(
        self,
        num_frames: int = 8,
        frame_size: tuple[int, int] = (224, 224),
        normalize: bool = True
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.normalize = normalize

    def extract_frames(self, video_path: str) -> torch.Tensor:
        """
        Extract evenly-spaced frames from a video.

        Args:
            video_path: Path to video file (mp4, avi, etc.)

        Returns:
            frames: (num_frames, 3, H, W) tensor
        """
        video_path = str(video_path)

        # Open video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")

        # Calculate which frames to sample (evenly spaced)
        # Example: 100 frames, want 8 → indices [0, 14, 28, 42, 57, 71, 85, 99]
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Video shorter than num_frames: take all and pad
            frame_indices = np.arange(total_frames)

        frames = []

        for idx in frame_indices:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # Read frame
            ret, frame = cap.read()

            if not ret:
                # If frame read fails, use black frame
                frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)

            # BGR → RGB (OpenCV loads as BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to target size
            frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))

            frames.append(frame)

        cap.release()

        # Pad if video was too short
        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())  # Repeat last frame

        # Stack into numpy array: (num_frames, H, W, 3)
        frames = np.stack(frames)

        # Convert to tensor and rearrange: (num_frames, 3, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        # Normalize to [0, 1]
        if self.normalize:
            frames = frames / 255.0

        return frames


class AudioProcessor:
    """
    Extracts audio waveform from video files.

    Args:
        sample_rate: Target sample rate (16000 for Wav2Vec2)
        duration: How many seconds of audio to extract
        mono: Convert to mono (single channel)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 5.0,
        mono: bool = True
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
        self.target_length = int(sample_rate * duration)

    def extract_audio(self, video_path: str) -> torch.Tensor:
        """
        Extract audio waveform from a video file.

        Args:
            video_path: Path to video file

        Returns:
            waveform: (num_samples,) tensor, e.g., (80000,) for 5 seconds
        """
        video_path = str(video_path)

        try:
            # librosa can extract audio from video files
            # sr=None loads at original sample rate, then we resample
            waveform, orig_sr = librosa.load(
                video_path,
                sr=self.sample_rate,  # Resample to target rate
                mono=self.mono,
                duration=self.duration  # Only load first N seconds
            )
        except Exception as e:
            # If audio extraction fails, return silence
            print(f"Warning: Could not extract audio from {video_path}: {e}")
            waveform = np.zeros(self.target_length, dtype=np.float32)

        # Ensure correct length (pad or truncate)
        if len(waveform) < self.target_length:
            # Pad with zeros (silence)
            padding = self.target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')
        elif len(waveform) > self.target_length:
            # Truncate
            waveform = waveform[:self.target_length]

        # Convert to tensor
        waveform = torch.from_numpy(waveform).float()

        return waveform


# Test code
if __name__ == "__main__":
    print("Testing processors...")

    video_proc = VideoProcessor(num_frames=8, frame_size=(224, 224))
    audio_proc = AudioProcessor(sample_rate=16000, duration=5.0)

    print(f"VideoProcessor: extracts {video_proc.num_frames} frames at {video_proc.frame_size}")
    print(f"AudioProcessor: extracts {audio_proc.duration}s at {audio_proc.sample_rate}Hz")

    # To test with a real video:
    # frames = video_proc.extract_frames("path/to/video.mp4")
    # print(f"Frames shape: {frames.shape}")  # (8, 3, 224, 224)
    #
    # audio = audio_proc.extract_audio("path/to/video.mp4")
    # print(f"Audio shape: {audio.shape}")  # (80000,)

    print("✓ Processors ready!")
