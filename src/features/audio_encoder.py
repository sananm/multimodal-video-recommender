"""
AudioEncoder: Extracts audio features from waveforms using Wav2Vec2

Key Concepts:
1. Audio is a 1D signal (amplitude over time)
2. Wav2Vec2 was pre-trained on 1000s of hours of speech
3. It outputs a sequence of feature vectors (one per ~20ms)
4. We aggregate these into a single audio embedding

This mirrors what VideoEncoder does, but for audio instead of frames.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class AudioEncoder(nn.Module):
    """
    Encodes audio waveforms into fixed-size embeddings using Wav2Vec2.

    Args:
        model_name: Which Wav2Vec2 model to use (base or large)
        feature_dim: Final embedding size (512 to match video)
        aggregation: How to combine time steps ('mean' or 'attention')
        freeze_backbone: If True, don't train Wav2Vec2 (faster, less memory)
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        feature_dim: int = 512,
        aggregation: str = "mean",
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.aggregation = aggregation

        # Load pre-trained Wav2Vec2
        # This model was trained on 960 hours of speech
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)

        # Wav2Vec2-base outputs 768-dimensional features
        # Wav2Vec2-large outputs 1024-dimensional features
        if "large" in model_name:
            backbone_dim = 1024
        else:
            backbone_dim = 768

        # Freeze backbone to save memory and speed up training
        # The pre-trained features are already good!
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection: Map Wav2Vec2 features to our feature_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, feature_dim)
        )

        # Optional: Attention mechanism for weighted aggregation
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveforms: (batch_size, num_samples)
                      e.g., (8, 16000) = 8 audio clips, 1 second each at 16kHz
            attention_mask: (batch_size, num_samples) optional
                           Masks padded regions for variable-length audio

        Returns:
            embeddings: (batch_size, feature_dim)
        """
        # Pass through Wav2Vec2
        # Output shape: (batch_size, sequence_length, 768)
        # sequence_length ≈ num_samples / 320 (depends on model)
        outputs = self.backbone(
            waveforms,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get the hidden states (feature sequence)
        # Shape: (batch_size, seq_len, 768)
        features = outputs.last_hidden_state

        # Project: (batch_size, seq_len, 768) -> (batch_size, seq_len, 512)
        features = self.projection(features)

        # Aggregate across time into single embedding
        if self.aggregation == "mean":
            # Simple average across time dimension
            # Shape: (batch_size, 512)
            audio_emb = features.mean(dim=1)

        elif self.aggregation == "attention":
            # Learn which time steps are important
            # attention_scores: (batch_size, seq_len, 1)
            attention_scores = self.attention(features)
            attention_weights = torch.softmax(attention_scores, dim=1)
            # Weighted sum: (batch_size, 512)
            audio_emb = (features * attention_weights).sum(dim=1)

        return audio_emb


# Test code to verify it works
if __name__ == "__main__":
    print("Testing AudioEncoder...")
    print("(This will download Wav2Vec2 on first run, ~360MB)")

    encoder = AudioEncoder(
        model_name="facebook/wav2vec2-base-960h",
        feature_dim=512,
        aggregation="mean",
        freeze_backbone=True
    )

    # Simulate 4 audio clips, each 1 second at 16kHz
    # In real use, these would be actual audio samples
    dummy_audio = torch.randn(4, 16000)

    output = encoder(dummy_audio)

    print(f"Input shape: {dummy_audio.shape}")   # (4, 16000)
    print(f"Output shape: {output.shape}")       # (4, 512)
    print("[OK] AudioEncoder working!")
