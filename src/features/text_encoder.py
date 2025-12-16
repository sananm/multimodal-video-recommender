"""
TextEncoder: Extracts text features from video titles/descriptions using BERT

Key Concepts:
1. Text is tokenized into subwords (e.g., "playing" → "play" + "##ing")
2. BERT converts tokens into contextualized embeddings
3. Each token gets a 768-d vector that depends on surrounding context
4. We aggregate into a single text embedding

Same pattern as Video and Audio encoders!
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    """
    Encodes text (titles, descriptions) into fixed-size embeddings using BERT.

    Args:
        model_name: Which BERT model to use
        feature_dim: Final embedding size (512 to match video/audio)
        aggregation: How to combine token features ('cls', 'mean', 'attention')
        freeze_backbone: If True, don't train BERT (faster, less memory)
        max_length: Maximum number of tokens (longer text gets truncated)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        feature_dim: int = 512,
        aggregation: str = "cls",
        freeze_backbone: bool = True,
        max_length: int = 128
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.aggregation = aggregation
        self.max_length = max_length

        # Load pre-trained BERT
        # "uncased" means it lowercases everything ("Hello" → "hello")
        self.backbone = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # BERT-base outputs 768-dimensional features
        # BERT-large outputs 1024-dimensional features
        if "large" in model_name:
            backbone_dim = 1024
        else:
            backbone_dim = 768

        # Freeze backbone to save memory
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection: Map BERT features to our feature_dim
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

    def tokenize(self, texts: list[str]) -> dict:
        """
        Convert list of strings to token IDs.

        Args:
            texts: List of strings, e.g., ["How to cook pasta", "Funny cat video"]

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        return self.tokenizer(
            texts,
            padding=True,              # Pad shorter texts to match longest
            truncation=True,           # Cut off texts longer than max_length
            max_length=self.max_length,
            return_tensors="pt"        # Return PyTorch tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        texts: list[str] = None
    ) -> torch.Tensor:
        """
        Args:
            Option 1 - Pre-tokenized:
                input_ids: (batch_size, seq_len) token IDs
                attention_mask: (batch_size, seq_len) mask for padding

            Option 2 - Raw text:
                texts: List of strings to encode

        Returns:
            embeddings: (batch_size, feature_dim)
        """
        # If raw text provided, tokenize first
        if texts is not None:
            tokens = self.tokenize(texts)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            # Move to same device as model
            input_ids = input_ids.to(self.backbone.device)
            attention_mask = attention_mask.to(self.backbone.device)

        # Pass through BERT
        # Output shape: (batch_size, seq_len, 768)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get hidden states for all tokens
        # Shape: (batch_size, seq_len, 768)
        hidden_states = outputs.last_hidden_state

        # Aggregate tokens into single embedding
        if self.aggregation == "cls":
            # Use [CLS] token embedding (first token)
            # This is designed to represent the whole sentence
            # Shape: (batch_size, 768)
            features = hidden_states[:, 0, :]

        elif self.aggregation == "mean":
            # Average all token embeddings (excluding padding)
            # attention_mask: 1 for real tokens, 0 for padding
            # Shape: (batch_size, 768)
            mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            features = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

        elif self.aggregation == "attention":
            # First project, then apply attention
            projected = self.projection(hidden_states)  # (batch, seq_len, 512)
            attention_scores = self.attention(projected)  # (batch, seq_len, 1)

            # Mask padding tokens before softmax
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    attention_mask.unsqueeze(-1) == 0,
                    float("-inf")
                )

            attention_weights = torch.softmax(attention_scores, dim=1)
            text_emb = (projected * attention_weights).sum(dim=1)
            return text_emb

        # Project to final dimension
        # Shape: (batch_size, 512)
        text_emb = self.projection(features)

        return text_emb


# Test code to verify it works
if __name__ == "__main__":
    print("Testing TextEncoder...")
    print("(This will download BERT on first run, ~440MB)")

    encoder = TextEncoder(
        model_name="bert-base-uncased",
        feature_dim=512,
        aggregation="cls",
        freeze_backbone=True
    )

    # Test with some video titles/descriptions
    test_texts = [
        "How to make delicious pasta at home",
        "Funny cat compilation 2024",
        "Learn Python programming for beginners",
        "Amazing travel vlog from Japan"
    ]

    output = encoder(texts=test_texts)

    print(f"Input: {len(test_texts)} text strings")
    print(f"Output shape: {output.shape}")  # (4, 512)
    print("✓ TextEncoder working!")
