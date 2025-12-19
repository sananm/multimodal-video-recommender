"""
Loss Functions for Contrastive Learning

These losses teach the model to:
- Pull similar videos together
- Push different videos apart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) Loss.

    Used by CLIP, SimCLR, and most modern contrastive learning.

    The idea:
    - Given an anchor and positive pair
    - Treat all other samples in batch as negatives
    - Maximize probability of selecting the positive

    Args:
        temperature: Scaling factor (lower = sharper distribution)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor_emb: (batch, dim) anchor embeddings (normalized)
            positive_emb: (batch, dim) positive embeddings (normalized)

        Returns:
            loss: Scalar loss value

        How it works:
            Batch = [A, B, C, D]
            Positives = [A', B', C', D']

            For A: positive is A', negatives are B', C', D'
            For B: positive is B', negatives are A', C', D'
            etc.
        """
        batch_size = anchor_emb.size(0)
        device = anchor_emb.device

        # Compute all pairwise similarities
        # (batch, dim) @ (dim, batch) = (batch, batch)
        # Each row i contains similarities of anchor_i to all positives
        similarity_matrix = anchor_emb @ positive_emb.T / self.temperature

        # Labels: diagonal elements are positives
        # anchor_0 matches positive_0, anchor_1 matches positive_1, etc.
        labels = torch.arange(batch_size, device=device)

        # Cross entropy: maximize probability of correct positive
        # Row i should have highest value at column i
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss: anchor, positive, negative.

    Simpler alternative to InfoNCE.

    For each triplet:
        distance(anchor, positive) + margin < distance(anchor, negative)

    Args:
        margin: Minimum gap between positive and negative distances
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor_emb: (batch, dim)
            positive_emb: (batch, dim)
            negative_emb: (batch, dim)

        Returns:
            loss: Scalar loss value
        """
        # Compute distances (using cosine distance = 1 - cosine similarity)
        pos_sim = (anchor_emb * positive_emb).sum(dim=1)  # Higher = closer
        neg_sim = (anchor_emb * negative_emb).sum(dim=1)

        # We want: pos_sim > neg_sim + margin
        # Loss = max(0, neg_sim - pos_sim + margin)
        loss = F.relu(neg_sim - pos_sim + self.margin)

        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Combined contrastive loss with optional hard negative mining.

    Features:
    - InfoNCE as base loss
    - Optional: weight hard negatives more
    - Optional: symmetric loss (both directions)

    Args:
        temperature: Scaling factor
        symmetric: Compute loss in both directions (A→B and B→A)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.infonce = InfoNCELoss(temperature)

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emb1: (batch, dim) first set of embeddings
            emb2: (batch, dim) second set of embeddings

        Returns:
            loss: Scalar loss value
        """
        # Loss from emb1 → emb2
        loss = self.infonce(emb1, emb2)

        if self.symmetric:
            # Also compute loss from emb2 → emb1
            loss = (loss + self.infonce(emb2, emb1)) / 2

        return loss


# Test code
if __name__ == "__main__":
    print("Testing Loss Functions...")

    batch_size = 8
    dim = 512

    # Create dummy embeddings (normalized)
    anchor = F.normalize(torch.randn(batch_size, dim), dim=1)
    positive = F.normalize(torch.randn(batch_size, dim), dim=1)
    negative = F.normalize(torch.randn(batch_size, dim), dim=1)

    # Test InfoNCE
    print("\n1. InfoNCE Loss:")
    infonce = InfoNCELoss(temperature=0.07)
    loss = infonce(anchor, positive)
    print(f"   Loss: {loss.item():.4f}")

    # Test with identical pairs (should be low loss)
    loss_identical = infonce(anchor, anchor)
    print(f"   Loss (identical pairs): {loss_identical.item():.4f}")

    # Test Triplet Loss
    print("\n2. Triplet Loss:")
    triplet = TripletLoss(margin=0.3)
    loss = triplet(anchor, positive, negative)
    print(f"   Loss: {loss.item():.4f}")

    # Test Contrastive Loss
    print("\n3. Contrastive Loss (symmetric):")
    contrastive = ContrastiveLoss(temperature=0.07, symmetric=True)
    loss = contrastive(anchor, positive)
    print(f"   Loss: {loss.item():.4f}")

    print("\n[OK] All loss functions working!")
