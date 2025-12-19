"""
Training Loop for Two-Tower Model

This handles:
- Training epochs
- Validation
- Checkpointing
- Logging to TensorBoard
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from .losses import ContrastiveLoss

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.device import get_device


class Trainer:
    """
    Trainer for the Two-Tower recommendation model.

    Args:
        model: TwoTowerModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        num_epochs: Number of training epochs
        checkpoint_dir: Where to save model checkpoints
        log_dir: Where to save TensorBoard logs
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: str = None
    ):
        # Set device (supports CUDA, MPS, and CPU)
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Model
        self.model = model.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training params
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Loss function
        self.criterion = ContrastiveLoss(temperature=0.07, symmetric=True)

        # Optimizer (AdamW is standard for transformers)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (cosine decay)
        total_steps = len(train_loader) * num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate / 100
        )

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
            print("TensorBoard not available. Install with: pip install tensorboard")

        # Track best model
        self.best_val_loss = float('inf')
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Returns:
            average_loss: Mean loss over the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            # Move data to device
            frames = batch["frames"].to(self.device)
            audio = batch["audio"].to(self.device)
            titles = batch["title"]  # List of strings

            # For contrastive learning, we need pairs
            # Simple approach: use augmentation or split batch
            # Here we use in-batch negatives (each video is negative for others)

            # Encode all videos
            embeddings = self.model.encode_video(frames, audio, titles)

            # Create positive pairs by adding noise (simple augmentation)
            # In practice, you'd use proper augmentation or paired data
            noise = torch.randn_like(embeddings) * 0.1
            positive_embeddings = embeddings + noise
            positive_embeddings = nn.functional.normalize(positive_embeddings, dim=1)

            # Compute loss
            loss = self.criterion(embeddings, positive_embeddings)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            # Log to TensorBoard
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation.

        Returns:
            average_loss: Mean loss over validation set
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            frames = batch["frames"].to(self.device)
            audio = batch["audio"].to(self.device)
            titles = batch["title"]

            embeddings = self.model.encode_video(frames, audio, titles)

            # Same augmentation as training for fair comparison
            noise = torch.randn_like(embeddings) * 0.1
            positive_embeddings = embeddings + noise
            positive_embeddings = nn.functional.normalize(positive_embeddings, dim=1)

            loss = self.criterion(embeddings, positive_embeddings)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, self.global_step)

        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step
        }

        # Save latest
        path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, path)
            print(f"  New best model saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            path = self.checkpoint_dir / f'epoch_{epoch+1}.pt'
            torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']

        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")

        return checkpoint['epoch']

    def train(self):
        """
        Full training loop.

        Returns:
            best_val_loss: Best validation loss achieved
        """
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print()

        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)

            # Log epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.num_epochs} complete in {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            if self.val_loader:
                print(f"  Val Loss: {val_loss:.4f}")
            print()

        total_time = time.time() - start_time
        print(f"Training complete in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        if self.writer:
            self.writer.close()

        return self.best_val_loss


# Convenience function
def train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=10,
    learning_rate=1e-4,
    checkpoint_dir="checkpoints",
    log_dir="logs"
):
    """
    Train a model with default settings.

    Example:
        model = TwoTowerModel()
        train_loader = create_dataloader("data/train.json", batch_size=32)
        val_loader = create_dataloader("data/val.json", batch_size=32)

        train_model(model, train_loader, val_loader, num_epochs=10)
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )

    return trainer.train()
