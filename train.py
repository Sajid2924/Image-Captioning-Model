# train.py  —  Training Loop with Early Stopping
#
# Features:
#  - Mixed precision (float16) to save VRAM
#  - Learning rate warmup + cosine decay
#  - Early stopping — auto stops when val loss stops improving
#  - Checkpoint saving and resuming
#  - Saves: best_model.pt + last 2 epoch checkpoints only

import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from model import ImageCaptioningModel
from dataset import get_dataloaders
from config import cfg


# Early Stopping tracker
class EarlyStopping:
    """
    Stops training when val loss hasn't improved for `patience` epochs.

    min_delta: minimum improvement required to count as progress
               e.g. 0.001 means val loss must drop by at least 0.001
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0  # epochs without improvement
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Call after each epoch.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            print(f"  [EarlyStopping] Val loss improved → {val_loss:.4f}")
        else:
            # No improvement
            self.counter += 1
            print(
                f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  [EarlyStopping] Stopping training — patience exhausted")

        return self.should_stop


# Utility: save / load checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    print(
        f"  [Checkpoint] Resumed from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})"
    )
    return start_epoch


def delete_old_checkpoint(epoch, keep_last_n=2):
    old_epoch = epoch - keep_last_n
    if old_epoch >= 1:
        old_path = os.path.join(cfg.checkpoint_dir, f"epoch_{old_epoch:02d}.pt")
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"  [Cleanup] Deleted old checkpoint: epoch_{old_epoch:02d}.pt")


# One epoch of training
def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    start_time = time.time()

    for step, batch in enumerate(loader):
        images = batch["image"].to(cfg.device, non_blocking=True)
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels = batch["labels"].to(cfg.device, non_blocking=True)

        with autocast(enabled=cfg.mixed_precision):
            loss, _ = model(images, input_ids, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % cfg.log_every == 0:
            avg_loss = total_loss / (step + 1)
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:2d} | Step {step+1:4d}/{num_batches} "
                f"| Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed:.0f}s"
            )

    return total_loss / num_batches


# Validation loop
@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(cfg.device, non_blocking=True)
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels = batch["labels"].to(cfg.device, non_blocking=True)

        with autocast(enabled=cfg.mixed_precision):
            loss, _ = model(images, input_ids, labels)

        total_loss += loss.item()

    return total_loss / len(loader)


# Main training entry point
def train():
    print(f"\n{'='*55}")
    print(f"  Image Captioning — Training")
    print(f"  Device:           {cfg.device}")
    print(f"  Max epochs:       {cfg.num_epochs}")
    print(f"  Batch size:       {cfg.batch_size}")
    print(f"  Learning rate:    {cfg.learning_rate}")
    print(f"  Weight decay:     {cfg.weight_decay}")
    print(f"  Encoder frozen:   {cfg.encoder_frozen}")
    print(f"  Early stopping:   patience={cfg.early_stopping_patience}")
    print(f"  Mixed precision:  {cfg.mixed_precision}")
    print(f"{'='*55}\n")

    # Dataset
    print("[Main] Loading datasets...")
    train_loader, val_loader = get_dataloaders()

    # Model
    print("\n[Main] Building model...")
    model = ImageCaptioningModel().to(cfg.device)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR Scheduler
    total_steps = cfg.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # GradScaler
    scaler = GradScaler(enabled=cfg.mixed_precision)

    # Early Stopping
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
    )

    # Resume from checkpoint
    start_epoch = 1
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        start_epoch = load_checkpoint(model, optimizer, scheduler, cfg.resume_from)

    # Training Loop
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        epoch_start = time.time()
        print(f"\n{''*55}")
        print(f"  EPOCH {epoch}/{cfg.num_epochs}")
        print(f"{''*55}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch
        )

        # Validate
        val_loss = validate(model, val_loader)

        epoch_time = time.time() - epoch_start
        print(f"\n  ✓ Epoch {epoch} complete in {epoch_time/60:.1f} min")
        print(f"    Train loss: {train_loss:.4f}")
        print(f"    Val   loss: {val_loss:.4f}")

        # Save current epoch checkpoint
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:02d}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, ckpt_path)
        delete_old_checkpoint(epoch, keep_last_n=2)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
            print(f"    ★ New best model saved (val_loss={val_loss:.4f})")

        # Early stopping check
        if early_stopper.step(val_loss):
            print(f"\n  Training stopped at epoch {epoch} by early stopping.")
            print(f"  Best val loss was {best_val_loss:.4f}")
            break

    print(f"\n{'='*55}")
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Best model: {cfg.checkpoint_dir}/best_model.pt")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()
