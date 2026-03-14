# =============================================================
#  train.py  —  Training Loop
#
#  Features:
#    - Mixed precision (float16) to save VRAM
#    - Gradient accumulation (simulate large batch sizes)
#    - Learning rate warmup + cosine decay
#    - Checkpoint saving and resuming
#    - Train + validation loss logging
# =============================================================

import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from model   import ImageCaptioningModel
from dataset import get_dataloaders
from config  import cfg


# ─────────────────────────────────────────────────────────────
#  Utility: save / load checkpoint
# ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"     : epoch,
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "scheduler" : scheduler.state_dict(),
        "loss"      : loss,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    print(f"  [Checkpoint] Resumed from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return start_epoch


# ─────────────────────────────────────────────────────────────
#  One epoch of training
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss  = 0.0
    num_batches = len(loader)
    start_time  = time.time()

    for step, batch in enumerate(loader):
        # ── Move batch to GPU ──────────────────────────────────
        images     = batch["image"].to(cfg.device, non_blocking=True)
        input_ids  = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels     = batch["labels"].to(cfg.device, non_blocking=True)

        # ── Mixed precision forward pass ──────────────────────
        # autocast: runs forward pass in float16 → less VRAM, faster
        with autocast(enabled=cfg.mixed_precision):
            loss, _ = model(images, input_ids, labels)

        # ── Backward pass ─────────────────────────────────────
        # GradScaler handles the float16 → float32 gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping: prevents exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Optimizer + scheduler step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()

        # ── Logging ───────────────────────────────────────────
        if (step + 1) % cfg.log_every == 0:
            avg_loss = total_loss / (step + 1)
            elapsed  = time.time() - start_time
            lr       = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:2d} | Step {step+1:4d}/{num_batches} "
                  f"| Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed:.0f}s")

    return total_loss / num_batches


# ─────────────────────────────────────────────────────────────
#  Validation loop
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images    = batch["image"].to(cfg.device, non_blocking=True)
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels    = batch["labels"].to(cfg.device, non_blocking=True)

        with autocast(enabled=cfg.mixed_precision):
            loss, _ = model(images, input_ids, labels)

        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────
#  Main training entry point
# ─────────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*50}")
    print(f"  Image Captioning — Training")
    print(f"  Device: {cfg.device}")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Batch:  {cfg.batch_size}")
    print(f"  Mixed precision: {cfg.mixed_precision}")
    print(f"{'='*50}\n")

    # ── Dataset ───────────────────────────────────────────────
    print("[Main] Loading datasets...")
    train_loader, val_loader = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────
    print("\n[Main] Building model...")
    model = ImageCaptioningModel().to(cfg.device)

    # ── Optimizer ─────────────────────────────────────────────
    # Only optimize trainable parameters (projection + GPT-2)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.95),    # GPT-style betas
    )

    # ── LR Scheduler: warmup → cosine decay ───────────────────
    total_steps = cfg.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = cfg.warmup_steps,
        num_training_steps = total_steps,
    )

    # ── GradScaler for mixed precision ────────────────────────
    scaler = GradScaler(enabled=cfg.mixed_precision)

    # ── Resume from checkpoint if set ─────────────────────────
    start_epoch = 1
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        start_epoch = load_checkpoint(model, optimizer, scheduler, cfg.resume_from)

    # ── Training Loop ─────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        epoch_start = time.time()
        print(f"\n{'─'*50}")
        print(f"  EPOCH {epoch}/{cfg.num_epochs}")
        print(f"{'─'*50}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)

        # Validate
        val_loss = validate(model, val_loader)

        epoch_time = time.time() - epoch_start
        print(f"\n  ✓ Epoch {epoch} complete in {epoch_time/60:.1f} min")
        print(f"    Train loss: {train_loss:.4f}")
        print(f"    Val   loss: {val_loss:.4f}")

        # Save checkpoint every N epochs
        if epoch % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:02d}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, ckpt_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
            print(f"    ★ New best model saved (val_loss={val_loss:.4f})")

    print(f"\n{'='*50}")
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Best model saved to: {cfg.checkpoint_dir}/best_model.pt")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    train()
