# sample_test.py  —  Tiny training loop (for CPU)
#
# Purpose:
#  Runs a mini training loop on just 50 images for 3 epochs
#  so you can verify the full pipeline works and get a
#  sample best_model.pt WITHOUT needing a GPU
#
# Expected time: ~5-10 minutes on CPU
# Run: python sample_test.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup

from model import ImageCaptioningModel
from dataset import Flickr8kDataset
from config import cfg

# Override settings for tiny test run
DEVICE = "cpu"  # force CPU — no GPU needed
NUM_IMAGES = 50  # only 50 images
BATCH_SIZE = 4  # small batch for CPU
NUM_EPOCHS = 3  # 3 quick epochs
LOG_EVERY = 5  # print every 5 steps
SAVE_DIR = "./checkpoints"
SAVE_PATH = os.path.join(SAVE_DIR, "sample_best_model.pt")


def save_checkpoint(model, epoch, loss):
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "loss": loss,
        },
        SAVE_PATH,
    )
    print(f"  [Saved] → {SAVE_PATH}")


def run_sample_training():
    print("\n" + "=" * 55)
    print("  SAMPLE TEST — Tiny Training Loop (CPU)")
    print(f"  Images:  {NUM_IMAGES}")
    print(f"  Epochs:  {NUM_EPOCHS}")
    print(f"  Device:  {DEVICE}")
    print("=" * 55 + "\n")

    # Load tiny subset of dataset
    print("[1/4] Loading dataset subset...")
    full_dataset = Flickr8kDataset(cfg.data_dir, cfg.captions_file, split="train")
    small_dataset = Subset(full_dataset, list(range(NUM_IMAGES)))
    loader = DataLoader(small_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"      Using {NUM_IMAGES} image-caption pairs\n")

    # Build model on CPU
    print("[2/4] Loading model...")
    model = ImageCaptioningModel().to(DEVICE)
    model.train()
    print()

    # Optimizer
    print("[3/4] Setting up optimizer...")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)
    total_steps = NUM_EPOCHS * len(loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=total_steps
    )
    print(f"      Trainable params: {sum(p.numel() for p in trainable):,}\n")

    # Training loop
    print("[4/4] Training...\n")
    best_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"  {''*45}")
        print(f"  EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"  {''*45}")

        epoch_loss = 0.0
        num_steps = len(loader)

        for step, batch in enumerate(loader):
            # Move to CPU (already there, but keeps code consistent)
            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            loss, _ = model(images, input_ids, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (step + 1) % LOG_EVERY == 0 or (step + 1) == num_steps:
                avg = epoch_loss / (step + 1)
                print(
                    f"    Step {step+1:3d}/{num_steps} | Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_epoch_loss = epoch_loss / num_steps
        print(f"\n  Epoch {epoch} avg loss: {avg_epoch_loss:.4f}")

        # Save best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, epoch, best_loss)
            print(f"  ★ Best model updated!\n")

    print("\n" + "=" * 55)
    print(f"  Training complete!")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Saved to:   {SAVE_PATH}")
    print("=" * 55)
    print("\n  Now run:  python predict.py --image your_image.jpg")
    print(f"            --checkpoint {SAVE_PATH}\n")


if __name__ == "__main__":
    run_sample_training()
