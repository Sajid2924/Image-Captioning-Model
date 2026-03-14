# =============================================================
#  dataset.py  —  Flickr8k Dataset Loader
#
#  Flickr8k structure:
#    data/
#      Images/          ← all .jpg images
#      captions.txt     ← format: "image.jpg,caption text here"
#
#  What this file does:
#    1. Parses captions.txt to get (image_path, caption) pairs
#    2. Tokenizes captions with GPT-2 tokenizer
#    3. Returns tensors ready for the model
# =============================================================

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer

from encoder import get_image_transform
from config import cfg


# ─────────────────────────────────────────────────────────────
#  Dataset class
# ─────────────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    """
    Each item returns:
        image    : (3, 224, 224) normalized image tensor
        input_ids: (max_caption_len,) token IDs for the caption
        labels   : (max_caption_len,) same as input_ids but with
                   padding positions set to -100 (ignored by loss)
    """

    def __init__(self, data_dir: str, captions_file: str, split: str = "train"):
        self.image_dir = os.path.join(data_dir, "Images")
        self.transform = get_image_transform()

        # ── Load GPT-2 tokenizer ───────────────────────────────
        # The tokenizer converts text to token IDs
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # GPT-2 has no pad token by default — use EOS token as pad
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id  = self.tokenizer.eos_token_id
        self.max_len = cfg.max_caption_len

        # ── Parse captions file ───────────────────────────────
        print(f"[Dataset] Loading captions from {captions_file}")
        pairs = self._load_captions(captions_file)

        # ── Train / Val split ─────────────────────────────────
        split_idx = int(len(pairs) * cfg.train_split)
        if split == "train":
            self.pairs = pairs[:split_idx]
        else:
            self.pairs = pairs[split_idx:]

        print(f"[Dataset] {split} set: {len(self.pairs)} image-caption pairs")

    def _load_captions(self, captions_file: str):
        """
        Parses captions.txt
        Expected format (Flickr8k):
            image1.jpg,A dog runs across a field.
            image1.jpg,A brown dog plays outside.
            ...
        Returns list of (image_filename, caption_text) tuples
        """
        pairs = []
        with open(captions_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line_num == 0:   # skip empty lines / header
                    continue
                # Split only on first comma
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                image_name, caption = parts
                image_path = os.path.join(self.image_dir, image_name.strip())
                if os.path.exists(image_path):
                    pairs.append((image_path, caption.strip()))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, caption_text = self.pairs[idx]

        # ── Load and transform image ───────────────────────────
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)    # (3, 224, 224)

        # ── Tokenize caption ──────────────────────────────────
        # Add EOS at the end so the model learns when to stop
        caption_with_eos = caption_text + self.tokenizer.eos_token

        encoding = self.tokenizer(
            caption_with_eos,
            max_length    = self.max_len,
            padding       = "max_length",   # pad shorter sequences
            truncation    = True,           # truncate longer sequences
            return_tensors = "pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)        # (max_len,)
        attn_mask = encoding["attention_mask"].squeeze(0)   # (max_len,) 1=real, 0=pad

        # ── Labels for cross-entropy loss ─────────────────────
        # Same as input_ids but padding positions → -100
        # PyTorch's CrossEntropyLoss ignores positions with label -100
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100    # mask padding from loss

        return {
            "image"     : image,        # (3, 224, 224)
            "input_ids" : input_ids,    # (max_len,)
            "labels"    : labels,       # (max_len,)  -100 at pad positions
            "attn_mask" : attn_mask,    # (max_len,)
        }


# ─────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────

def get_dataloaders(data_dir: str = cfg.data_dir,
                   captions_file: str = cfg.captions_file):
    """
    Returns (train_loader, val_loader) ready for training.
    """
    train_dataset = Flickr8kDataset(data_dir, captions_file, split="train")
    val_dataset   = Flickr8kDataset(data_dir, captions_file, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = 2,           # parallel data loading
        pin_memory  = True,        # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────
#  Quick test — python dataset.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Testing Dataset ===\n")

    # This will only work after you download Flickr8k into ./data/
    try:
        train_loader, val_loader = get_dataloaders()
        batch = next(iter(train_loader))

        print(f"Batch keys:      {list(batch.keys())}")
        print(f"Image shape:     {batch['image'].shape}")       # (B, 3, 224, 224)
        print(f"input_ids shape: {batch['input_ids'].shape}")   # (B, 64)
        print(f"labels shape:    {batch['labels'].shape}")      # (B, 64)

        # Decode one caption back to text to verify tokenization
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        ids = batch["input_ids"][0]
        ids = ids[ids != tok.eos_token_id]   # remove padding
        print(f"\nSample caption: {tok.decode(ids)}")

    except FileNotFoundError:
        print("Dataset not found. Please download Flickr8k to ./data/")
        print("Download from: https://www.kaggle.com/datasets/adityajn105/flickr8k")
