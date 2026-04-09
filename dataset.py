# dataset.py  —  Combined Dataset Loader
#
# Loads from 3 sources and combines them:
#  1. Flickr8k  → data/Images/ + data/captions.txt
#  2. COCO 2014 → data/train2014/ + data/annotations_trainval2014/captions_train2014.json
#  3. COCO 2017 → data/val2017/   + data/annotations_trainval2017/captions_val2017.json
#
# Total pairs: ~40k (Flickr) + ~590k (COCO2014) + ~25k (COCO2017) = ~655k pairs
#
# Only dataset.py changed — everything else untouched.

import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

from encoder import get_image_transform
from config import cfg


class CombinedCaptionDataset(Dataset):
    """
    Combines Flickr8k + COCO 2014 train + COCO 2017 val into one dataset.

    Each item returns:
        image     : (3, 224, 224) normalized image tensor
        input_ids : (max_caption_len,) token IDs
        labels    : (max_caption_len,) same but -100 at padding
        attn_mask : (max_caption_len,) 1=real token, 0=padding

    """

    def __init__(self, data_dir: str, split: str = "train"):
        self.transform = get_image_transform()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = cfg.max_caption_len

        all_pairs = []

        # Source 1: Flickr8k
        flickr_pairs = self._load_flickr8k(data_dir)
        print(f"[Dataset] Flickr8k pairs:    {len(flickr_pairs):,}")
        all_pairs.extend(flickr_pairs)

        # Source 2: COCO 2014 train
        coco2014_json = os.path.join(
            data_dir, "annotations_trainval2014", "captions_train2014.json"
        )
        coco2014_img_dir = os.path.join(data_dir, "train2014")
        if os.path.exists(coco2014_json) and os.path.exists(coco2014_img_dir):
            coco2014_pairs = self._load_coco_json(coco2014_json, coco2014_img_dir)
            print(f"[Dataset] COCO 2014 train pairs: {len(coco2014_pairs):,}")
            all_pairs.extend(coco2014_pairs)
        else:
            print(f"[Dataset] COCO 2014 not found — skipping")

        # Source 3: COCO 2017 val
        coco2017_json = os.path.join(
            data_dir, "annotations_trainval2017", "captions_val2017.json"
        )
        coco2017_img_dir = os.path.join(data_dir, "val2017")
        if os.path.exists(coco2017_json) and os.path.exists(coco2017_img_dir):
            coco2017_pairs = self._load_coco_json(coco2017_json, coco2017_img_dir)
            print(f"[Dataset] COCO 2017 val pairs:   {len(coco2017_pairs):,}")
            all_pairs.extend(coco2017_pairs)
        else:
            print(f"[Dataset] COCO 2017 not found — skipping")

        # Shuffle before split so all 3 sources appear in both train and val
        # seed=42 ensures same split every run — reproducible
        random.seed(42)
        random.shuffle(all_pairs)

        print(f"[Dataset] Total pairs combined:  {len(all_pairs):,}")

        # Train / Val split
        split_idx = int(len(all_pairs) * cfg.train_split)
        if split == "train":
            self.pairs = all_pairs[:split_idx]
        else:
            self.pairs = all_pairs[split_idx:]

        print(f"[Dataset] {split} set: {len(self.pairs):,} pairs")

    # Flickr8k loader
    def _load_flickr8k(self, data_dir: str):
        """
        Loads from data/captions.txt + data/Images/
        Format: image.jpg,caption text
        """
        captions_file = os.path.join(data_dir, "captions.txt")
        image_dir = os.path.join(data_dir, "Images")

        if not os.path.exists(captions_file):
            print(f"[Dataset] Flickr8k captions.txt not found — skipping")
            return []

        pairs = []
        with open(captions_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line_num == 0:
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                image_name, caption = parts
                image_path = os.path.join(image_dir, image_name.strip())
                if os.path.exists(image_path):
                    pairs.append((image_path, caption.strip()))
        return pairs

    # COCO JSON loader (works for both 2014 and 2017)
    def _load_coco_json(self, json_path: str, image_dir: str):
        """
        Loads COCO caption JSON.

        JSON structure:
          {
            "images":      [{"id": 123, "file_name": "COCO_train2014_...jpg"}, ...]
            "annotations": [{"image_id": 123, "caption": "A dog..."}, ...]
          }
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build image_id → file_name mapping
        id_to_filename = {}
        for img in data["images"]:
            id_to_filename[img["id"]] = img["file_name"]

        # Build (image_path, caption) pairs
        pairs = []
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"].strip()

            if image_id not in id_to_filename:
                continue

            filename = id_to_filename[image_id]
            image_path = os.path.join(image_dir, filename)

            if os.path.exists(image_path):
                pairs.append((image_path, caption))

        return pairs

    # Dataset interface

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, caption_text = self.pairs[idx]

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # (3, 224, 224)

        # Tokenize caption
        caption_with_eos = caption_text + self.tokenizer.eos_token

        encoding = self.tokenizer(
            caption_with_eos,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attn_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "attn_mask": attn_mask,
        }


# DataLoader factory — called by train.py
def get_dataloaders(
    data_dir: str = cfg.data_dir, captions_file: str = cfg.captions_file
):
    """
    Returns (train_loader, val_loader) using combined dataset.
    captions_file param kept for backward compatibility — not used here.
    """
    train_dataset = CombinedCaptionDataset(data_dir, split="train")
    val_dataset = CombinedCaptionDataset(data_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,  # increase for large dataset
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


# Quick test — python dataset.py
if __name__ == "__main__":
    print("\n=== Testing Combined Dataset ===\n")

    train_loader, val_loader = get_dataloaders()
    batch = next(iter(train_loader))

    print(f"\nBatch keys:      {list(batch.keys())}")
    print(f"Image shape:     {batch['image'].shape}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"labels shape:    {batch['labels'].shape}")

    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ids = batch["input_ids"][0]
    ids = ids[ids != tok.eos_token_id]
    print(f"\nSample caption:  {tok.decode(ids)}")
