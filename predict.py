# =============================================================
#  predict.py  —  Generate a caption for YOUR image
#
#  Usage:
#    python predict.py                          ← uses image.jpg in project folder
#    python predict.py --image photo.jpg        ← custom image name
#    python predict.py --image "C:\Users\sajid\Pictures\dog.jpg"
# =============================================================

import argparse
import os
import torch
from PIL import Image
from transformers import GPT2Tokenizer

from model   import ImageCaptioningModel
from encoder import get_image_transform
from config  import cfg


# ── Defaults — just drop image.jpg in your project folder ─────
DEFAULT_IMAGE      = "./image.jpg"
DEFAULT_CHECKPOINT = "./checkpoints/sample_best_model.pt"


def load_model(checkpoint_path: str):
    """Load model and restore weights from checkpoint."""

    print(f"\n[predict] Loading model architecture...")
    model = ImageCaptioningModel().to("cpu")

    if not os.path.exists(checkpoint_path):
        print(f"\n  ❌ Checkpoint not found: {checkpoint_path}")
        print(f"     Run sample_test.py first to generate weights.\n")
        exit(1)

    print(f"[predict] Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    trained_epoch = ckpt.get("epoch", "?")
    trained_loss  = ckpt.get("loss",  "?")
    print(f"[predict] Checkpoint info — epoch: {trained_epoch}, loss: {trained_loss:.4f}")

    model.eval()
    return model


def predict(image_path: str, checkpoint_path: str):
    """Full pipeline: image path → caption string."""

    # ── Validate image path ───────────────────────────────────
    if not os.path.exists(image_path):
        print(f"\n  ❌ Image not found: {image_path}")
        print(f"     Make sure image.jpg is in your PRML project folder.\n")
        exit(1)

    # ── Load model ────────────────────────────────────────────
    model = load_model(checkpoint_path)

    # ── Load tokenizer ────────────────────────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load and preprocess image ─────────────────────────────
    print(f"[predict] Loading image: {image_path}")
    transform = get_image_transform()
    image     = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)   # (1, 3, 224, 224)

    # ── Generate caption ──────────────────────────────────────
    print(f"[predict] Generating caption...\n")

    with torch.no_grad():
        caption = model.generate(
            images         = image_tensor,
            tokenizer      = tokenizer,
            max_new_tokens = 30,
            temperature    = 1.0,
            top_k          = 50,
        )

    # ── Print result ──────────────────────────────────────────
    print("="*50)
    print(f"  Image:   {image_path}")
    print(f"  Caption: {caption}")
    print("="*50 + "\n")

    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate caption for an image")
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Path to your image file (default: ./image.jpg)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    args = parser.parse_args()

    predict(args.image, args.checkpoint)