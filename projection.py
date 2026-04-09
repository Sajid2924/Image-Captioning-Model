# projection.py  —  Image Feature → GPT-2 Prefix Mapper
#
# What this does:
#  Takes the 2048-dim CNN feature vector and maps it to
#  `prefix_length` tokens of size 768 (GPT-2 embedding size).
#
#  These prefix tokens are prepended to the caption tokens
#  before the GPT-2 transformer, acting as a "visual prompt"
#  that conditions the language model on the image.
#
# Architecture:
#  Linear(2048 → prefix_length * 768)
#  → Reshape to (batch, prefix_length, 768)
#
# This is the ONLY trainable part if CNN is frozen.
# It's a small module (~20M params) that learns to translate
# vision features into the language model's embedding space.

import torch
import torch.nn as nn
from config import cfg


class ProjectionLayer(nn.Module):
    """
    Maps CNN image features (2048-dim) to GPT-2 prefix tokens.

    The projection is a simple 2-layer MLP with a Tanh activation.
    Using 2 layers (instead of 1 linear) allows learning a
    non-linear mapping between the vision and language spaces.
    """

    def __init__(
        self,
        encoder_dim: int = cfg.encoder_out_dim,  # 2048
        gpt2_embd_dim: int = cfg.gpt2_n_embd,  # 768
        prefix_length: int = cfg.prefix_length,  # 10
    ):
        super().__init__()

        self.prefix_length = prefix_length
        self.gpt2_embd_dim = gpt2_embd_dim

        # Output size = prefix_length tokens × token embedding size
        # e.g. 10 × 768 = 7680
        out_size = prefix_length * gpt2_embd_dim

        # 2-layer MLP: encoder_dim → hidden → out_size
        hidden_dim = (encoder_dim + out_size) // 2  # midpoint as hidden size

        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.Tanh(),  # smooth non-linearity
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_size),
        )

        print(f"[Projection] {encoder_dim} → {hidden_dim} → {out_size}")
        print(f"[Projection] Output reshaped to ({prefix_length}, {gpt2_embd_dim})")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (batch, 2048)   — from CNN encoder

        Returns:
            prefix: (batch, prefix_length, 768)  — ready for GPT-2
        """
        B = image_features.shape[0]

        # Project: (B, 2048) → (B, prefix_length * 768)
        projected = self.net(image_features)

        # Reshape: (B, prefix_length * 768) → (B, prefix_length, 768)
        prefix = projected.view(B, self.prefix_length, self.gpt2_embd_dim)

        return prefix


# Quick test  —  python projection.py
if __name__ == "__main__":
    print("\n=== Testing ProjectionLayer ===\n")

    proj = ProjectionLayer().to(cfg.device)

    # Simulate encoder output: batch of 4 images, each 2048-dim
    dummy_features = torch.randn(4, cfg.encoder_out_dim).to(cfg.device)

    prefix = proj(dummy_features)

    print(f"\nInput:  {dummy_features.shape}")  # (4, 2048)
    print(f"Output: {prefix.shape}")  # (4, 10, 768)
    print(f"\nProjection layer works correctly!")

    total = sum(p.numel() for p in proj.parameters())
    print(f"Trainable parameters: {total:,}")
