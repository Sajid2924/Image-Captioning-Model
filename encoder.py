# =============================================================
#  encoder.py  —  Pretrained CNN Image Encoder
#
#  What this does:
#    1. Loads a pretrained ResNet-50 from torchvision
#    2. Removes its final classification head (we don't want labels)
#    3. Extracts a 2048-dim feature vector for any input image
#    4. Optionally freezes all weights (no gradient updates on CNN)
#
#  Why freeze?  The CNN already has great visual features from
#  ImageNet training. Freezing saves VRAM and speeds up training.
# =============================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from config import cfg


class ImageEncoder(nn.Module):
    """
    Wraps a pretrained ResNet-50 and strips its classifier head.
    Returns a flat feature vector of shape (batch, 2048).
    """

    def __init__(self, frozen: bool = cfg.encoder_frozen):
        super().__init__()

        # ── Load pretrained ResNet-50 ──────────────────────────
        # weights="IMAGENET1K_V2" = best available ResNet-50 weights
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        print(f"[Encoder] Loaded pretrained ResNet-50")

        # ── Remove the final FC classification layer ───────────
        # ResNet architecture: conv layers → avgpool → fc(2048→1000)
        # We want the 2048-dim vector BEFORE the fc layer
        # nn.Sequential(*list(...)[:-1]) keeps everything except the last layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        #  Output shape after avgpool: (batch, 2048, 1, 1)

        # ── Freeze CNN weights ─────────────────────────────────
        if frozen:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print(f"[Encoder] CNN weights FROZEN — will not be updated during training")
        else:
            print(f"[Encoder] CNN weights TRAINABLE — will be fine-tuned")

        self.out_dim = cfg.encoder_out_dim   # 2048
        self.frozen  = frozen

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, 224, 224)  — normalized image tensors

        Returns:
            features: (batch, 2048)       — flat visual feature vectors
        """
        # Pass through all conv layers + avgpool
        features = self.feature_extractor(images)
        # Shape: (batch, 2048, 1, 1)

        # Flatten the spatial dimensions → (batch, 2048)
        features = features.squeeze(-1).squeeze(-1)

        return features


# =============================================================
#  Image preprocessing pipeline
#  All images must be resized and normalized before the CNN
#  These exact mean/std values match ImageNet training statistics
# =============================================================

def get_image_transform():
    """
    Returns the torchvision transform pipeline for preprocessing images.
    Must match what ResNet-50 was originally trained with.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),           # ResNet expects 224x224
        transforms.ToTensor(),                   # PIL Image → (3, H, W) tensor, values in [0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],          # ImageNet channel means
            std =[0.229, 0.224, 0.225]           # ImageNet channel stds
        )
    ])


# =============================================================
#  Quick test — run this file directly to verify encoder works
#  python encoder.py
# =============================================================

if __name__ == "__main__":
    import torch

    print("\n=== Testing ImageEncoder ===")
    encoder = ImageEncoder(frozen=True).to(cfg.device)

    # Create a fake batch of 4 images: (batch=4, channels=3, H=224, W=224)
    dummy_images = torch.randn(4, 3, 224, 224).to(cfg.device)

    with torch.no_grad():
        features = encoder(dummy_images)

    print(f"Input shape:   {dummy_images.shape}")     # (4, 3, 224, 224)
    print(f"Output shape:  {features.shape}")         # (4, 2048)
    print(f"Encoder works correctly!\n")

    # Count parameters
    total   = sum(p.numel() for p in encoder.parameters())
    frozen  = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    print(f"Total params:   {total:,}")
    print(f"Frozen params:  {frozen:,}")
    print(f"Trainable:      {total - frozen:,}")
