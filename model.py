# model.py  —  Full Image Captioning Model
#
# Combines:
#  ImageEncoder → ProjectionLayer → GPT2
#
# Forward pass:
#  image (3,224,224)
#    ↓ ImageEncoder
#  features (2048,)
#    ↓ ProjectionLayer
#  prefix_tokens (10, 768)
#    ↓ GPT2 (prepended to caption tokens)
#  logits (prefix_len + caption_len, vocab_size)
#    ↓ cross-entropy loss on caption positions only
#  loss

import torch
import torch.nn as nn
from encoder import ImageEncoder
from projection import ProjectionLayer
from gpt2 import GPT2, load_pretrained_gpt2
from config import cfg


class ImageCaptioningModel(nn.Module):
    """
    End-to-end image captioning model.

    Trainable components:
      - ProjectionLayer (always trained)
      - GPT-2 weights  (always trained)
      - CNN Encoder    (frozen by default — set encoder_frozen=False to unfreeze)
    """

    def __init__(self):
        super().__init__()

        # 1. CNN Encoder (pretrained, frozen)
        print("\n[Model] Loading CNN Encoder...")
        self.encoder = ImageEncoder(frozen=cfg.encoder_frozen)

        # 2. Projection Layer (trainable)
        print("\n[Model] Building Projection Layer...")
        self.projection = ProjectionLayer(
            encoder_dim=cfg.encoder_out_dim,
            gpt2_embd_dim=cfg.gpt2_n_embd,
            prefix_length=cfg.prefix_length,
        )

        # 3. GPT-2 (from scratch + pretrained weights)
        print("\n[Model] Building GPT-2 from scratch and loading pretrained weights...")
        self.gpt2 = GPT2(
            vocab_size=cfg.gpt2_vocab_size,
            n_embd=cfg.gpt2_n_embd,
            n_layer=cfg.gpt2_n_layer,
            n_head=cfg.gpt2_n_head,
            max_pos=cfg.gpt2_max_pos,
        )
        self.gpt2 = load_pretrained_gpt2(self.gpt2, cfg.gpt2_variant)

        self.prefix_length = cfg.prefix_length

        print("\n[Model] ✓ ImageCaptioningModel ready")
        self._print_param_counts()

    def forward(
        self,
        images: torch.Tensor,  # (B, 3, 224, 224)
        input_ids: torch.Tensor,  # (B, T)  caption token ids
        labels: torch.Tensor,  # (B, T)  same but -100 at pad positions
    ):
        """
        Full forward pass.

        Returns:
            loss   : scalar cross-entropy loss
            logits : (B, prefix_len + T, vocab_size)
        """

        # Step 1: Extract visual features
        # image (B,3,224,224) → features (B, 2048)
        with torch.set_grad_enabled(not cfg.encoder_frozen):
            image_features = self.encoder(images)

        # Step 2: Project to GPT-2 embedding space
        # features (B, 2048) → prefix (B, prefix_len, 768)
        prefix_embeds = self.projection(image_features)

        # Step 3: GPT-2 forward pass with prefix
        # Concatenates [prefix_tokens | text_tokens] inside GPT2.forward()
        # logits shape: (B, prefix_len + T, vocab_size)
        logits = self.gpt2(input_ids=input_ids, prefix_embeds=prefix_embeds)

        # Step 4: Compute cross-entropy loss
        # We only want loss on the TEXT tokens, not the prefix tokens
        # So we slice logits to skip the first prefix_len positions

        # Logits for text token predictions:
        # position (prefix_len - 1) predicts token 0
        # position (prefix_len)     predicts token 1  ...
        # We take positions [prefix_len-1 : prefix_len-1+T-1] to align
        # with next-token prediction (standard language model objective)

        # Shift for next-token prediction:
        # logits at position i predicts token at position i+1
        shift_logits = logits[:, self.prefix_length : -1, :]  # (B, T-1, vocab)
        shift_labels = labels[:, 1:]  # (B, T-1)

        # Flatten for CrossEntropyLoss
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, cfg.gpt2_vocab_size),  # (B*(T-1), vocab)
            shift_labels.reshape(-1),  # (B*(T-1),)
            ignore_index=-100,  # ignores padding positions in loss
        )

        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,  # (1, 3, 224, 224)  — single image
        tokenizer,
        max_new_tokens: int = cfg.max_gen_len,
        temperature: float = cfg.temperature,
        top_k: int = cfg.top_k,
    ) -> str:
        """
        Greedy / top-k sampling for caption generation at inference time.

        Args:
            images:    single image tensor  (1, 3, 224, 224)
            tokenizer: GPT-2 tokenizer (for encoding BOS and decoding output)

        Returns:
            caption: generated caption string
        """
        self.eval()
        device = images.device

        # Get image prefix
        image_features = self.encoder(images)  # (1, 2048)
        prefix_embeds = self.projection(image_features)  # (1, prefix_len, 768)

        # Start with BOS (beginning-of-sequence) token
        # GPT-2 uses EOS as BOS in practice
        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        generated = torch.tensor([[bos_id]], device=device)  # (1, 1)

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.gpt2(input_ids=generated, prefix_embeds=prefix_embeds)

            # Take logits for the LAST generated position only
            next_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # Top-k filtering: keep only top k candidates
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(
                    next_logits < min_val, float("-inf")
                )

            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)   #(1, 1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode tokens to string (skip BOS token at position 0)
        caption = tokenizer.decode(generated[0, 1:], skip_special_tokens=True)
        return caption.strip()

    def _print_param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n{''*45}")
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}")
        print(f"  Frozen parameters:    {frozen:>12,}")
        print(f"{''*45}\n")


# Quick test — python model.py
if __name__ == "__main__":
    print("\n=== Testing Full ImageCaptioningModel ===")

    model = ImageCaptioningModel().to(cfg.device)

    B, T = 2, 32
    dummy_images = torch.randn(B, 3, 224, 224).to(cfg.device)
    dummy_input_ids = torch.randint(0, cfg.gpt2_vocab_size, (B, T)).to(cfg.device)
    dummy_labels = dummy_input_ids.clone()
    dummy_labels[:, -5:] = -100  # simulate some padding at the end

    loss, logits = model(dummy_images, dummy_input_ids, dummy_labels)

    print(f"\nImages shape:  {dummy_images.shape}")
    print(f"Logits shape:  {logits.shape}")
    print(f"Loss:          {loss.item():.4f}")
    print(f"\nFull model forward pass works correctly!")
