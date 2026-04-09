# gpt2.py  —  GPT-2 Architecture built from scratch
#            + loading of pretrained OpenAI/HuggingFace weights
#
# Architecture breakdown:
#
#  GPT2 = Token Embedding
#        + Positional Embedding
#        + N × TransformerBlock (
#              LayerNorm → CausalSelfAttention → residual
#              LayerNorm → MLP → residual
#          )
#        + Final LayerNorm
#        + LM Head (linear projection → vocab logits)
#
# The "from scratch" part means we define every layer ourselves
# using only torch.nn primitives. Then we copy weights from the
# HuggingFace GPT-2 checkpoint into our matching layer names.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel  # only used for weight loading
from config import cfg


# 1. Causal Self-Attention, "Causal" = each token can only attend to past tokens
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Key idea:
      Q, K, V projections → scaled dot-product attention
      → apply causal mask (upper triangle = -inf)
      → softmax → weighted sum of V
      → output projection
    """

    def __init__(self, n_embd: int, n_head: int, max_pos: int):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dim must be divisible by num heads"

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head  # dimension per attention head

        # Single linear layer computes Q, K, V all at once (3x more efficient)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # → (Q, K, V) stacked

        # Output projection after concatenating heads
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Causal mask: upper-triangular matrix of -inf
        # Registered as a buffer so it moves to the right device automatically
        # Shape: (1, 1, max_pos, max_pos) — broadcastable
        mask = torch.tril(torch.ones(max_pos, max_pos)).view(1, 1, max_pos, max_pos)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_embd)
        Returns:
            out: (batch, seq_len, n_embd)
        """
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Step 1: Compute Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each: (B, T, C)

        # Step 2: Split into heads
        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Step 3: Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

        # Apply causal mask: future positions become -inf → softmax → 0
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # Step 4: Weighted sum of values
        out = attn @ v  # (B, n_head, T, head_dim)

        # Step 5: Merge heads back
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Step 6: Output projection
        return self.c_proj(out)


# 2. Feed-Forward MLP (called after attention in each block)
#   Expands to 4× hidden size, applies GELU, contracts back
class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)  # expand
        self.c_proj = nn.Linear(4 * n_embd, n_embd)  # contract
        self.act = nn.GELU()  # GPT-2 uses GELU

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


# 3. Transformer Block = Attention + MLP with residual connections
class TransformerBlock(nn.Module):
    """
    One GPT-2 layer:
        x → LayerNorm → Attention → + x  (residual)
          → LayerNorm → MLP       → + x  (residual)
    Pre-norm style (norm before, not after — matches GPT-2)
    """

    def __init__(self, n_embd: int, n_head: int, max_pos: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)  # norm before attn
        self.attn = CausalSelfAttention(n_embd, n_head, max_pos)  # attention
        self.ln_2 = nn.LayerNorm(n_embd)  # norm before mlp
        self.mlp = MLP(n_embd)  # feed-forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # attention sub-layer with residual
        x = x + self.mlp(self.ln_2(x))  # mlp sub-layer with residual
        return x


# 4. Full GPT-2 Model
class GPT2(nn.Module):
    """
    GPT-2 built from scratch.

    During captioning, image prefix tokens are PREPENDED to the
    token embeddings before being passed through the transformer.
    The model then generates caption tokens autoregressively.
    """

    def __init__(
        self,
        vocab_size: int = cfg.gpt2_vocab_size,
        n_embd: int = cfg.gpt2_n_embd,
        n_layer: int = cfg.gpt2_n_layer,
        n_head: int = cfg.gpt2_n_head,
        max_pos: int = cfg.gpt2_max_pos,
    ):
        super().__init__()

        self.n_embd = n_embd

        # Token + positional embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)  # word token embeddings
        self.wpe = nn.Embedding(max_pos, n_embd)  # word position embeddings

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, max_pos) for _ in range(n_layer)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # Language model head: maps hidden states → vocabulary logits
        # Weight is TIED to wte (standard GPT-2 design) — saves parameters
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T)  token indices
        prefix_embeds: torch.Tensor = None,  # (B, prefix_len, n_embd)  image prefix
        attention_mask: torch.Tensor = None,  # (B, T)  optional mask
    ) -> torch.Tensor:
        """
        Args:
            input_ids:     token IDs for the caption text  (B, T)
            prefix_embeds: projected image embeddings      (B, prefix_len, n_embd)
                           These are prepended before text tokens

        Returns:
            logits: (B, T_total, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings for text tokens
        token_embeds = self.wte(input_ids)  # (B, T, n_embd)

        # Prepend image prefix if provided
        if prefix_embeds is not None:
            # Concatenate: [image_prefix | text_tokens]
            # Shape: (B, prefix_len + T, n_embd)
            x = torch.cat([prefix_embeds, token_embeds], dim=1)
        else:
            x = token_embeds

        seq_len = x.shape[1]

        # Positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        x = x + self.wpe(positions)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (B, seq_len, vocab_size)

        return logits


# 5. Weight loading from HuggingFace pretrained GPT-2
def load_pretrained_gpt2(model: GPT2, variant: str = cfg.gpt2_variant) -> GPT2:
    """
    Loads pretrained GPT-2 weights from HuggingFace into our
    custom GPT2 model by matching parameter names manually.

    HuggingFace name          → Our name
    transformer.wte           → wte
    transformer.wpe           → wpe
    transformer.h.N.ln_1      → blocks.N.ln_1
    transformer.h.N.attn.c_attn → blocks.N.attn.c_attn
    transformer.h.N.attn.c_proj → blocks.N.attn.c_proj
    transformer.h.N.ln_2      → blocks.N.ln_2
    transformer.h.N.mlp.c_fc  → blocks.N.mlp.c_fc
    transformer.h.N.mlp.c_proj→ blocks.N.mlp.c_proj
    transformer.ln_f          → ln_f
    """
    print(f"[GPT-2] Downloading pretrained weights: '{variant}'")
    hf_model = GPT2LMHeadModel.from_pretrained(variant)
    hf_sd = hf_model.state_dict()

    our_sd = model.state_dict()

    # Build a mapping: our key → HuggingFace key
    # Most keys map with "transformer." prefix in HF
    copied = 0
    skipped = 0

    for our_key in our_sd.keys():
        # Construct the expected HuggingFace key
        hf_key = "transformer." + our_key  # e.g. "transformer.wte.weight"

        # blocks.N → h.N conversion
        if our_key.startswith("blocks."):
            # "blocks.0.ln_1.weight" → "transformer.h.0.ln_1.weight"
            hf_key = "transformer." + our_key.replace("blocks.", "h.", 1)

        # lm_head is under transformer. in HF
        if our_key == "lm_head.weight":
            hf_key = "lm_head.weight"

        if hf_key in hf_sd:
            hf_tensor = hf_sd[hf_key]
            our_tensor = our_sd[our_key]

            # Conv1D in HF GPT-2 stores weights transposed vs nn.Linear
            # HF uses Conv1D: weight shape is (in, out)
            # nn.Linear: weight shape is (out, in)
            if hf_tensor.shape != our_tensor.shape:
                hf_tensor = hf_tensor.t()  # transpose to match nn.Linear

            if hf_tensor.shape == our_tensor.shape:
                our_sd[our_key].copy_(hf_tensor)
                copied += 1
            else:
                print(
                    f"  [skip] shape mismatch: {our_key} {our_tensor.shape} vs {hf_tensor.shape}"
                )
                skipped += 1
        else:
            skipped += 1

    model.load_state_dict(our_sd)
    print(f"[GPT-2] Weights loaded: {copied} copied, {skipped} skipped")
    del hf_model  # free memory
    return model


# Quick test — python gpt2.py
if __name__ == "__main__":
    print("\n=== Testing GPT-2 from scratch ===\n")

    model = GPT2()
    model = load_pretrained_gpt2(model)
    model = model.to(cfg.device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")

    # Test forward pass with dummy inputs
    B, T = 2, 20
    dummy_ids = torch.randint(0, cfg.gpt2_vocab_size, (B, T)).to(cfg.device)
    dummy_prefix = torch.randn(B, cfg.prefix_length, cfg.gpt2_n_embd).to(cfg.device)

    with torch.no_grad():
        logits = model(input_ids=dummy_ids, prefix_embeds=dummy_prefix)

    expected_seq = cfg.prefix_length + T
    print(f"\nInput token ids: {dummy_ids.shape}")
    print(f"Input prefix:    {dummy_prefix.shape}")
    print(f"Output logits:   {logits.shape}")  # (2, prefix_len+T, 50257)
    print(f"Expected:        ({B}, {expected_seq}, {cfg.gpt2_vocab_size})")
    print("\nGPT-2 works correctly!")
