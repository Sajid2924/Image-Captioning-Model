# =============================================================
#  config.py  —  All hyperparameters and settings in one place
#  Change values here; every other file imports from here
# =============================================================

import torch

class Config:
    # ── Device ────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset ───────────────────────────────────────────────
    dataset_name   = "flickr8k"          # swap to "coco" later
    data_dir       = "./data"            # where images live
    captions_file  = "./data/captions.txt"
    train_split    = 0.90                # 90% train, 10% val
    max_caption_len = 64                 # token limit per caption

    # ── CNN Encoder ───────────────────────────────────────────
    encoder_model   = "resnet50"         # resnet50 | efficientnet_b0
    encoder_frozen  = False               # freeze CNN weights (saves VRAM)
    encoder_out_dim = 2048               # ResNet50 final feature size

    # ── GPT-2 ─────────────────────────────────────────────────
    gpt2_variant    = "gpt2"             # "gpt2" = small (117M params)
                                         # "gpt2-medium" if you have 8GB VRAM
    gpt2_vocab_size = 50257
    gpt2_n_layer    = 12                 # transformer blocks
    gpt2_n_head     = 12                 # attention heads
    gpt2_n_embd     = 768               # embedding dimension
    gpt2_max_pos    = 1024              # max sequence length

    # ── Projection Layer ──────────────────────────────────────
    # Maps encoder_out_dim (2048) → gpt2_n_embd (768)
    prefix_length   = 10                # number of prefix tokens fed to GPT-2

    # ── Training ──────────────────────────────────────────────
    batch_size      = 16                # lower to 8 if you get OOM errors
    num_epochs      = 50
    learning_rate   = 3e-5
    weight_decay    = 1e-3
    warmup_steps    = 1000
    grad_clip       = 1.0               # gradient clipping value
    mixed_precision = True              # use float16 — saves ~40% VRAM

    # ── Early Stopping ────────────────────────────────────────
    early_stopping_patience  = 3
    early_stopping_min_delta = 0.001

    # ── Checkpointing ─────────────────────────────────────────
    checkpoint_dir  = "./checkpoints"
    save_every      = 1                 # save after every N epochs
    resume_from     = None              # path to checkpoint to resume from

    # ── Logging ───────────────────────────────────────────────
    log_every       = 500               # print loss every N steps

    # ── Inference ─────────────────────────────────────────────
    max_gen_len     = 30               # max tokens to generate
    temperature     = 0.7
    top_k           = 50


cfg = Config()
