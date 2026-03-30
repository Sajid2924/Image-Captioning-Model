# =============================================================
#  evaluate.py  —  Evaluation Metrics on Flickr8k Val Set
#
#  Computes: BLEU-4, METEOR, ROUGE-L, CIDEr
#  Uses ONLY Flickr8k validation images (not COCO)
#
#  Install first:
#    pip install nltk rouge-score
#    python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
#
#  Run:
#    python evaluate.py
# =============================================================

import os
import random
import math
import torch
from PIL import Image
from collections import defaultdict, Counter
from transformers import GPT2Tokenizer

# ── Metric libraries ──────────────────────────────────────────
import nltk
from nltk.translate.bleu_score  import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score                 import rouge_scorer

# ── Project imports ───────────────────────────────────────────
from config  import cfg
from model   import ImageCaptioningModel
from encoder import get_image_transform

# ── Auto-download NLTK data if missing ────────────────────────
for resource in ['punkt', 'wordnet', 'punkt_tab', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{resource}') if 'punkt' in resource else nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# ─────────────────────────────────────────────────────────────
#  Settings — change these if needed
# ─────────────────────────────────────────────────────────────

CHECKPOINT  = "./checkpoints/best_model.pt"   # model weights to evaluate
DATA_DIR    = "./data"                         # where Flickr8k lives
VAL_SPLIT   = 0.10                            # must match dataset.py (10%)
NUM_IMAGES  = None                            # None = all val images, or set e.g. 100 for quick test
DEVICE      = cfg.device

# ─────────────────────────────────────────────────────────────
#  Load Flickr8k val images only
#  Uses same seed=42 shuffle as dataset.py for consistent split
# ─────────────────────────────────────────────────────────────

def load_flickr8k_val():
    """
    Returns dict: {image_path: [ref_caption_1, ref_caption_2, ...]}
    Only the val split (last 10%) of Flickr8k.
    """
    captions_file     = os.path.join(DATA_DIR, "captions.txt")
    image_dir         = os.path.join(DATA_DIR, "Images")
    image_to_captions = defaultdict(list)
    all_images        = []

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
                if image_path not in image_to_captions:
                    all_images.append(image_path)
                image_to_captions[image_path].append(caption.strip())

    # Same seed=42 shuffle as dataset.py → consistent val split
    random.seed(42)
    random.shuffle(all_images)

    split_idx  = int(len(all_images) * (1 - VAL_SPLIT))
    val_images = all_images[split_idx:]

    return {img: image_to_captions[img] for img in val_images}


# ─────────────────────────────────────────────────────────────
#  Load model from checkpoint
# ─────────────────────────────────────────────────────────────

def load_model():
    print(f"\n[Eval] Loading model from {CHECKPOINT} ...")
    if not os.path.exists(CHECKPOINT):
        print(f"  ❌ Checkpoint not found: {CHECKPOINT}")
        exit(1)

    model = ImageCaptioningModel().to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss",  0.0)
    print(f"[Eval] Checkpoint loaded — epoch: {epoch}, val_loss: {loss:.4f}")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
#  Generate caption for a single image
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_caption(model, image_path, tokenizer, transform):
    image        = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    caption      = model.generate(image_tensor, tokenizer)
    return caption


# ─────────────────────────────────────────────────────────────
#  CIDEr — TF-IDF weighted n-gram similarity
# ─────────────────────────────────────────────────────────────

def compute_cider(hypotheses, references):
    """
    hypotheses : list of generated caption strings
    references : list of lists of reference caption strings
    Returns CIDEr score (float)
    """
    def get_ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def tfidf_idf(ngram_counts_list):
        df      = Counter()
        n_docs  = len(ngram_counts_list)
        for counts in ngram_counts_list:
            for ngram in counts:
                df[ngram] += 1
        return {ng: math.log((n_docs + 1.0) / (df[ng] + 1.0)) for ng in df}

    scores = []
    for n in range(1, 5):   # n-grams 1 to 4
        hyp_ngrams = [get_ngrams(h.lower().split(), n) for h in hypotheses]
        ref_ngrams = [[get_ngrams(r.lower().split(), n) for r in refs] for refs in references]

        all_ref_ng = [ng for refs in ref_ngrams for ng in refs]
        idf        = tfidf_idf(all_ref_ng)

        sim_scores = []
        for hyp, refs in zip(hyp_ngrams, ref_ngrams):
            ref_avg = Counter()
            for r in refs:
                for ng, cnt in r.items():
                    ref_avg[ng] += cnt / len(refs)

            hyp_vec = {ng: cnt * idf.get(ng, 0) for ng, cnt in hyp.items()}
            ref_vec = {ng: cnt * idf.get(ng, 0) for ng, cnt in ref_avg.items()}

            dot      = sum(hyp_vec.get(ng, 0) * ref_vec.get(ng, 0) for ng in ref_vec)
            hyp_norm = math.sqrt(sum(v ** 2 for v in hyp_vec.values())) + 1e-10
            ref_norm = math.sqrt(sum(v ** 2 for v in ref_vec.values())) + 1e-10

            sim_scores.append(dot / (hyp_norm * ref_norm))

        scores.append(sum(sim_scores) / len(sim_scores))

    return (sum(scores) / 4) * 10.0   # standard CIDEr scaling


# ─────────────────────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────────────────────

def evaluate():
    # ── Load data ─────────────────────────────────────────────
    print("\n[Eval] Loading Flickr8k val set...")
    val_data    = load_flickr8k_val()
    image_paths = list(val_data.keys())

    if NUM_IMAGES is not None:
        image_paths = image_paths[:NUM_IMAGES]
        print(f"[Eval] Using subset: {NUM_IMAGES} images")

    print(f"[Eval] Total images to evaluate: {len(image_paths):,}")

    # ── Load model & tokenizer ────────────────────────────────
    model     = load_model()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    transform = get_image_transform()

    # ── Generate captions ─────────────────────────────────────
    print(f"\n[Eval] Generating captions...")
    hypotheses = []
    references = []

    for i, image_path in enumerate(image_paths):
        if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
            print(f"  Progress: {i+1}/{len(image_paths)}")

        generated = generate_caption(model, image_path, tokenizer, transform)
        hypotheses.append(generated)
        references.append(val_data[image_path])

    print(f"[Eval] Generation complete.\n")

    # ── BLEU-4 ────────────────────────────────────────────────
    hyp_tokens = [h.lower().split() for h in hypotheses]
    ref_tokens = [[r.lower().split() for r in refs] for refs in references]

    bleu4 = corpus_bleu(
        ref_tokens,
        hyp_tokens,
        weights            = (0.25, 0.25, 0.25, 0.25),
        smoothing_function = SmoothingFunction().method1
    )

    # ── METEOR ────────────────────────────────────────────────
    # meteor = sum(
    #     meteor_score(refs, hyp)
    #     for hyp, refs in zip(hypotheses, references)
    # ) / len(hypotheses)
    meteor = sum(
        meteor_score(
            [r.lower().split() for r in refs],   # tokenize references
            hyp.lower().split()                 # tokenize hypothesis
        )
        for hyp, refs in zip(hypotheses, references)
    ) / len(hypotheses)

    
    # ── ROUGE-L ───────────────────────────────────────────────
    scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(
        max(scorer.score(ref, hyp)["rougeL"].fmeasure for ref in refs)
        for hyp, refs in zip(hypotheses, references)
    ) / len(hypotheses)

    # ── CIDEr ─────────────────────────────────────────────────
    cider = compute_cider(hypotheses, references)

    # ── Print results ─────────────────────────────────────────
    print("=" * 52)
    print("  EVALUATION RESULTS — Flickr8k Val Set")
    print("=" * 52)
    print(f"  Images evaluated : {len(image_paths):,}")
    print(f"  Checkpoint       : {CHECKPOINT}")
    print(f"{'─' * 52}")
    print(f"  BLEU-4  : {bleu4:.4f}   ({bleu4*100:.2f}%)")
    print(f"  METEOR  : {meteor:.4f}   ({meteor*100:.2f}%)")
    print(f"  ROUGE-L : {rouge_l:.4f}   ({rouge_l*100:.2f}%)")
    print(f"  CIDEr   : {cider:.4f}")
    print("=" * 52)

    # ── Sample predictions ────────────────────────────────────
    print("\n  SAMPLE PREDICTIONS (first 5):")
    print("─" * 52)
    for i in range(min(5, len(hypotheses))):
        print(f"\n  [{i+1}] Generated : {hypotheses[i]}")
        print(f"       Reference : {references[i][0]}")


if __name__ == "__main__":
    evaluate()