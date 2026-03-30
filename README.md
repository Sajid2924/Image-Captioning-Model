# Image Captioning with ResNet-50 + GPT-2

This project implements an **end-to-end image captioning system** that automatically generates natural language descriptions for images. It combines a pretrained **Convolutional Neural Network (CNN)** for visual feature extraction with a **GPT-2 language model built from scratch with loaded pretrained weights** for caption generation.

---

## 🧠 Architecture Overview

```
Input Image
    ↓
ResNet-50 (pretrained, fine-tuned)       → 2048-dim visual feature vector
    ↓
Projection Layer (trainable MLP)     → 10 × 768 visual prefix tokens
    ↓
GPT-2 Small (built from scratch)     → caption tokens generated autoregressively
    ↓
Generated Caption
```

### Components

- **Image Encoder** — Pretrained ResNet-50 with the final FC layer removed. Outputs a 2048-dimensional feature vector per image. Weights are fine-tuned during training, allowing the encoder to adapt its ImageNet features to the captioning task.
- **Projection Layer** — A 2-layer MLP (`2048 → 4864 → 7680`) with Tanh activation and Dropout(0.3). Maps visual features into GPT-2's embedding space as 10 learnable prefix tokens.
- **GPT-2 (from scratch)** — Full GPT-2 Small architecture (12 layers, 12 heads, 768 embedding dim, 117M parameters) implemented using only PyTorch primitives. Pretrained OpenAI weights are loaded manually by matching layer names.

---

## 📁 Project Structure

```
Model/
├── data/
│   ├── Images/                          ← Flickr8k images
│   ├── captions.txt                     ← Flickr8k captions
│   ├── train2014/                       ← COCO 2014 train images
│   ├── val2017/                         ← COCO 2017 val images
│   ├── annotations_trainval2014/
│   │   └── captions_train2014.json
│   └── annotations_trainval2017/
│       └── captions_val2017.json
├── checkpoints/
│   ├── best_model.pt                    ← best checkpoint (by val loss)
│   └── epoch_XX.pt                      ← last 2 epoch checkpoints
├── config.py                            ← all hyperparameters
├── encoder.py                           ← ResNet-50 image encoder
├── gpt2.py                              ← GPT-2 from scratch + weight loader
├── projection.py                        ← visual feature → GPT-2 prefix mapper
├── model.py                             ← full captioning model
├── dataset.py                           ← combined dataset loader
├── train.py                             ← training loop with early stopping
├── evaluate.py                          ← BLEU-4, METEOR, ROUGE-L, CIDEr metrics
├── predict.py                           ← caption generation for any image
├── sample_test.py                       ← quick CPU test (50 images, 3 epochs)
├── check_dataset.py                     ← verify all 3 dataset sources
└── requirements.txt
```

---

## 📦 Dataset

### Sources Used

| Dataset | Images | Captions | Split Used |
|---|---|---|---|
| Flickr8k | 8,091 | 40,455 | Train + Val |
| COCO 2014 train | 82,783 | 414,113 | Train + Val |
| COCO 2017 val | 5,000 | 25,014 | Train + Val |
| **Total** | **95,874** | **479,582** | — |

### Split Strategy

All 479,582 pairs are combined, shuffled with a fixed `seed=42` for reproducibility, and split:

```
Train set : 90%  →  431,623 pairs
Val   set : 10%  →   47,959 pairs
```

The fixed seed ensures the train/val split is **identical every run** and across all machines.

### Dataset Links

- **Flickr8k** → [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **COCO 2014** → [COCO Dataset](https://cocodataset.org/#download) (train2014 images + annotations)
- **COCO 2017** → [COCO Dataset](https://cocodataset.org/#download) (val2017 images + annotations)

---

## 📊 Evaluation Results

Evaluated on **250 images from COCO 2017 val set** using `best_model.pt` (epoch 3):

| Metric | Score |
|---|---|
| **BLEU-4** | 0.2812 (28.12%) |
| **METEOR** | 0.4394 (43.94%) |
| **ROUGE-L** | 0.5146 (51.46%) |
| **CIDEr** | 1.9546 |

### Sample Predictions

| Image | Generated Caption | Reference Caption |
|---|---|---|
| Dog jumping over water | *a dog is jumping into the water* | A dog leaps over a body of water |
| Kid kicking football | *a young boy kicking a soccer ball on a field* | A kid playing soccer on a grass field |

---

## ⚙️ Configuration (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| `encoder_frozen` | `False` | ResNet-50 unfrozen — fine-tuned end-to-end alongside the rest of the model |
| `prefix_length` | `10` | Number of visual prefix tokens fed to GPT-2 |
| `batch_size` | `16` | Training batch size |
| `num_epochs` | `50` | Max epochs (early stopping handles actual stop) |
| `learning_rate` | `3e-5` | AdamW learning rate |
| `weight_decay` | `1e-3` | L2 regularization |
| `warmup_steps` | `1000` | LR warmup steps |
| `mixed_precision` | `True` | FP16 training — saves ~40% VRAM |
| `early_stopping_patience` | `3` | Stop after 3 epochs with no val loss improvement |
| `temperature` | `0.7` | Sampling temperature at inference |
| `top_k` | `50` | Top-K filtering at inference |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Sajid2924/Image-Captioning-Model.git
cd Image-Captioning-Model
```

### 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
```

### 3. Set up dataset

Download datasets and place them in `data/` as shown in the project structure above.

Verify all sources load correctly:
```bash
python check_dataset.py
```

### 4. Quick CPU test (no GPU needed)

Trains on 50 images for 3 epochs — verifies the full pipeline works:
```bash
python sample_test.py
```

### 5. Full training

```bash
python train.py
```

Training will automatically stop when validation loss stops improving (early stopping). The best model is saved to `checkpoints/best_model.pt`.

### 6. Generate caption for any image

```bash
# Uses image_dog_water.jpg by default
python predict.py

# Custom image
python predict.py --image path/to/your/image.jpg

# Custom checkpoint
python predict.py --image photo.jpg --checkpoint checkpoints/best_model.pt
```

### 7. Evaluate

```bash
python evaluate.py
```

Runs BLEU-4, METEOR, ROUGE-L, and CIDEr on 250 COCO 2017 val images.

---

## 🔁 Training Details

### Training Loop

- **Optimizer** — AdamW with betas `(0.9, 0.95)`
- **Scheduler** — Cosine decay with linear warmup
- **Mixed Precision** — `torch.cuda.amp` (FP16) for memory efficiency
- **Gradient Clipping** — max norm = 1.0
- **Early Stopping** — stops when val loss doesn't improve by ≥ 0.001 for 3 consecutive epochs
- **Checkpointing** — saves `best_model.pt` + last 2 epoch checkpoints (auto-deletes older ones to save disk space)

### What Gets Trained

| Component | Parameters | Trained? |
|---|---|---|
| ResNet-50 encoder | 23,508,032 | ✅ Yes (fine-tuned) |
| Projection Layer | ~23M | ✅ Yes |
| GPT-2 Small | ~124M | ✅ Yes |
| **Total trainable** | **~147M** | ✅ |

### Training Progression (observed)

```
Epoch 1 :  Train loss 3.21  |  Val loss 2.74  ← warming up
Epoch 2 :  Train loss 2.60  |  Val loss 2.67  ← best model saved ★
Epoch 3 :  Train loss 2.39  |  Val loss 2.68  ← val rising (overfit signal)
```

---

## 🧩 GPT-2 From Scratch

The GPT-2 architecture is implemented entirely from scratch using `torch.nn` primitives:

- `CausalSelfAttention` — Multi-head masked self-attention with 12 heads
- `MLP` — Feed-forward block (768 → 3072 → 768) with GELU activation
- `TransformerBlock` — Pre-norm (LayerNorm → Attention → Residual → LayerNorm → MLP → Residual)
- `GPT2` — Stack of 12 transformer blocks with token + positional embeddings

Pretrained OpenAI weights are loaded by **manually mapping HuggingFace layer names** to our custom layer names, handling the Conv1D → Linear weight transposition.

---

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
pillow>=9.0.0
numpy>=1.24.0
tqdm>=4.65.0
nltk>=3.8.0
rouge-score>=0.1.2
```

---

## 💡 Key Design Decisions

- **Fine-tuned ResNet** — With ~96k images, the ResNet-50 encoder is fine-tuned end-to-end, allowing it to adapt its pretrained ImageNet features specifically to the captioning task and learn more caption-relevant visual representations.
- **Prefix tokens** — Visual features are projected into 10 "visual word" tokens prepended to the caption sequence. GPT-2 attends to these at every generation step.
- **Weight decay + early stopping** — The main overfitting countermeasures. With 479k pairs and 147M trainable params, the model memorizes quickly without regularization.
- **Seed=42 shuffle** — Ensures the same train/val split on every run and every machine, making results fully reproducible.
