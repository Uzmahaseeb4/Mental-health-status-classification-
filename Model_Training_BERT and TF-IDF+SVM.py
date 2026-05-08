#!/usr/bin/env python
# coding: utf-8

# # 🤖 Model Training Notebook
# ## Two Approaches: TF-IDF + SVM  ·  BERT Fine-Tuning
# 
# This notebook trains **two different models** on the same cleaned dataset produced by the preprocessing step (`preprocessed_dataset.csv`).
# 
# | | TF-IDF + SVM | BERT |
# |---|---|---|
# | **Speed** | Seconds–minutes | Minutes–hours |
# | **Hardware** | CPU only | GPU strongly recommended |
# | **Accuracy** | Good baseline | State-of-the-art |
# | **Deployment** | Tiny (`pickle` file) | Large (~400 MB) |
# | **Interpretability** | High (feature weights) | Low (black box) |
# 
# **Run order:** Execute all cells top to bottom. Start with Section A (TF-IDF + SVM) — it trains fast and gives you a working model immediately. Section B (BERT) can then run on GPU for maximum accuracy.
# 
# ---
# 

# ---
# # 📊 SECTION A — TF-IDF + SVM Training
# ### Classical machine-learning pipeline: fast, interpretable, CPU-only
# 
# **How it works:**
# 1. **TF-IDF Vectorizer** converts each text into a sparse numerical vector. Each dimension represents a word or bigram; the value reflects how important that word is in the document relative to the whole dataset.
# 2. **LinearSVC** (Support Vector Machine) finds the optimal decision boundary between classes in that high-dimensional space.
# 3. **CalibratedClassifierCV** wraps the SVM to add probability outputs (needed for Streamlit confidence scores).
# 
# > **When to use this model:** When you need a quick, deployable baseline — or when you don't have GPU access for BERT.
# 

# ### A1 · Install & import libraries
# All libraries used here come pre-installed in Google Colab. The `!pip install` line is a safety net.

# In[ ]:


# Install any missing packages (Colab usually has these already)
# If you need to install dependencies locally, run:
# pip install scikit-learn pandas numpy matplotlib seaborn

# ── Imports ──────────────────────────────────────────────────────────────────
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection          import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.svm                      import LinearSVC
from sklearn.pipeline                 import Pipeline
from sklearn.calibration              import CalibratedClassifierCV
from sklearn.metrics                  import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

print("✓ All libraries imported successfully")


# ### A2 · Configuration
# All tunable settings live here — edit this cell, then re-run the rest.
# 
# - **`DATA_PATH`** — path to your cleaned CSV (output of the preprocessing notebook).
# - **`TFIDF_PARAMS`** — controls the vocabulary. `ngram_range=(1,2)` means the model learns both single words *and* two-word phrases like "not good".
# - **`SVC_C_VALUES`** — the grid of regularisation strengths to search. Larger C = less regularisation = fits training data more tightly (risk of overfitting).
# 

# In[ ]:


# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("preprocessed_dataset.csv")   # clearly point to the balanced cleaned dataset
OUTPUT_DIR = Path("svm_output")
SEED       = 42
TEST_SPLIT = 0.10   # 10 % held out as final test  set
VAL_SPLIT  = 0.10   # 10 % held out as validation set

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── TF-IDF ────────────────────────────────────────────────────────────────────
TFIDF_PARAMS = dict(
    max_features  = 100_000,   # Vocabulary size cap
    ngram_range   = (1, 2),    # Unigrams + bigrams
    sublinear_tf  = True,      # log(1 + tf) dampens very frequent terms
    min_df        = 2,         # Ignore words in < 2 documents
    max_df        = 0.95,      # Ignore near-universal words (stop-word-like)
    analyzer      = "word",
    token_pattern = r"(?u)\b\w\w+\b",
)

# ── SVM ───────────────────────────────────────────────────────────────────────
SVC_C_VALUES = [0.01, 0.1, 1.0, 5.0, 10.0]   # Grid search candidates

print(f"Data path  : {DATA_PATH}")
print(f"Output dir : {OUTPUT_DIR}")
print(f"Seed       : {SEED}")


# ### A3 · Load & inspect the dataset
# We load the cleaned CSV produced by the preprocessing notebook.
# Required columns: **`clean_text`** (input) and **`target`** (label).
# 

# In[ ]:


df = pd.read_csv(DATA_PATH)
print(f"Shape   : {df.shape}")
print(f"Columns : {df.columns.tolist()}")

# Defensive checks
assert "clean_text" in df.columns, "ERROR: 'clean_text' column missing!"
assert "target"     in df.columns, "ERROR: 'target' column missing!"

df = df.dropna(subset=["clean_text"]).reset_index(drop=True)

texts   = df["clean_text"].tolist()
labels  = df["target"].tolist()
classes = sorted(set(labels))

print(f"\nNumber of classes : {len(classes)}")
print(f"Class labels      : {classes}")
print(f"\nClass distribution:")
dist = df["target"].value_counts().sort_index()
for cls, cnt in dist.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  Class {cls}: {cnt:>5} samples  ({pct:5.1f} %)  {bar}")

# Preview a sample
print("\nSample text (first row):")
print(df["clean_text"].iloc[0][:250])


# ### A4 · Stratified train / val / test split
# We split the data **before** fitting TF-IDF to prevent **data leakage** — the vocabulary must be learned only from training documents, not from validation or test data.
# 
# `stratify=labels` ensures every split has the same class ratio as the full dataset.
# 

# In[ ]:


# Step 1: carve out the test set (10 %)
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, labels,
    test_size=TEST_SPLIT, random_state=SEED, stratify=labels,
)

# Step 2: from remaining data, carve out validation (10 % of total)
val_frac = VAL_SPLIT / (1.0 - TEST_SPLIT)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_frac, random_state=SEED, stratify=y_temp,
)

print(f"Train samples : {len(X_train)}")
print(f"Val   samples : {len(X_val)}")
print(f"Test  samples : {len(X_test)}")
print("\n⚠️  TF-IDF vocabulary will be fitted ONLY on training data.")


# ### A5 · Build the sklearn Pipeline
# A `Pipeline` chains preprocessing + model together so that:
# - During grid-search, the vectorizer is re-fitted on each fold's training split only (no leakage).
# - At inference time, you call a single `pipeline.predict(text)` — no manual vectorization needed.
# 
# `CalibratedClassifierCV` adds **Platt scaling** on top of LinearSVC, which converts raw decision scores into proper probabilities (required for Streamlit confidence bars).
# 

# In[ ]:


# LinearSVC is very fast but has no built-in predict_proba().
# CalibratedClassifierCV adds probability output via Platt scaling.
base_svc = LinearSVC(
    max_iter     = 2000,
    class_weight = "balanced",   # Automatically handles class imbalance
    random_state = SEED,
)
calibrated_svc = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
    ("clf",   calibrated_svc),
])

print("Pipeline built:")
for name, step in pipeline.steps:
    print(f"  [{name}] → {type(step).__name__}")
print("\n✓ Ready for hyper-parameter tuning")


# ### A6 · Hyper-parameter tuning with GridSearchCV
# We search over the SVM regularisation parameter **C**:
# - Small C → strong regularisation → simpler decision boundary (may underfit)
# - Large C → weak regularisation → complex boundary (may overfit)
# 
# `StratifiedKFold` with 3 folds ensures each fold has balanced class proportions. The search is parallelised across all CPU cores (`n_jobs=-1`).
# 
# > ⏱️ This cell typically takes **1–3 minutes** on Colab CPU.
# 

# In[ ]:


param_grid = {
    "clf__estimator__C": SVC_C_VALUES,
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

grid_search = GridSearchCV(
    estimator  = pipeline,
    param_grid = param_grid,
    cv         = cv,
    scoring    = "f1_weighted",   # Primary metric: weighted F1
    n_jobs     = -1,              # Use all CPU cores
    verbose    = 1,
    refit      = True,            # Refit best params on full training data
)

print("Starting GridSearchCV…")
grid_search.fit(X_train, y_train)

best_C  = grid_search.best_params_["clf__estimator__C"]
best_cv = grid_search.best_score_

print(f"\n✓ Grid search complete")
print(f"  Best C               : {best_C}")
print(f"  Best CV F1 (weighted): {best_cv:.4f}")

# Show all results
cv_df = pd.DataFrame(grid_search.cv_results_)
print("\nCV results by C value:")
print(
    cv_df[["param_clf__estimator__C", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .to_string(index=False)
)

best_pipeline = grid_search.best_estimator_


# ### A7 · Validation set evaluation
# Before touching the final test set, we evaluate on the held-out validation set. This lets us confirm the model is working without "spending" our test data.
# 

# In[ ]:


y_val_pred = best_pipeline.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_f1  = f1_score(y_val, y_val_pred, average="weighted")

print(f"Validation Accuracy : {val_acc:.4f}  ({val_acc*100:.2f} %)")
print(f"Validation F1 (w)   : {val_f1:.4f}")
print("\nDetailed classification report:")
print(classification_report(y_val, y_val_pred, digits=4))


# ### A8 · Final test set evaluation *(touch only once!)*
# The test set is evaluated **exactly once** after all tuning decisions are made. This gives an honest, unbiased estimate of how the model will perform on new real-world data.
# 

# In[ ]:


y_test_pred = best_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1  = f1_score(y_test, y_test_pred, average="weighted")

print(f"Test Accuracy       : {test_acc:.4f}  ({test_acc*100:.2f} %)")
print(f"Test F1 (weighted)  : {test_f1:.4f}")
print("\nDetailed classification report:")
print(classification_report(y_test, y_test_pred, digits=4))


# ### A9 · Confusion matrix
# The confusion matrix shows which classes the model confuses most often. Rows = true labels, Columns = predicted labels. The diagonal is correct predictions.
# 

# In[ ]:


cm = confusion_matrix(y_test, y_test_pred, labels=classes)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    ax=ax, colorbar=True, cmap="Blues"
)
ax.set_title("TF-IDF + SVM — Confusion Matrix (Test Set)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved to svm_output/confusion_matrix.png")


# ### A10 · Top features per class
# One advantage of TF-IDF + SVM over BERT is **interpretability**. We can inspect which words/bigrams the model weighted most strongly for each class — giving human-readable insight into what drives predictions.
# 

# In[ ]:


tfidf_vocab = best_pipeline.named_steps["tfidf"].get_feature_names_out()
inner_svc   = best_pipeline.named_steps["clf"].calibrated_classifiers_[0].estimator
coef_matrix = inner_svc.coef_    # shape: (n_classes, n_features)
TOP_N       = 15

fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 5))
if len(classes) == 1:
    axes = [axes]

for i, cls in enumerate(inner_svc.classes_):
    top_idx    = np.argsort(coef_matrix[i])[-TOP_N:][::-1]
    top_words  = tfidf_vocab[top_idx]
    top_scores = coef_matrix[i][top_idx]

    print(f"\nClass {cls} — top {TOP_N} discriminating words/bigrams:")
    for w, s in zip(top_words, top_scores):
        print(f"  {w:<30} {s:+.4f}")

    axes[i].barh(range(TOP_N), top_scores[::-1], color="steelblue")
    axes[i].set_yticks(range(TOP_N))
    axes[i].set_yticklabels(top_words[::-1], fontsize=8)
    axes[i].set_title(f"Class {cls}", fontsize=11)
    axes[i].set_xlabel("SVM coefficient")

fig.suptitle("Top TF-IDF Features per Class", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_features_per_class.png", dpi=150, bbox_inches="tight")
plt.show()


# ### A11 · Save the trained pipeline
# We save the entire pipeline (TF-IDF + SVM) as a single `pickle` file. In Streamlit, you load it in one line and call `predict()` or `predict_proba()` directly on raw text — no preprocessing needed.
# 

# In[ ]:


import pickle

model_path = OUTPUT_DIR / "tfidf_svm_pipeline.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_pipeline, f)

classes_path = OUTPUT_DIR / "classes.json"
with open(classes_path, "w") as f:
    json.dump({"classes": classes}, f)

summary = {
    "model"              : "TF-IDF + LinearSVC (calibrated)",
    "dataset_rows"       : len(df),
    "best_C"             : best_C,
    "best_cv_f1_weighted": round(best_cv, 4),
    "val_accuracy"       : round(val_acc, 4),
    "val_f1_weighted"    : round(val_f1, 4),
    "test_accuracy"      : round(test_acc, 4),
    "test_f1_weighted"   : round(test_f1, 4),
}
with open(OUTPUT_DIR / "svm_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✓ Model saved!")
print(f"  Pipeline : {model_path}")
print(f"  Classes  : {classes_path}")
print(f"\nResults summary:")
print(json.dumps(summary, indent=2))

print("\n--- Streamlit quick-load snippet ---")
print("import pickle")
print(f"model = pickle.load(open('{model_path}', 'rb'))")
print("label = model.predict(['your text here'])[0]")
print("proba = model.predict_proba(['your text here'])[0]")


# ---
# # 🧠 SECTION B — BERT Fine-Tuning
# ### Deep learning pipeline: maximum accuracy, GPU required
# 
# **How it works:**
# 1. **BertTokenizer** converts each text into token IDs with special `[CLS]` and `[SEP]` tokens.
# 2. **BertForSequenceClassification** is a pre-trained 110M-parameter transformer. The encoder weights capture rich language understanding from pre-training on billions of words.
# 3. **Fine-tuning** — we train the full model (encoder + classification head) on our dataset for a few epochs, nudging the weights to specialise for our task.
# 
# > **Requirement:** A GPU is strongly recommended. On Colab, go to *Runtime → Change runtime type → GPU*.
# 

# ### B1 · Install & import libraries

# In[ ]:


# If you need to install dependencies locally, run:
# pip install transformers torch scikit-learn pandas tqdm

import os, json, random
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from tqdm import tqdm

print("✓ All libraries imported")
print(f"  PyTorch version      : {torch.__version__}")
print(f"  CUDA available       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU                  : {torch.cuda.get_device_name(0)}")


# ### B2 · Configuration
# Key parameters explained:
# - **`MAX_LEN=256`** — BERT supports up to 512 tokens, but 256 cuts memory usage in half with minimal accuracy loss on short-medium texts.
# - **`LR=2e-5`** — very small learning rate; BERT is already pre-trained and needs gentle nudging, not a hard reset.
# - **`EPOCHS=4`** — typically 3–5 epochs is enough for fine-tuning; more risks overfitting.
# - **`WARMUP_RATIO=0.1`** — the learning rate linearly ramps up for the first 10 % of training steps, then decays. This stabilises early training.
# 

# In[ ]:


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("preprocessed_dataset.csv")
OUTPUT_DIR  = Path("bert_output")
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
SEED        = 42

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"   # Pre-trained checkpoint from HuggingFace
MAX_LEN      = 256    # Token sequence length (256 balances speed vs coverage)
DROPOUT      = 0.3    # Dropout rate on the classification head

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16     # Reduce to 8 if you run out of GPU memory
EPOCHS       = 4
LR           = 2e-5   # Learning rate (very small — BERT is already pre-trained)
WEIGHT_DECAY = 0.01   # L2 regularisation applied to non-bias weights
WARMUP_RATIO = 0.1    # Fraction of steps used for LR warm-up

# ── Split ─────────────────────────────────────────────────────────────────────
TEST_SPLIT  = 0.10
VAL_SPLIT   = 0.10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU detected. Training will be slow on CPU.")


# ### B3 · Load dataset & split

# In[ ]:


df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["clean_text"]).reset_index(drop=True)

texts  = df["clean_text"].tolist()
labels = df["target"].tolist()
num_labels = len(set(labels))

print(f"Rows           : {len(df)}")
print(f"Classes        : {sorted(set(labels))}  ({num_labels} total)")
print(f"Distribution:\n{df['target'].value_counts().sort_index().to_string()}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, labels, test_size=TEST_SPLIT, random_state=SEED, stratify=labels,
)
val_frac = VAL_SPLIT / (1.0 - TEST_SPLIT)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_frac, random_state=SEED, stratify=y_temp,
)
print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")


# ### B4 · PyTorch Dataset & DataLoaders
# `TextClassificationDataset` is a thin wrapper that tokenizes one sample at a time inside `__getitem__`. This is more memory-efficient than pre-tokenizing all samples at once (which would hold a full `(N × 512)` tensor in RAM).
# 
# A `DataLoader` wraps the Dataset and handles batching, shuffling, and parallel loading (`num_workers`).
# 

# In[ ]:


class TextClassificationDataset(Dataset):
    """
    Tokenizes text samples on-the-fly (one at a time) to avoid
    loading a huge pre-tokenized matrix into memory.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length            = self.max_len,
            padding               = "max_length",
            truncation            = True,
            return_attention_mask = True,
            return_token_type_ids = True,
            return_tensors        = "pt",
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "token_type_ids" : enc["token_type_ids"].squeeze(0),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Load tokenizer & create datasets ─────────────────────────────────────────
print(f"Loading tokenizer: {MODEL_NAME} …")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_ds = TextClassificationDataset(X_train, y_train, tokenizer, MAX_LEN)
val_ds   = TextClassificationDataset(X_val,   y_val,   tokenizer, MAX_LEN)
test_ds  = TextClassificationDataset(X_test,  y_test,  tokenizer, MAX_LEN)

nw = min(4, os.cpu_count() or 1)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True)

print(f"✓ Datasets ready")
print(f"  Batches — Train: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")


# ### B5 · Load BERT model
# `BertForSequenceClassification` is a pre-trained BERT encoder + a randomly-initialised linear classification head on top. The head size is `hidden_size → num_labels` (768 → 5 for our dataset).
# 
# `ignore_mismatched_sizes=True` suppresses the expected warning that the pre-trained head (size 2) doesn't match our task (size N).
# 

# In[ ]:


model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels                   = num_labels,
    hidden_dropout_prob          = DROPOUT,
    attention_probs_dropout_prob = DROPOUT,
    ignore_mismatched_sizes      = True,
)
model = model.to(device)

total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Model loaded: {MODEL_NAME}")
print(f"  Total params     : {total_params:,}")
print(f"  Trainable params : {trainable_params:,}")


# ### B6 · Optimizer & learning rate scheduler
# - **AdamW** — Adam with decoupled weight decay (the standard for BERT fine-tuning).
# - **Weight decay** is applied only to weight matrices, NOT to bias terms or LayerNorm — this is the recommended BERT fine-tuning pattern.
# - **Linear warmup + decay scheduler** — LR ramps linearly from 0 to `LR` over the first 10 % of steps, then decays back to 0.
# 

# In[ ]:


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_params = [
    {
        "params"       : [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
        "weight_decay" : WEIGHT_DECAY,
    },
    {
        "params"       : [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay)],
        "weight_decay" : 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_params, lr=LR, eps=1e-8)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps  = warmup_steps,
    num_training_steps = total_steps,
)
print(f"Total training steps : {total_steps}")
print(f"Warmup steps         : {warmup_steps}  ({WARMUP_RATIO*100:.0f} %)")


# ### B7 · Training & evaluation helper functions
# `train_one_epoch` runs one full pass over the training data:
# - Gradient clipping at `max_norm=1.0` prevents exploding gradients.
# - Optional mixed-precision (`fp16`) halves memory usage on GPU.
# 
# `evaluate` runs inference without gradient computation (`@torch.no_grad`) — faster and uses less memory.
# 

# In[ ]:


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="  Training", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tids = batch["token_type_ids"].to(device)
        lbl  = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler:
            from torch.cuda.amp import autocast
            with autocast():
                out  = model(input_ids=ids, attention_mask=mask,
                             token_type_ids=tids, labels=lbl)
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids=ids, attention_mask=mask,
                        token_type_ids=tids, labels=lbl)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += out.loss.item()
        all_preds.extend(out.logits.argmax(-1).cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, device, split_name="Val"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc=f"  {split_name}", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tids = batch["token_type_ids"].to(device)
        lbl  = batch["label"].to(device)

        out = model(input_ids=ids, attention_mask=mask,
                    token_type_ids=tids, labels=lbl)

        total_loss += out.loss.item()
        all_preds.extend(out.logits.argmax(-1).cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, all_preds, all_labels

print("✓ Training functions defined")


# ### B8 · Training loop
# Each epoch: train → validate → save best checkpoint.
# 
# The **best checkpoint** is the model weights that achieved the highest weighted F1 on the validation set. This protects against overfitting in later epochs — we always deploy the best-seen weights, not the last-epoch weights.
# 

# In[ ]:


history = {"train_loss": [], "train_acc": [],
           "val_loss":   [], "val_acc":   [], "val_f1": []}
best_val_f1    = 0.0

# Enable mixed precision only if CUDA is available
scaler = None
if device.type == "cuda":
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    print("Mixed-precision (fp16) enabled")

print(f"\n{'='*55}")
print(f"  TRAINING  ({EPOCHS} epochs,  batch_size={BATCH_SIZE})")
print(f"{'='*55}")

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
    vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, device, "Val")

    history["train_loss"].append(round(tr_loss, 4))
    history["train_acc" ].append(round(tr_acc,  4))
    history["val_loss"  ].append(round(vl_loss, 4))
    history["val_acc"   ].append(round(vl_acc,  4))
    history["val_f1"    ].append(round(vl_f1,   4))

    print(f"  Train — loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
    print(f"  Val   — loss: {vl_loss:.4f}  acc: {vl_acc:.4f}  f1(w): {vl_f1:.4f}")

    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(BEST_MODEL_DIR))
        tokenizer.save_pretrained(str(BEST_MODEL_DIR))
        print(f"  ✓ Best model saved  (val_f1 = {best_val_f1:.4f})")

print(f"\n✓ Training complete. Best val F1 = {best_val_f1:.4f}")


# ### B9 · Training curves
# Plot loss and accuracy over epochs to check for overfitting (val loss rising while train loss falls).

# In[ ]:


epochs_range = range(1, EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, history["train_loss"], "o-", label="Train loss")
ax1.plot(epochs_range, history["val_loss"],   "s-", label="Val loss")
ax1.set_xlabel("Epoch");  ax1.set_ylabel("Loss")
ax1.set_title("Loss per Epoch");  ax1.legend()

ax2.plot(epochs_range, history["train_acc"], "o-", label="Train acc")
ax2.plot(epochs_range, history["val_acc"],   "s-", label="Val acc")
ax2.plot(epochs_range, history["val_f1"],    "^--", label="Val F1")
ax2.set_xlabel("Epoch");  ax2.set_ylabel("Score")
ax2.set_title("Accuracy & F1 per Epoch");  ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved to bert_output/training_curves.png")


# ### B10 · Final test evaluation (best checkpoint)
# Load the best-saved checkpoint and evaluate on the held-out test set. This is the number to report.
# 

# In[ ]:


best_model = BertForSequenceClassification.from_pretrained(str(BEST_MODEL_DIR))
best_model = best_model.to(device)

ts_loss, ts_acc, ts_f1, ts_preds, ts_labels = evaluate(
    best_model, test_loader, device, "Test"
)

print(f"Test loss      : {ts_loss:.4f}")
print(f"Test accuracy  : {ts_acc:.4f}  ({ts_acc*100:.2f} %)")
print(f"Test F1 (w)    : {ts_f1:.4f}")
print("\nClassification report:")
print(classification_report(ts_labels, ts_preds, digits=4))

cm = confusion_matrix(ts_labels, ts_preds)
fig, ax = plt.subplots(figsize=(7, 6))
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(cm, display_labels=sorted(set(labels))).plot(
    ax=ax, colorbar=True, cmap="Blues"
)
ax.set_title("BERT — Confusion Matrix (Test Set)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bert_confusion_matrix.png", dpi=150)
plt.show()


# ### B11 · Save results & inference helper
# The model is already saved via `save_pretrained()` above. Here we save the training history and final metrics, and define a convenience inference function for Streamlit.
# 

# In[ ]:


# Save training history
with open(OUTPUT_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# Save test results
results = {
    "model_name"       : MODEL_NAME,
    "test_loss"        : round(ts_loss, 4),
    "test_accuracy"    : round(ts_acc,  4),
    "test_f1_weighted" : round(ts_f1,   4),
    "best_val_f1"      : round(best_val_f1, 4),
    "epochs"           : EPOCHS,
    "max_len"          : MAX_LEN,
    "batch_size"       : BATCH_SIZE,
    "learning_rate"    : LR,
    "num_labels"       : num_labels,
}
with open(OUTPUT_DIR / "bert_test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Artifacts saved:")
print(f"  Best model : {BEST_MODEL_DIR}/")
print(f"  History    : {OUTPUT_DIR}/training_history.json")
print(f"  Results    : {OUTPUT_DIR}/bert_test_results.json")
print("\nResults summary:")
print(json.dumps(results, indent=2))


# ### B12 · BERT inference helper (for Streamlit)
# Copy this function into your Streamlit app to load and run the fine-tuned BERT model.
# 

# In[ ]:


def bert_predict(texts: list, model_dir: str, max_len: int = 256, batch_size: int = 32):
    """
    Load the fine-tuned BERT model and return predictions + probabilities.

    Parameters
    ----------
    texts      : list of str — cleaned text samples
    model_dir  : path to saved model directory (e.g. 'bert_output/best_model')
    max_len    : must match the value used during training
    batch_size : inference batch size

    Returns
    -------
    preds : list of int — predicted class for each input
    probs : list of list[float] — softmax class probabilities
    """
    device_inf = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizer.from_pretrained(model_dir)
    mdl = BertForSequenceClassification.from_pretrained(model_dir).to(device_inf)
    mdl.eval()

    all_preds, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc   = tok(chunk, max_length=max_len, padding="max_length",
                    truncation=True, return_attention_mask=True,
                    return_token_type_ids=True, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(
                input_ids      = enc["input_ids"].to(device_inf),
                attention_mask = enc["attention_mask"].to(device_inf),
                token_type_ids = enc["token_type_ids"].to(device_inf),
            ).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(probs.argmax(axis=-1).tolist())
        all_probs.extend(probs.tolist())

    return all_preds, all_probs


# ── Quick test ────────────────────────────────────────────────────────────────
sample_texts = ["i feel really hopeless today", "everything is going great"]
preds, probs = bert_predict(sample_texts, str(BEST_MODEL_DIR), max_len=MAX_LEN)
for txt, pred, prob in zip(sample_texts, preds, probs):
    print(f"Text  : {txt}")
    print(f"Pred  : class {pred}   Proba: {[f'{p:.3f}' for p in prob]}")
    print()


# ---
# # 📈 SECTION C — Model Comparison
# Compare both models side-by-side. Run this after both Section A and Section B are complete.
# 

# In[ ]:


# Load results
with open(Path("svm_output") / "svm_summary.json") as f:
    svm_res = json.load(f)
with open(OUTPUT_DIR / "bert_test_results.json") as f:
    bert_res = json.load(f)

compare = pd.DataFrame({
    "Model"        : ["TF-IDF + SVM", "BERT (fine-tuned)"],
    "Test Accuracy": [svm_res["test_accuracy"],  bert_res["test_accuracy"]],
    "Test F1 (w)"  : [svm_res["test_f1_weighted"], bert_res["test_f1_weighted"]],
})
compare = compare.set_index("Model")
print("Model Comparison:")
print(compare.to_string())

# Bar chart
ax = compare.plot(kind="bar", figsize=(7, 4), rot=0, color=["steelblue", "coral"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("TF-IDF + SVM  vs  BERT — Test Set Performance")
ax.legend(loc="lower right")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
                ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(Path("model_comparison.png"), dpi=150)
plt.show()
print("\n✓ Comparison chart saved.")

