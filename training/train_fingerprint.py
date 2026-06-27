"""
Multi-LLM Fingerprinter Training Script — Novelty #1
Run this on Google Colab with T4 GPU (free tier works).

What it does:
  - Downloads RAID dataset (11 AI models + human text, labeled per model)
  - Trains a RoBERTa-base 8-class classifier
  - Saves the best checkpoint to models/fingerprint/best/
  - Upload that folder back into your project

Colab Instructions:
  1. Runtime → Change runtime type → T4 GPU
  2. Upload this file to Colab
  3. Run all cells
  4. Download models/fingerprint/best.zip
  5. Extract into e:\\Projects\\AI-Detection-main\\AI-Detection-main\\models\\fingerprint\\best\\
"""

# ── Cell 1: Install dependencies ─────────────────────────────────────────────
# !pip uninstall torchvision -y
# !pip install transformers datasets accelerate scikit-learn evaluate -q

import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score

# ── Configuration ─────────────────────────────────────────────────────────────

LABELS = ["Human", "ChatGPT", "GPT-4", "Claude", "Gemini", "Llama", "Mistral", "DeepSeek"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

BASE_MODEL = "roberta-base"
OUTPUT_DIR = "./models/fingerprint"
BEST_DIR = f"{OUTPUT_DIR}/best"
MAX_LENGTH = 512
MAX_TRAIN_SAMPLES = 80_000
MAX_EVAL_SAMPLES = 16_000
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# ── Cell 2: Load and prepare RAID dataset ────────────────────────────────────

def load_raid_data() -> tuple[list[str], list[int]]:
    """
    Load RAID dataset and map model names to our label space.
    RAID has text from: human, chatgpt, gpt4, claude, gemini, llama, mistral, etc.
    """
    print("Loading RAID dataset...")
    ds = load_dataset("liamdugan/raid", split="train")
    print(f"  Raw RAID size: {len(ds)}")

    # RAID model name → our label mapping
    model_map = {
        "human": "Human",
        "chatgpt": "ChatGPT",
        "gpt-3.5-turbo": "ChatGPT",
        "gpt-4": "GPT-4",
        "gpt4": "GPT-4",
        "claude": "Claude",
        "claude-instant": "Claude",
        "claude-2": "Claude",
        "gemini": "Gemini",
        "bard": "Gemini",
        "llama": "Llama",
        "llama-2": "Llama",
        "llama-3": "Llama",
        "mistral": "Mistral",
        "deepseek": "DeepSeek",
    }

    texts, labels = [], []
    class_counts = {i: 0 for i in range(NUM_LABELS)}
    MAX_PER_CLASS = 15_000  # Cap at 15k per class to save RAM and balance dataset

    for row in ds:
        # Stop early if we have enough data for all classes!
        if all(c >= MAX_PER_CLASS for c in class_counts.values()):
            break

        # RAID schema: text, model (or generator), label
        text = row.get("text") or row.get("generation", "")
        generator = (row.get("model") or row.get("generator") or "human").lower()

        # Map generator name to our label
        label_name = None
        for key, mapped in model_map.items():
            if key in generator:
                label_name = mapped
                break
        if label_name is None:
            continue

        label_id = LABEL2ID[label_name]

        # Skip if we already have enough of this specific class
        if class_counts[label_id] >= MAX_PER_CLASS:
            continue

        if text and len(text.strip()) > 50:
            texts.append(text[:4096])  # Truncate for tokenizer
            labels.append(label_id)
            class_counts[label_id] += 1

    print(f"  Filtered samples: {len(texts)}")
    print(f"  Label distribution: {class_counts}")
    return texts, labels


# Load data (HC3 removed — HuggingFace no longer supports its loading script)
raid_texts, raid_labels = load_raid_data()

all_texts = raid_texts
all_labels = raid_labels

# Shuffle and subsample
indices = np.random.permutation(len(all_texts))
all_texts = [all_texts[i] for i in indices]
all_labels = [all_labels[i] for i in indices]

# Build HuggingFace dataset
full_ds = Dataset.from_dict({"text": all_texts, "label": all_labels})
split = full_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"].select(range(min(MAX_TRAIN_SAMPLES, len(split["train"]))))
eval_ds = split["test"].select(range(min(MAX_EVAL_SAMPLES, len(split["test"]))))

print(f"\nFinal train size: {len(train_ds)}")
print(f"Final eval size:  {len(eval_ds)}")

# ── Cell 3: Tokenize ─────────────────────────────────────────────────────────

print(f"\nLoading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])
train_ds.set_format("torch")
eval_ds.set_format("torch")

# ── Cell 4: Load model ───────────────────────────────────────────────────────

print(f"\nLoading model: {BASE_MODEL} with {NUM_LABELS} labels")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Cell 5: Training ─────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(label_ids, preds)
    f1_macro = f1_score(label_ids, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(label_ids, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=True,  # Use fp16 on T4 (T4 does not support bf16)
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nStarting training...")
trainer.train()

# ── Cell 6: Save best model ──────────────────────────────────────────────────

print(f"\nSaving best model to {BEST_DIR}...")
trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)

# Print final eval metrics
metrics = trainer.evaluate()
print("\n=== Final Evaluation ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print(f"\n✅ Done! Download {BEST_DIR}/ and copy to your project:")
print(f"   <project>/models/fingerprint/best/")

# ── Cell 7: Zip for download ─────────────────────────────────────────────────
import shutil
shutil.make_archive("fingerprint_best", "zip", OUTPUT_DIR, "best")
print("\n✅ fingerprint_best.zip ready for download!")
print("   Files menu → right-click fingerprint_best.zip → Download")
