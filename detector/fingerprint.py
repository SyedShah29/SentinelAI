"""
Multi-LLM Fingerprinter — Novelty #1

Identifies WHICH AI model generated the text (not just Human vs AI).

Supported labels: Human, ChatGPT, GPT-4, Claude, Gemini, Llama, Mistral, DeepSeek

Architecture:
  - RoBERTa-base fine-tuned for 8-class classification
  - Trained on RAID dataset (which has per-model labels)
  - Returns probability distribution over all 8 classes
  - Gracefully returns None if no fine-tuned model is found

Training: Run training/train_fingerprint.py on Colab.
"""

import torch
import torch.nn.functional as F
from typing import Any

import config


class LLMFingerprinter:
    """
    Multi-class classifier that identifies the source AI model.

    Returns:
        - top_label: most likely source ("Human", "GPT-4", etc.)
        - top_probability: confidence in top prediction
        - distribution: probability over all 8 classes
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load()

    def _load(self):
        """Load the fine-tuned fingerprint classifier if it exists."""
        checkpoint = config.FINGERPRINT_CHECKPOINT
        if not checkpoint:
            print("  ℹ️  Fingerprint model not found — multi-LLM identification disabled.")
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading LLM fingerprinter from {checkpoint}...")
            self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint,
                num_labels=config.FINGERPRINT_NUM_LABELS,
            )
            self._model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self.device)
            self._loaded = True
            print("  [OK] LLM fingerprinter loaded")
        except Exception as e:
            print(f"  [WARN] Failed to load fingerprint model: {e}")
            self._loaded = False

    @property
    def is_available(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> dict[str, Any] | None:
        """
        Predict which AI model generated the text.

        Args:
            text: Input text to classify

        Returns:
            dict with top_label, top_probability, distribution
            or None if model not available
        """
        if not self._loaded:
            return None

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.FINGERPRINT_TRAINING_CONFIG.get("max_length", 512),
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

        labels = config.FINGERPRINT_LABELS
        distribution = {
            label: round(prob, 4)
            for label, prob in zip(labels, probs)
        }

        top_idx = max(range(len(probs)), key=lambda i: probs[i])
        top_label = labels[top_idx]
        top_probability = probs[top_idx]

        # Rank non-Human classes for the "most likely AI source" ranking
        ai_classes = [
            (label, prob)
            for label, prob in distribution.items()
            if label != "Human"
        ]
        ai_ranking = sorted(ai_classes, key=lambda x: x[1], reverse=True)

        return {
            "top_label": top_label,
            "top_probability": round(top_probability, 4),
            "distribution": distribution,
            "ai_source_ranking": [
                {"model": label, "probability": round(prob, 4)}
                for label, prob in ai_ranking
            ],
        }
