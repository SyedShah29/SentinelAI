"""
Stylometry Analyzer — 4th Detection Signal (Novelty #3)

Uses linguistic features that AI writing exhibits systematically:
  - Function word overuse (connectives like "however", "moreover")
  - Type-Token Ratio (vocabulary richness)
  - Sentence length distribution (AI clusters around 15-25 words)
  - Passive voice ratio (AI overuses passive constructions)
  - Punctuation entropy (AI uses limited punctuation variety)
  - Dependency parse features (subordinate clause depth)

These features are language-model-agnostic — they don't depend on PPL.
They capture the *rhetorical fingerprint* of AI text, not its predictability.
"""

import re
import math
import string
from collections import Counter
from typing import Any

import config


class StylometryAnalyzer:
    """
    Analyzes text for stylometric features characteristic of AI-generated writing.

    Requires spaCy's 'en_core_web_sm' model.
    Install: python -m spacy download en_core_web_sm
    """

    def __init__(self):
        self._nlp = None  # Lazy-loaded

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(config.STYLOMETRY_SPACY_MODEL)
            except OSError:
                raise RuntimeError(
                    f"spaCy model '{config.STYLOMETRY_SPACY_MODEL}' not found. "
                    f"Run: python -m spacy download {config.STYLOMETRY_SPACY_MODEL}"
                )
        return self._nlp

    def analyze(self, text: str, sentences: list[str]) -> dict[str, Any]:
        """
        Run full stylometric analysis on text.

        Returns:
            dict with:
                - ai_probability: normalized [0, 1] where 1 = likely AI
                - features: detailed feature dict
                - flagged_phrases: AI connectives found in text
        """
        try:
            doc = self.nlp(text[:50000])  # Cap at 50k chars for speed
        except Exception as e:
            return self._fallback_result(f"spaCy error: {e}")

        features = {}

        # ── Feature 1: Connective / transitional phrase overuse ────────────
        text_lower = text.lower()
        ai_connectives = config.STYLOMETRY_AI_CONNECTIVES
        flagged = [phrase for phrase in ai_connectives if phrase in text_lower]
        word_count = max(len(text.split()), 1)
        connective_rate = len(flagged) / (word_count / 100)  # per 100 words
        features["connective_rate"] = round(connective_rate, 4)
        features["flagged_connectives"] = sorted(flagged)

        # ── Feature 2: Type-Token Ratio (vocabulary richness) ──────────────
        tokens = [t.lower_ for t in doc if t.is_alpha and not t.is_stop]
        if tokens:
            ttr = len(set(tokens)) / len(tokens)
        else:
            ttr = 0.5
        # Corrected TTR (CTTR) — normalized for text length
        cttr = ttr / math.sqrt(2 * max(len(tokens), 1))
        features["type_token_ratio"] = round(ttr, 4)
        features["cttr"] = round(cttr, 4)

        # ── Feature 3: Sentence length statistics ──────────────────────────
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sent_lengths:
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            # AI sentences cluster around 15-25 words; compute how many are in that range
            in_ai_range = sum(1 for l in sent_lengths if 14 <= l <= 26)
            ai_range_ratio = in_ai_range / len(sent_lengths)
        else:
            avg_sent_len = 0.0
            ai_range_ratio = 0.0
        features["avg_sentence_length"] = round(avg_sent_len, 2)
        features["ai_sentence_length_ratio"] = round(ai_range_ratio, 4)

        # ── Feature 4: Passive voice ratio ─────────────────────────────────
        passive_count = sum(
            1 for token in doc
            if token.dep_ == "nsubjpass" or token.dep_ == "auxpass"
        )
        sentence_count = max(len(list(doc.sents)), 1)
        passive_ratio = passive_count / sentence_count
        features["passive_ratio"] = round(passive_ratio, 4)

        # ── Feature 5: Punctuation entropy ─────────────────────────────────
        punct_chars = [c for c in text if c in string.punctuation]
        if punct_chars:
            punct_counts = Counter(punct_chars)
            total_punct = sum(punct_counts.values())
            punct_entropy = -sum(
                (count / total_punct) * math.log2(count / total_punct)
                for count in punct_counts.values()
                if count > 0
            )
        else:
            punct_entropy = 0.0
        # Normalize: max theoretical entropy for ~15 punct types = log2(15) ≈ 3.9
        punct_entropy_norm = min(punct_entropy / 4.0, 1.0)
        features["punctuation_entropy"] = round(punct_entropy, 4)
        features["punctuation_entropy_norm"] = round(punct_entropy_norm, 4)

        # ── Feature 6: Exclamation / question mark usage ───────────────────
        exclamation_rate = text.count("!") / (word_count / 100)
        question_rate = text.count("?") / (word_count / 100)
        em_dash_rate = (text.count("—") + text.count("–")) / (word_count / 100)
        features["exclamation_rate"] = round(exclamation_rate, 4)
        features["question_rate"] = round(question_rate, 4)
        features["em_dash_rate"] = round(em_dash_rate, 4)

        # ── Feature 7: Named entity density ────────────────────────────────
        ent_count = len(doc.ents)
        ent_density = ent_count / max(sentence_count, 1)
        features["entity_density"] = round(ent_density, 4)

        # ── Combine into AI probability score ──────────────────────────────
        ai_prob = self._compute_ai_probability(features)

        return {
            "ai_probability": round(ai_prob, 4),
            "features": features,
            "flagged_connectives": flagged,
        }

    def _compute_ai_probability(self, features: dict) -> float:
        """
        Combine stylometric features into a single AI probability.

        Uses a weighted linear combination of normalized feature scores.
        Each component produces a [0, 1] AI-likelihood score.
        """
        scores = []

        # Connective overuse → higher = more AI-like
        # Threshold: >2 connectives per 100 words is AI-like
        conn_rate = features.get("connective_rate", 0)
        conn_score = min(conn_rate / 4.0, 1.0)
        scores.append((conn_score, 0.30))

        # AI sentence length ratio → higher = more AI-like
        ai_len_ratio = features.get("ai_sentence_length_ratio", 0)
        scores.append((ai_len_ratio, 0.20))

        # Passive ratio → higher = more AI-like
        # Human: ~0.1, AI: ~0.25+
        passive = features.get("passive_ratio", 0)
        passive_score = min(passive / 0.4, 1.0)
        scores.append((passive_score, 0.15))

        # Punctuation entropy → LOWER = more AI-like (AI uses fewer punct types)
        # Low variety of punctuation is AI-characteristic
        punct_norm = features.get("punctuation_entropy_norm", 0.5)
        punct_ai_score = 1.0 - punct_norm  # Invert: low entropy = more AI
        scores.append((punct_ai_score, 0.15))

        # Em dash rate → LOWER = more AI (AI rarely uses em dashes)
        em_dash = features.get("em_dash_rate", 0)
        em_dash_ai = max(0.0, 1.0 - em_dash * 3)  # Penalize for em dashes
        scores.append((em_dash_ai, 0.10))

        # Type-token ratio → AI has slightly inflated TTR at medium lengths
        # This is noisy, low weight
        ttr = features.get("type_token_ratio", 0.5)
        ttr_ai = min(max((ttr - 0.5) * 2, 0), 1)  # AI TTR tends > 0.5
        scores.append((ttr_ai, 0.10))

        # Weighted average
        total_weight = sum(w for _, w in scores)
        ai_prob = sum(s * w for s, w in scores) / total_weight
        return max(0.0, min(1.0, ai_prob))

    def _fallback_result(self, error_msg: str = "") -> dict:
        return {
            "ai_probability": 0.5,
            "features": {},
            "flagged_connectives": [],
            "error": error_msg,
        }
