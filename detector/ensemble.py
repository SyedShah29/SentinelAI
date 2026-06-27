"""
Ensemble Combiner — Aggregates signals from all detection components.

SentinelAI multi-component detection pipeline:
  - Perplexity (how predictable is the text?)
  - Burstiness (how variable is the perplexity across sentences?)
  - Classifier (what does a fine-tuned transformer say?)
  - Token attribution (which words drive the AI/Human prediction?)
  - Stylometry (what does the writing STYLE say?)
  - MC Dropout uncertainty (how confident is the classifier?)
  - Domain-aware calibration
  - Multi-LLM fingerprinting

Key improvements:
  - Dynamic weights based on classifier training status
  - Consistent per-sentence probability mapping
  - Confidence tiers (high/moderate/low)
  - Trinary classification (Human/AI/Mixed)
  - Calibrated output mapping (biased toward false negatives)
  - Domain-specific perplexity thresholds
  - Bayesian uncertainty via MC Dropout
"""

import math
import numpy as np

import config
from detector.perplexity import PerplexityEngine
from detector.burstiness import BurstinessAnalyzer
from detector.classifier import TextClassifier
from detector.preprocessor import preprocess
from detector.stylometry import StylometryAnalyzer
from detector.domain_classifier import DomainClassifier
from detector.fingerprint import LLMFingerprinter


class EnsembleDetector:
    """
    Multi-component AI text detector.

    Combines signals:
        1. Perplexity (how predictable is the text?)
        2. Burstiness (how variable is the perplexity across sentences?)
        3. Classifier (what does a fine-tuned DeBERTa/Qwen say?)
        4. Stylometry (what does writing style say?) [Novelty #3]
        5. MC Dropout uncertainty [Novelty #7]
        6. Domain-aware calibration [Novelty #8]
        7. Multi-LLM fingerprinting [Novelty #1]
    """

    def __init__(self, classifier_path: str | None = None):
        """
        Initialize all detection components.

        Args:
            classifier_path: Optional path to fine-tuned classifier checkpoint.
        """
        self.perplexity_engine = PerplexityEngine()
        self.burstiness_analyzer = BurstinessAnalyzer()
        self.classifier = TextClassifier(model_path=classifier_path)

        # Novelty #3: Stylometry
        self._stylo: StylometryAnalyzer | None = None
        if config.STYLOMETRY_ENABLED:
            try:
                self._stylo = StylometryAnalyzer()
                # Trigger lazy load at startup (avoids cold start during first analysis)
                _ = self._stylo.nlp
                print("  [OK] Stylometry analyzer ready")
            except Exception as e:
                print(f"  [WARN] Stylometry disabled: {e}")
                self._stylo = None

        # Novelty #8: Domain classifier
        self._domain_clf = DomainClassifier() if config.DOMAIN_ENABLED else None

        # Novelty #1: Multi-LLM fingerprinter
        self._fingerprinter = LLMFingerprinter()

        # Use appropriate weights based on classifier status
        if self.classifier.is_fine_tuned:
            self.weights = dict(config.ENSEMBLE_WEIGHTS_TRAINED)
        else:
            self.weights = dict(config.ENSEMBLE_WEIGHTS_UNTRAINED)

        # Add stylometry weight if available
        if self._stylo is not None and self.classifier.is_fine_tuned:
            # Redistribute 10% from perplexity + burstiness to stylometry
            self.weights["stylometry"] = config.STYLOMETRY_WEIGHT
            factor = 1.0 - config.STYLOMETRY_WEIGHT
            self.weights["perplexity"] = round(self.weights["perplexity"] * factor, 3)
            self.weights["burstiness"] = round(self.weights["burstiness"] * factor, 3)
            self.weights["classifier"] = round(self.weights["classifier"] * factor, 3)

    def analyze(self, text: str) -> dict:
        """
        Run full detection pipeline on input text.

        Returns dict with:
            - verdict: "Human", "AI", or "Mixed"
            - ai_probability: final ensemble score [0, 1]
            - confidence: "high", "moderate", or "low"
            - confidence_category: detailed confidence label
            - uncertainty: MC Dropout uncertainty info [Novelty #7]
            - components: per-component scores
            - sentences: per-sentence analysis for highlighting
            - metadata: preprocessing metadata
            - domain: detected domain [Novelty #8]
            - fingerprint: source model prediction [Novelty #1]
        """
        # Step 1: Preprocess
        processed = preprocess(text)
        full_text = processed["full_text"]
        sentences = processed["sentences"]

        if not sentences:
            return self._empty_result()

        # Step 2: Domain detection (calibrates PPL thresholds)
        domain_info = {"domain": "general", "ppl_threshold": config.PERPLEXITY_THRESHOLD_AI,
                       "sigmoid_k": config.PERPLEXITY_SIGMOID_K, "confidence": 0.3}
        if self._domain_clf:
            domain_info = self._domain_clf.classify(full_text)
            # Inject domain-specific parameters into perplexity engine
            self.perplexity_engine._domain_threshold = domain_info["ppl_threshold"]
            self.perplexity_engine._domain_k = domain_info["sigmoid_k"]

        # Step 3: Run each component
        ppl_result = self.perplexity_engine.score_text(full_text, sentences)
        burst_result = self.burstiness_analyzer.score(ppl_result["sentence_ppls"])
        clf_result = self.classifier.score_text(full_text, sentences)

        # Step 4: Stylometry [Novelty #3]
        stylo_result = None
        if self._stylo is not None:
            try:
                stylo_result = self._stylo.analyze(full_text, sentences)
            except Exception as e:
                print(f"  [WARN] Stylometry analysis failed: {e}")

        # Step 5: MC Dropout Uncertainty [Novelty #7]
        mc_result = None
        if config.MC_DROPOUT_ENABLED and self.classifier.is_fine_tuned:
            try:
                mc_result = self.classifier.predict_with_uncertainty(full_text)
                # Use MC mean as improved classifier probability estimate
                clf_result["ai_probability"] = mc_result["mean_ai_probability"]
            except Exception as e:
                print(f"  [WARN] MC Dropout failed: {e}")

        # Step 6: Weighted ensemble
        w = self.weights
        ensemble_prob = (
            w["perplexity"] * ppl_result["ai_probability"]
            + w["burstiness"] * burst_result["ai_probability"]
            + w["classifier"] * clf_result["ai_probability"]
        )
        if stylo_result and "stylometry" in w:
            ensemble_prob += w["stylometry"] * stylo_result["ai_probability"]

        ensemble_prob = max(0.0, min(1.0, ensemble_prob))

        # Step 7: Per-sentence scores
        sentence_scores = self._compute_sentence_scores(
            sentences,
            ppl_result["sentence_ppls"],
            clf_result["sentence_ai_probs"],
            clf_result.get("sentence_attributions", []),
        )

        # Step 8: Verdict + Confidence
        verdict = self._determine_verdict(ensemble_prob, sentence_scores)
        confidence, confidence_category = self._determine_confidence(
            ensemble_prob, verdict, mc_result
        )

        # Step 9: Multi-LLM Fingerprinting [Novelty #1]
        fingerprint = None
        if self._fingerprinter.is_available:
            try:
                fingerprint = self._fingerprinter.predict(full_text)
            except Exception as e:
                print(f"  [WARN] Fingerprint prediction failed: {e}")

        # Assemble components dict
        components: dict = {
            "perplexity": {
                "global_ppl": round(ppl_result["global_ppl"], 2),
                "ai_probability": round(ppl_result["ai_probability"], 4),
            },
            "burstiness": {
                "score": round(burst_result["burstiness"], 4),
                "ppl_std": round(burst_result["ppl_std"], 2),
                "ppl_mean": round(burst_result["ppl_mean"], 2),
                "ai_probability": round(burst_result["ai_probability"], 4),
                "features": burst_result.get("features", {}),
            },
            "classifier": {
                "ai_probability": round(clf_result["ai_probability"], 4),
                "is_fine_tuned": clf_result["is_fine_tuned"],
            },
        }
        if stylo_result:
            components["stylometry"] = {
                "ai_probability": round(stylo_result["ai_probability"], 4),
                "features": stylo_result.get("features", {}),
                "flagged_connectives": stylo_result.get("flagged_connectives", []),
            }

        return {
            "verdict": verdict,
            "ai_probability": round(ensemble_prob, 4),
            "confidence": confidence,
            "confidence_category": confidence_category,
            "uncertainty": mc_result,
            "components": components,
            "sentences": sentence_scores,
            "domain": domain_info.get("domain", "general"),
            "domain_info": domain_info,
            "fingerprint": fingerprint,
            "metadata": {
                "original_length": processed.get("original_length", len(text)),
                "num_sentences": len(sentences),
                "language": processed.get("language", "unknown"),
                "homoglyphs_detected": processed.get("homoglyphs_detected", False),
                "classifier_trained": self.classifier.is_fine_tuned,
                "ensemble_weights": dict(self.weights),
                "domain": domain_info.get("domain", "general"),
                "stylometry_available": self._stylo is not None,
                "fingerprint_available": self._fingerprinter.is_available,
            },
        }

    def _compute_sentence_scores(
        self,
        sentences: list[str],
        sentence_ppls: list[float],
        sentence_clf_probs: list[float],
        sentence_attributions: list[dict] | None = None,
    ) -> list[dict]:
        """
        Build per-sentence analysis for UI highlighting.

        Uses the SAME sigmoid mapping as the global perplexity score
        for consistency. Now also includes token-level attribution data.
        """
        results = []
        clf_weight = 0.6 if self.classifier.is_fine_tuned else 0.0
        ppl_weight = 1.0 - clf_weight

        if sentence_attributions is None:
            sentence_attributions = []

        for i, sentence in enumerate(sentences):
            ppl = sentence_ppls[i] if i < len(sentence_ppls) else float("inf")
            clf_prob = sentence_clf_probs[i] if i < len(sentence_clf_probs) else 0.5

            # Use the SAME sigmoid as global perplexity (consistent mapping)
            ppl_prob = self.perplexity_engine.ppl_to_ai_probability(ppl)

            # Blend based on classifier availability
            sentence_ai_prob = ppl_weight * ppl_prob + clf_weight * clf_prob

            sentence_result = {
                "text": sentence,
                "perplexity": round(ppl, 2),
                "ppl_ai_probability": round(ppl_prob, 4),
                "classifier_prob": round(clf_prob, 4),
                "ai_probability": round(sentence_ai_prob, 4),
            }

            # Attach token attribution data if available
            if i < len(sentence_attributions):
                attr = sentence_attributions[i]
                sentence_result["word_attributions"] = attr.get("word_attributions", [])
                sentence_result["top_ai_tokens"] = attr.get("top_ai_tokens", [])
                sentence_result["top_human_tokens"] = attr.get("top_human_tokens", [])
            else:
                sentence_result["word_attributions"] = []
                sentence_result["top_ai_tokens"] = []
                sentence_result["top_human_tokens"] = []

            results.append(sentence_result)
        return results

    def _determine_verdict(
        self, ensemble_prob: float, sentence_scores: list[dict]
    ) -> str:
        """
        Determine overall verdict based on ensemble and sentence-level analysis.

        Trinary classification: Human, AI, or Mixed.
        """
        if ensemble_prob > config.VERDICT_AI_THRESHOLD:
            return "AI"
        elif ensemble_prob < config.VERDICT_HUMAN_THRESHOLD:
            return "Human"

        # Check for mixed authorship: some sentences AI, some human
        ai_sentences = sum(
            1 for s in sentence_scores
            if s["ai_probability"] > config.VERDICT_MIXED_AI_SENTENCE
        )
        human_sentences = sum(
            1 for s in sentence_scores
            if s["ai_probability"] < config.VERDICT_MIXED_HUMAN_SENTENCE
        )
        total = len(sentence_scores)

        # Mixed: at least 20% AI sentences AND 20% human sentences
        if total >= 3 and ai_sentences >= total * 0.2 and human_sentences >= total * 0.2:
            return "Mixed"

        return "AI" if ensemble_prob > 0.5 else "Human"

    def _determine_confidence(
        self, ensemble_prob: float, verdict: str, mc_result: dict | None = None
    ) -> tuple[str, str]:
        """
        Determine confidence level for the prediction.

        Now also considers MC Dropout uncertainty [Novelty #7]:
        High MC uncertainty downgrades confidence tier.

        Returns:
            (confidence_level, confidence_category)
        """
        thresholds = config.CONFIDENCE_THRESHOLDS

        if ensemble_prob >= thresholds["high_ai"]:
            base_confidence, base_category = "high", "Highly confident this is AI-generated"
        elif ensemble_prob >= thresholds["moderate_ai"]:
            base_confidence, base_category = "moderate", "Moderately confident this is AI-generated"
        elif ensemble_prob >= thresholds["uncertain_high"]:
            base_confidence, base_category = "low", "Uncertain, leaning toward AI-generated"
        elif ensemble_prob >= thresholds["uncertain_low"]:
            base_confidence, base_category = "low", "Uncertain"
        elif ensemble_prob >= thresholds["moderate_human"]:
            base_confidence, base_category = "moderate", "Moderately confident this is human-written"
        else:
            base_confidence, base_category = "high", "Highly confident this is human-written"

        # Downgrade if MC Dropout says the classifier is uncertain [Novelty #7]
        if mc_result and mc_result.get("uncertainty_level") == "high":
            downgrade = {"high": "moderate", "moderate": "low", "low": "low"}
            base_confidence = downgrade.get(base_confidence, base_confidence)
            base_category += " (high uncertainty)"

        return base_confidence, base_category

    @staticmethod
    def _empty_result() -> dict:
        """Return a default result when no analyzable text is found."""
        return {
            "verdict": "Unknown",
            "ai_probability": 0.0,
            "confidence": "low",
            "confidence_category": "Insufficient text for analysis",
            "uncertainty": None,
            "components": {},
            "sentences": [],
            "domain": "general",
            "domain_info": {},
            "fingerprint": None,
            "metadata": {},
        }
