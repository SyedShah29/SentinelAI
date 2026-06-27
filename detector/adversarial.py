"""
Adversarial Robustness Tester (Novelty #2)

Tests how robust a detection result is against paraphrasing attacks.

Strategy:
  1. Generate N paraphrases of the input text using T5-based paraphraser
  2. Score each paraphrase with the full ensemble
  3. Report:
     - Evasion success rate: % of paraphrases that flip to "Human" verdict
     - Mean AI probability across paraphrases (stability score)
     - Std deviation (how unstable is detection?)
     - Evasion risk: "Low / Medium / High"

This answers: "If someone paraphrases this AI text, can they evade detection?"

Model: humarin/chatgpt_paraphraser_on_T5_base (small, fast, no GPU needed)
"""

import re
from typing import Any, TYPE_CHECKING

import config

if TYPE_CHECKING:
    from detector.ensemble import EnsembleDetector


class AdversarialTester:
    """
    Tests detection robustness by paraphrasing and re-scoring input text.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load the paraphrase model."""
        if self._loaded:
            return
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            model_name = config.ADVERSARIAL_PARAPHRASE_MODEL
            print(f"Loading paraphrase model: {model_name}...")
            self._tokenizer = T5Tokenizer.from_pretrained(model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(model_name)
            self._model.eval()
            self._loaded = True
            print("  [OK] Paraphrase model loaded")
        except Exception as e:
            print(f"  [WARN] Failed to load paraphrase model: {e}")
            self._loaded = False

    def _paraphrase(self, text: str) -> list[str]:
        """
        Generate N paraphrases of the input text.

        Args:
            text: Input text to paraphrase (truncated to 256 words for speed)

        Returns:
            List of paraphrased strings.
        """
        if not self._loaded:
            return []

        import torch

        # Truncate to avoid slow inference on very long texts
        words = text.split()
        if len(words) > 200:
            text = " ".join(words[:200])

        encoding = self._tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            padding="longest",
            max_length=config.ADVERSARIAL_MAX_LENGTH,
            truncation=True,
        )

        with torch.no_grad():
            output = self._model.generate(
                **encoding,
                max_length=config.ADVERSARIAL_MAX_LENGTH,
                num_beams=config.ADVERSARIAL_NUM_BEAMS,
                num_return_sequences=config.ADVERSARIAL_NUM_PARAPHRASES,
                temperature=1.5,
                repetition_penalty=1.5,
            )

        paraphrases = [
            self._tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output
        ]
        # Deduplicate and filter identical paraphrases
        seen = {text}
        unique = []
        for p in paraphrases:
            if p not in seen and len(p) > 20:
                seen.add(p)
                unique.append(p)
        return unique

    def test(self, text: str, original_result: dict, detector: "EnsembleDetector") -> dict[str, Any]:
        """
        Run adversarial paraphrase test.

        Args:
            text: Original input text
            original_result: The result from detector.analyze(text)
            detector: The EnsembleDetector instance to re-score paraphrases

        Returns:
            dict with evasion_risk, evasion_rate, stability_score, paraphrase_results
        """
        if not config.ADVERSARIAL_ENABLED:
            return {"enabled": False}

        self._load_model()
        if not self._loaded:
            return {
                "enabled": False,
                "error": "Paraphrase model failed to load",
            }

        original_prob = original_result.get("ai_probability", 0.5)
        original_verdict = original_result.get("verdict", "Unknown")

        # Skip if original is already classified as Human (nothing to evade)
        if original_verdict == "Human" or original_prob < 0.5:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "Text already classified as Human — no evasion needed",
            }

        # Generate paraphrases
        paraphrases = self._paraphrase(text)

        if not paraphrases:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "Could not generate paraphrases",
            }

        # Score each paraphrase
        paraphrase_results = []
        for p in paraphrases:
            try:
                result = detector.analyze(p)
                paraphrase_results.append({
                    "text_preview": p[:150] + ("..." if len(p) > 150 else ""),
                    "ai_probability": result["ai_probability"],
                    "verdict": result["verdict"],
                })
            except Exception:
                continue

        if not paraphrase_results:
            return {"enabled": True, "skipped": True, "reason": "Scoring failed"}

        # Compute evasion metrics
        evaded = [r for r in paraphrase_results if r["verdict"] == "Human"]
        evasion_rate = len(evaded) / len(paraphrase_results)
        ai_probs = [r["ai_probability"] for r in paraphrase_results]
        mean_prob = sum(ai_probs) / len(ai_probs)
        prob_std = (
            (sum((p - mean_prob) ** 2 for p in ai_probs) / len(ai_probs)) ** 0.5
        )

        # Evasion risk classification
        if evasion_rate >= 0.6:
            evasion_risk = "High"
            risk_description = (
                "This text is easy to paraphrase into undetectable form. "
                "Results should be treated with caution."
            )
        elif evasion_rate >= 0.3:
            evasion_risk = "Medium"
            risk_description = (
                "Some paraphrases evade detection. Moderate confidence in the verdict."
            )
        else:
            evasion_risk = "Low"
            risk_description = (
                "Detection is robust — paraphrasing does not evade the detector."
            )

        # Probability drop: how much does AI prob drop after paraphrasing?
        prob_drop = original_prob - mean_prob

        return {
            "enabled": True,
            "skipped": False,
            "evasion_risk": evasion_risk,
            "evasion_risk_description": risk_description,
            "evasion_rate": round(evasion_rate, 3),
            "original_ai_probability": round(original_prob, 4),
            "mean_paraphrase_ai_probability": round(mean_prob, 4),
            "probability_drop": round(prob_drop, 4),
            "probability_std": round(prob_std, 4),
            "num_paraphrases_tested": len(paraphrase_results),
            "paraphrase_results": paraphrase_results,
        }
