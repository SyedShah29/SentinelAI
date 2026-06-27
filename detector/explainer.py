"""
Contrastive Explanation Engine — Novelty #9

Uses a local Ollama LLM to explain WHY sentences are flagged as AI,
and suggests human-sounding rewrites (contrastive NLG).

This turns the detector from a binary scanner into a writing coach:
  - Identifies the top N most AI-flagged sentences
  - Asks Ollama to rewrite them to sound more human
  - Shows a before/after diff for each sentence

Requirements:
  - Ollama installed: winget install Ollama.Ollama
  - Model pulled: ollama pull llama3.2
  - Service running (ollama serve happens automatically after install)
"""

import json
from typing import Any

import config


class OllamaExplainer:
    """
    Uses Ollama's local LLM to generate contrastive rewrites
    for the most AI-flagged sentences.
    """

    def __init__(self):
        self._available = None  # Cached availability check

    def _check_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        if self._available is not None:
            return self._available
        try:
            import httpx
            resp = httpx.get(
                f"{config.OLLAMA_HOST}/api/tags",
                timeout=3.0,
            )
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                target = config.OLLAMA_MODEL.split(":")[0]
                self._available = target in model_names
            else:
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def is_available(self) -> bool:
        return config.OLLAMA_ENABLED and self._check_available()

    def _rewrite_sentence(self, sentence: str, ai_prob: float) -> str:
        """
        Ask Ollama to rewrite a sentence to sound more human-written.

        Returns:
            Rewritten sentence string, or empty string on failure.
        """
        import httpx

        reason = (
            f"This sentence was flagged as {ai_prob:.0%} likely AI-generated."
        )

        prompt = (
            f"You are a writing coach helping someone make their text sound more natural and human-written.\n\n"
            f"Original sentence: \"{sentence}\"\n\n"
            f"{reason} Please rewrite it to:\n"
            f"- Vary sentence structure and length\n"
            f"- Use more natural, conversational language\n"
            f"- Remove overly formal connectives (however, moreover, furthermore)\n"
            f"- Add a personal touch or specific detail if possible\n\n"
            f"Provide ONLY the rewritten sentence, no explanation."
        )

        try:
            resp = httpx.post(
                f"{config.OLLAMA_HOST}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 200,
                    },
                },
                timeout=config.OLLAMA_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return ""
        except Exception:
            return ""

    def explain(self, sentences: list[dict]) -> dict[str, Any]:
        """
        Generate contrastive explanations for the most AI-flagged sentences.

        Args:
            sentences: List of sentence dicts from EnsembleDetector output,
                       each with 'text' and 'ai_probability'.

        Returns:
            dict with:
                - available: bool
                - explanations: list of {original, rewritten, ai_probability, improvement}
        """
        if not self.is_available():
            return {
                "available": False,
                "reason": (
                    "Ollama is not running or the model is not installed. "
                    "Install: winget install Ollama.Ollama && ollama pull llama3.2"
                ),
                "explanations": [],
            }

        # Select top N most AI-flagged sentences
        ai_sentences = sorted(
            [s for s in sentences if s.get("ai_probability", 0) > 0.6],
            key=lambda s: s["ai_probability"],
            reverse=True,
        )[:config.OLLAMA_MAX_SENTENCES]

        if not ai_sentences:
            return {
                "available": True,
                "explanations": [],
                "message": "No strongly AI-flagged sentences to explain.",
            }

        explanations = []
        for s in ai_sentences:
            original = s["text"]
            ai_prob = s["ai_probability"]
            rewritten = self._rewrite_sentence(original, ai_prob)
            if rewritten:
                # Estimate "improvement" based on how different the rewrite is
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(
                    None, original.lower(), rewritten.lower()
                ).ratio()
                explanations.append({
                    "original": original,
                    "rewritten": rewritten,
                    "ai_probability": round(ai_prob, 4),
                    "similarity_ratio": round(similarity, 3),
                    "change_level": (
                        "Major rewrite" if similarity < 0.5
                        else "Moderate rewrite" if similarity < 0.75
                        else "Minor rewrite"
                    ),
                })

        return {
            "available": True,
            "explanations": explanations,
        }
