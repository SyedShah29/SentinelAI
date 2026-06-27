"""
Domain Classifier — Domain-Aware Detection (Novelty #8)

Detects the domain of input text and applies calibrated, domain-specific
perplexity thresholds and sigmoid parameters.

Why this matters:
  - Academic writing has naturally lower perplexity than Reddit posts
  - Code is highly predictable regardless of author
  - Applying one global threshold produces false positives on academic/legal text
  - Domain-specific calibration dramatically reduces false alarm rates

Domains supported:
  - general      → default thresholds
  - academic     → lower PPL threshold (formulaic by nature)
  - code         → very low PPL threshold; pattern-based code detection
  - social_media → higher PPL threshold (chaotic / slang)
  - news         → slightly lower threshold (journalistic style)
  - legal        → low threshold (highly formulaic language)
"""

import re
from typing import Any

import config


# ── Domain keyword/pattern rules ─────────────────────────────────────────────
# Each rule is (pattern, weight_delta). Higher positive weight_delta
# means more confident in that domain.

_DOMAIN_RULES: dict[str, list[re.Pattern]] = {
    "code": [
        re.compile(r"\bdef\s+\w+\s*\(", re.IGNORECASE),          # Python function
        re.compile(r"\bfunction\s+\w+\s*\("),                      # JS function
        re.compile(r"\bclass\s+\w+[\s:{]"),                        # class definition
        re.compile(r"^\s*(import|from)\s+\w+", re.MULTILINE),     # imports
        re.compile(r"```[\w]*\n"),                                  # code block
        re.compile(r"\bif\s*\(.+\)\s*\{"),                        # C-style if
        re.compile(r"<\w+[\s>].*</\w+>"),                         # HTML tags
        re.compile(r"SELECT\s+.+\s+FROM\s+\w+", re.IGNORECASE),   # SQL
        re.compile(r"\b(var|let|const)\s+\w+\s*="),               # JS vars
        re.compile(r"#include\s*<\w+\.h>"),                        # C/C++ include
    ],
    "academic": [
        re.compile(r"\b(abstract|methodology|hypothesis|empirical|findings)\b", re.I),
        re.compile(r"\b(et al\.|ibid\.|op\. cit\.)\b"),
        re.compile(r"\(\d{4}\)"),                           # In-text citation (Smith, 2020)
        re.compile(r"\b(journal|proceedings|doi|arxiv)\b", re.I),
        re.compile(r"\b(p-value|significance|regression|variance|coefficient)\b", re.I),
        re.compile(r"\b(conclusion|introduction|background|related work)\b", re.I),
        re.compile(r"\b(figure|table|section|appendix)\s+\d+", re.I),
        re.compile(r"\b(literature review|theoretical framework)\b", re.I),
    ],
    "legal": [
        re.compile(r"\b(whereas|hereinafter|pursuant to|notwithstanding)\b", re.I),
        re.compile(r"\b(party|parties|plaintiff|defendant|jurisdiction)\b", re.I),
        re.compile(r"\b(indemnify|liability|clause|provision|statute)\b", re.I),
        re.compile(r"\b(§\s*\d+|section\s+\d+\.\d+)\b", re.I),
        re.compile(r"\b(agreement|contract|terms and conditions)\b", re.I),
    ],
    "news": [
        re.compile(r"\b(reported|according to|sources said|officials said)\b", re.I),
        re.compile(r"\b(breaking|exclusive|update|developing)\b", re.I),
        re.compile(r"\b[A-Z][a-z]+,\s+[A-Z]{2}\s+[—–]\s+"),  # City, ST — dateline
        re.compile(r"\b(AP|Reuters|AFP|BBC|CNN|FOX|NPR)\b"),
        re.compile(r"\b(mayor|president|senator|spokesperson)\b", re.I),
    ],
    "social_media": [
        re.compile(r"#\w+"),                                   # hashtags
        re.compile(r"@\w+"),                                   # mentions
        re.compile(r"\b(lol|omg|tbh|imo|smh|ngl|irl|fwiw)\b", re.I),
        re.compile(r"😂|❤️|👍|🔥|💯|😭|✨|🙏"),            # common emojis
        re.compile(r"\b(reddit|twitter|instagram|tiktok|tweet)\b", re.I),
        re.compile(r"\.{3,}"),                                  # ellipsis overuse
        re.compile(r"!{2,}"),                                   # multiple exclamation marks
    ],
}


class DomainClassifier:
    """
    Classifies text into one of the supported domains and returns
    calibrated perplexity thresholds.
    """

    def classify(self, text: str) -> dict[str, Any]:
        """
        Classify the domain of input text.

        Returns:
            dict with:
                - domain: detected domain label
                - confidence: detection confidence [0, 1]
                - ppl_threshold: calibrated PPL threshold for this domain
                - sigmoid_k: calibrated sigmoid steepness
                - scores: per-domain match scores
        """
        scores = self._score_domains(text)
        best_domain = max(scores, key=lambda d: scores[d])
        best_score = scores[best_domain]

        # Fall back to "general" if no domain has strong signal
        if best_score < 1:
            best_domain = "general"
            confidence = 0.3
        elif best_score < 3:
            confidence = 0.6
        else:
            confidence = min(1.0, best_score / 8.0 + 0.5)

        return {
            "domain": best_domain,
            "confidence": round(confidence, 3),
            "ppl_threshold": config.DOMAIN_PPL_THRESHOLDS.get(
                best_domain, config.PERPLEXITY_THRESHOLD_AI
            ),
            "sigmoid_k": config.DOMAIN_SIGMOID_K.get(
                best_domain, config.PERPLEXITY_SIGMOID_K
            ),
            "scores": {d: s for d, s in scores.items()},
        }

    def _score_domains(self, text: str) -> dict[str, int]:
        """Count pattern matches per domain."""
        scores = {d: 0 for d in config.DOMAIN_LABELS}
        scores["general"] = 0  # Always possible

        for domain, patterns in _DOMAIN_RULES.items():
            for pattern in patterns:
                if pattern.search(text):
                    scores[domain] += 1

        return scores
