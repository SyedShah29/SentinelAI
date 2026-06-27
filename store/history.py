"""
History Store — SQLite-backed analysis history for the AI detector.

Novelty #10: Analysis History & Drift Tracking

Features:
  - Log every analysis: timestamp, verdict, scores, word count, domain
  - Never stores raw text (privacy-preserving — SHA-256 hash only)
  - Per-session and all-time stats for the dashboard
  - Auto-prune oldest entries when limit exceeded
  - Drift detection: track user's own writing score over time
"""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import config


class HistoryManager:
    """Manages local SQLite analysis history."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS analyses (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              REAL NOT NULL,            -- Unix timestamp
        text_hash       TEXT NOT NULL,            -- SHA-256 of normalized text
        verdict         TEXT NOT NULL,            -- Human / AI / Mixed / Unknown
        ai_probability  REAL NOT NULL,            -- 0.0–1.0
        confidence      TEXT,                     -- high / moderate / low
        ppl_score       REAL,                     -- Global perplexity
        burst_score     REAL,                     -- Burstiness CoV
        clf_score       REAL,                     -- Classifier probability
        stylo_score     REAL,                     -- Stylometry probability (or NULL)
        word_count      INTEGER,
        sentence_count  INTEGER,
        domain          TEXT DEFAULT 'general',   -- Detected domain
        char_count      INTEGER,
        model_tag       TEXT,                     -- Top fingerprint label (or NULL)
        session_id      TEXT                      -- Optional session grouping
    );
    CREATE INDEX IF NOT EXISTS idx_ts ON analyses (ts);
    CREATE INDEX IF NOT EXISTS idx_verdict ON analyses (verdict);
    CREATE INDEX IF NOT EXISTS idx_domain ON analyses (domain);
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or config.HISTORY_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def log(self, text: str, result: dict[str, Any], session_id: str | None = None) -> int:
        """
        Log an analysis result.

        Args:
            text: Original input text (only hash is stored)
            result: Full result dict from EnsembleDetector.analyze()
            session_id: Optional session identifier

        Returns:
            Row ID of the inserted record
        """
        comp = result.get("components", {})
        meta = result.get("metadata", {})
        sentences = result.get("sentences", [])

        ppl = comp.get("perplexity", {}).get("global_ppl")
        burst = comp.get("burstiness", {}).get("score")
        clf = comp.get("classifier", {}).get("ai_probability")
        stylo = comp.get("stylometry", {}).get("ai_probability") if "stylometry" in comp else None
        model_tag = result.get("fingerprint", {}).get("top_label") if "fingerprint" in result else None

        word_count = len(text.split())
        char_count = meta.get("original_length", len(text))
        sentence_count = meta.get("num_sentences", len(sentences))
        domain = result.get("domain", "general")

        row = {
            "ts": time.time(),
            "text_hash": self._hash_text(text),
            "verdict": result.get("verdict", "Unknown"),
            "ai_probability": result.get("ai_probability", 0.0),
            "confidence": result.get("confidence", "low"),
            "ppl_score": ppl,
            "burst_score": burst,
            "clf_score": clf,
            "stylo_score": stylo,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "domain": domain,
            "char_count": char_count,
            "model_tag": model_tag,
            "session_id": session_id,
        }

        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO analyses
                   (ts, text_hash, verdict, ai_probability, confidence,
                    ppl_score, burst_score, clf_score, stylo_score,
                    word_count, sentence_count, domain, char_count, model_tag, session_id)
                   VALUES
                   (:ts, :text_hash, :verdict, :ai_probability, :confidence,
                    :ppl_score, :burst_score, :clf_score, :stylo_score,
                    :word_count, :sentence_count, :domain, :char_count, :model_tag, :session_id)""",
                row,
            )
            row_id = cursor.lastrowid

        # Auto-prune if over limit
        self._prune_if_needed()
        return row_id

    def _prune_if_needed(self):
        """Remove oldest entries if total exceeds HISTORY_MAX_ENTRIES."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            if count > config.HISTORY_MAX_ENTRIES:
                excess = count - config.HISTORY_MAX_ENTRIES
                conn.execute(
                    "DELETE FROM analyses WHERE id IN "
                    "(SELECT id FROM analyses ORDER BY ts ASC LIMIT ?)",
                    (excess,),
                )

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent N analysis records."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """
        Compute aggregate statistics for the History Dashboard.

        Returns:
            Dict with total_count, verdict_breakdown, domain_breakdown,
            avg_ai_probability, time_series (daily buckets for last 30 days),
            score_distribution (histogram buckets).
        """
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            if total == 0:
                return {"total_count": 0}

            # Verdict breakdown
            verdict_rows = conn.execute(
                "SELECT verdict, COUNT(*) as cnt FROM analyses GROUP BY verdict"
            ).fetchall()
            verdict_breakdown = {r["verdict"]: r["cnt"] for r in verdict_rows}

            # Domain breakdown
            domain_rows = conn.execute(
                "SELECT domain, COUNT(*) as cnt FROM analyses GROUP BY domain ORDER BY cnt DESC"
            ).fetchall()
            domain_breakdown = {r["domain"]: r["cnt"] for r in domain_rows}

            # Average AI probability
            avg_prob = conn.execute(
                "SELECT AVG(ai_probability) FROM analyses"
            ).fetchone()[0]

            # Time series — last 30 days, grouped by day
            cutoff = time.time() - 30 * 86400
            ts_rows = conn.execute(
                """SELECT
                       DATE(ts, 'unixepoch') as day,
                       COUNT(*) as count,
                       AVG(ai_probability) as avg_prob,
                       SUM(CASE WHEN verdict='AI' THEN 1 ELSE 0 END) as ai_count
                   FROM analyses
                   WHERE ts >= ?
                   GROUP BY day
                   ORDER BY day ASC""",
                (cutoff,),
            ).fetchall()
            time_series = [dict(r) for r in ts_rows]

            # Score histogram (10 buckets: 0-10%, 10-20%, ..., 90-100%)
            hist_rows = conn.execute(
                """SELECT
                       CAST(ai_probability * 10 AS INT) as bucket,
                       COUNT(*) as count
                   FROM analyses
                   GROUP BY bucket
                   ORDER BY bucket ASC"""
            ).fetchall()
            score_histogram = {r["bucket"] * 10: r["count"] for r in hist_rows}

            # Model fingerprint breakdown (if available)
            model_rows = conn.execute(
                """SELECT model_tag, COUNT(*) as cnt
                   FROM analyses
                   WHERE model_tag IS NOT NULL
                   GROUP BY model_tag
                   ORDER BY cnt DESC"""
            ).fetchall()
            model_breakdown = {r["model_tag"]: r["cnt"] for r in model_rows}

            # Drift: rolling 7-day average AI probability (last 8 weeks)
            drift_rows = conn.execute(
                """SELECT
                       strftime('%Y-%W', ts, 'unixepoch') as week,
                       AVG(ai_probability) as avg_prob,
                       COUNT(*) as count
                   FROM analyses
                   WHERE ts >= ?
                   GROUP BY week
                   ORDER BY week ASC""",
                (time.time() - 56 * 86400,),
            ).fetchall()
            drift = [dict(r) for r in drift_rows]

        return {
            "total_count": total,
            "verdict_breakdown": verdict_breakdown,
            "domain_breakdown": domain_breakdown,
            "avg_ai_probability": round(avg_prob or 0.0, 4),
            "time_series": time_series,
            "score_histogram": score_histogram,
            "model_breakdown": model_breakdown,
            "drift": drift,
        }

    def clear(self):
        """Delete all history records."""
        with self._connect() as conn:
            conn.execute("DELETE FROM analyses")

    def count(self) -> int:
        """Return total number of stored analyses."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
