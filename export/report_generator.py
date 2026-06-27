"""
Report Generator — PDF/HTML Export (Novelty #6)

Generates a beautifully formatted analysis report from detector results.

Supports two formats:
  - HTML: Full dark-themed report with color-coded sentences
  - PDF: Rendered from the HTML template using reportlab (fallback: weasyprint)

Usage:
    from export.report_generator import ReportGenerator
    gen = ReportGenerator()
    html_bytes = gen.render_html(text, result)
    pdf_bytes = gen.render_pdf(text, result)   # requires reportlab
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import config


class ReportGenerator:
    """Generates HTML and PDF reports from analysis results."""

    def __init__(self):
        self._jinja_env = None
        self._init_jinja()

    def _init_jinja(self):
        """Initialize the Jinja2 environment."""
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            template_dir = config.EXPORT_TEMPLATE_DIR
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html"]),
            )
        except ImportError:
            self._jinja_env = None

    def _build_context(self, text: str, result: dict[str, Any]) -> dict[str, Any]:
        """Build Jinja2 template context from detector result."""
        comp = result.get("components", {})
        meta = result.get("metadata", {})
        sentences = result.get("sentences", [])
        fingerprint = result.get("fingerprint")
        uncertainty = result.get("uncertainty")
        domain = result.get("domain", "general")

        # Limit sentence count for very long documents
        sentences_truncated = sentences[:config.EXPORT_MAX_SENTENCES_IN_REPORT]

        # Verdict label
        verdict = result.get("verdict", "Unknown")
        ai_prob = result.get("ai_probability", 0.0)
        if verdict == "AI":
            verdict_label = f"AI-Generated ({ai_prob:.0%})"
        elif verdict == "Human":
            verdict_label = f"Human-Written ({1 - ai_prob:.0%})"
        elif verdict == "Mixed":
            verdict_label = "Mixed Authorship"
        else:
            verdict_label = "Unknown"

        # Warnings
        warnings = []
        if meta.get("homoglyphs_detected"):
            warnings.append(
                "Homoglyph characters detected. This text may have been modified "
                "to evade AI detection. Characters were normalized before analysis."
            )
        if not meta.get("classifier_trained"):
            warnings.append(
                "The DeBERTa classifier is not fine-tuned. Results rely on "
                "perplexity and burstiness only."
            )

        # Stylometry
        stylo = comp.get("stylometry", {})
        stylo_ai = stylo.get("ai_probability") if stylo else None
        stylo_connectives = stylo.get("flagged_connectives", []) if stylo else []

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verdict": verdict,
            "verdict_label": verdict_label,
            "ai_probability": ai_prob,
            "confidence": result.get("confidence", "low"),
            "confidence_category": result.get("confidence_category", ""),
            "char_count": f"{meta.get('original_length', len(text)):,}",
            "sentence_count": meta.get("num_sentences", len(sentences)),
            "language": meta.get("language", "unknown"),
            "domain": domain,
            "homoglyphs": meta.get("homoglyphs_detected", False),
            "clf_trained": meta.get("classifier_trained", False),
            # Component scores
            "ppl": comp.get("perplexity", {}).get("global_ppl", 0.0),
            "ppl_ai": comp.get("perplexity", {}).get("ai_probability", 0.0),
            "burst": comp.get("burstiness", {}).get("score", 0.0),
            "burst_ai": comp.get("burstiness", {}).get("ai_probability", 0.0),
            "clf_ai": comp.get("classifier", {}).get("ai_probability", 0.0),
            "stylo_ai": stylo_ai,
            "stylo_connectives": stylo_connectives,
            # Full data
            "sentences": sentences_truncated,
            "uncertainty": uncertainty,
            "fingerprint": fingerprint,
            "warnings": warnings,
        }

    def render_html(self, text: str, result: dict[str, Any]) -> bytes:
        """
        Render the analysis report as HTML bytes.

        Returns:
            UTF-8 encoded HTML bytes.
        """
        if self._jinja_env is None:
            return self._fallback_html(result).encode("utf-8")

        try:
            template = self._jinja_env.get_template("report.html")
            ctx = self._build_context(text, result)
            rendered = template.render(**ctx)
            return rendered.encode("utf-8")
        except Exception as e:
            return self._fallback_html(result, error=str(e)).encode("utf-8")

    def render_pdf(self, text: str, result: dict[str, Any]) -> bytes | None:
        """
        Render the analysis report as PDF bytes.

        Uses reportlab for a clean, structured PDF.

        Returns:
            PDF bytes, or None if reportlab is not available.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, KeepTogether
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=20*mm, leftMargin=20*mm,
                topMargin=20*mm, bottomMargin=20*mm,
                title="AI Detection Report",
            )

            ctx = self._build_context(text, result)
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                "Title2", parent=styles["Heading1"],
                fontSize=18, spaceAfter=4,
                textColor=colors.HexColor("#7c3aed"),
            )
            h2_style = ParagraphStyle(
                "H2", parent=styles["Heading2"],
                fontSize=13, spaceAfter=6, spaceBefore=14,
                textColor=colors.HexColor("#a78bfa"),
            )
            body_style = ParagraphStyle(
                "Body2", parent=styles["Normal"],
                fontSize=10, spaceAfter=4,
                textColor=colors.HexColor("#374151"),
            )
            ai_style = ParagraphStyle(
                "AI", parent=body_style,
                backColor=colors.HexColor("#fff1f1"),
                leftIndent=8, borderColor=colors.HexColor("#ff4b4b"),
                borderWidth=2, borderPadding=4,
            )
            human_style = ParagraphStyle(
                "Human", parent=body_style,
                backColor=colors.HexColor("#f0fff4"),
                leftIndent=8, borderColor=colors.HexColor("#00c853"),
                borderWidth=2, borderPadding=4,
            )
            mixed_style = ParagraphStyle(
                "Mixed", parent=body_style,
                backColor=colors.HexColor("#fffbf0"),
                leftIndent=8, borderColor=colors.HexColor("#ff9800"),
                borderWidth=2, borderPadding=4,
            )

            story = []

            # Title
            story.append(Paragraph("🔍 AI Text Detection Report", title_style))
            story.append(Paragraph(
                f"Generated {ctx['timestamp']} | {ctx['char_count']} characters | {ctx['sentence_count']} sentences",
                ParagraphStyle("sub", parent=body_style, textColor=colors.HexColor("#9ca3af"), fontSize=9)
            ))
            story.append(HRFlowable(color=colors.HexColor("#7c3aed"), thickness=1.5))
            story.append(Spacer(1, 8))

            # Verdict
            verdict_color = {"AI": "#ff4b4b", "Human": "#00c853", "Mixed": "#ff9800"}.get(
                ctx["verdict"], "#6b7280"
            )
            story.append(Paragraph(
                f"<b>Verdict: {ctx['verdict_label']}</b>",
                ParagraphStyle("verdict", parent=body_style, fontSize=14,
                               textColor=colors.HexColor(verdict_color), spaceAfter=2)
            ))
            story.append(Paragraph(
                ctx["confidence_category"],
                ParagraphStyle("conf", parent=body_style, textColor=colors.HexColor("#ff9800"), fontSize=10)
            ))
            story.append(Spacer(1, 10))

            # Scores table
            story.append(Paragraph("Component Breakdown", h2_style))
            table_data = [
                ["Component", "Score", "AI Probability"],
                ["Perplexity", f"{ctx['ppl']:.1f}", f"{ctx['ppl_ai']:.0%}"],
                ["Burstiness", f"{ctx['burst']:.3f}", f"{ctx['burst_ai']:.0%}"],
                ["Classifier", "—", f"{ctx['clf_ai']:.0%}"],
            ]
            if ctx["stylo_ai"] is not None:
                table_data.append(["Stylometry", "—", f"{ctx['stylo_ai']:.0%}"])
            table_data.append(["ENSEMBLE", "—", f"{ctx['ai_probability']:.0%}"])

            tbl = Table(table_data, colWidths=[80*mm, 50*mm, 50*mm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9ff")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 10))

            # Sentence analysis
            story.append(Paragraph("Sentence-Level Analysis", h2_style))
            for i, s in enumerate(ctx["sentences"]):
                prob = s["ai_probability"]
                if prob > 0.6:
                    icon = "🔴 AI"
                    style = ai_style
                elif prob < 0.4:
                    icon = "🟢 Human"
                    style = human_style
                else:
                    icon = "🟡 Uncertain"
                    style = mixed_style

                story.append(Paragraph(
                    f"<b>[S{i+1}] {icon} ({prob:.0%})</b> — PPL: {s['perplexity']:.1f}<br/>{s['text']}",
                    style
                ))
                story.append(Spacer(1, 3))

            # Footer
            story.append(Spacer(1, 16))
            story.append(HRFlowable(color=colors.HexColor("#e5e7eb"), thickness=0.5))
            story.append(Paragraph(
                "AI Text Detector Report | For informational purposes only. Human review is always recommended.",
                ParagraphStyle("footer", parent=body_style, fontSize=8,
                               textColor=colors.HexColor("#9ca3af"), alignment=TA_CENTER)
            ))

            doc.build(story)
            buffer.seek(0)
            return buffer.read()

        except ImportError:
            return None
        except Exception as e:
            print(f"PDF generation failed: {e}")
            return None

    def _fallback_html(self, result: dict, error: str = "") -> str:
        """Minimal HTML fallback if Jinja2 is not available."""
        verdict = result.get("verdict", "Unknown")
        prob = result.get("ai_probability", 0.0)
        return f"""<!DOCTYPE html><html><body>
        <h1>AI Detection Report</h1>
        <p>Verdict: <b>{verdict}</b> ({prob:.0%} AI probability)</p>
        {'<p style="color:red">Render error: ' + error + '</p>' if error else ''}
        </body></html>"""

    def save_html(self, text: str, result: dict, output_path: str | Path) -> Path:
        """Render and save HTML report to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.render_html(text, result))
        return output_path

    def save_pdf(self, text: str, result: dict, output_path: str | Path) -> Path | None:
        """Render and save PDF report to disk."""
        pdf_bytes = self.render_pdf(text, result)
        if pdf_bytes is None:
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)
        return output_path
