"""
SentinelAI — AI Content Detection Platform

Capabilities:
  - Multi-LLM Fingerprinting: Identify which AI model authored the text
  - Adversarial Robustness: Evasion risk testing via paraphrasing
  - Stylometry: Writing style signal (connectives, passive voice, etc.)
  - Browser Extension: One-click detection from any webpage
  - Streaming Analysis: Live per-sentence probability reveal
  - PDF/HTML Export: Download full analysis reports
  - MC Dropout Uncertainty: Bayesian confidence intervals
  - Domain-Aware Detection: Auto-calibrated thresholds per domain
  - Contrastive Explanation: Humanization suggestions via local LLM
  - History Dashboard: Trends, drift, score distribution over time
"""

import sys
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector.ensemble import EnsembleDetector
from store.history import HistoryManager
from export.report_generator import ReportGenerator

import config

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelAI — AI Content Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-header {
    text-align: center;
    padding: 1.2rem 0 0.5rem;
  }
  .main-header h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; }

  /* Verdict banners */
  .verdict-ai {
    background: linear-gradient(135deg, #ff4b4b, #c0392b);
    color: white; padding: 1rem 2rem; border-radius: 14px;
    text-align: center; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.5rem;
    box-shadow: 0 4px 24px rgba(255,75,75,0.35);
  }
  .verdict-human {
    background: linear-gradient(135deg, #00c853, #1b5e20);
    color: white; padding: 1rem 2rem; border-radius: 14px;
    text-align: center; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.5rem;
    box-shadow: 0 4px 24px rgba(0,200,83,0.35);
  }
  .verdict-mixed {
    background: linear-gradient(135deg, #ff9800, #e65100);
    color: white; padding: 1rem 2rem; border-radius: 14px;
    text-align: center; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.5rem;
    box-shadow: 0 4px 24px rgba(255,152,0,0.35);
  }

  /* Confidence badges */
  .confidence-badge {
    display: inline-block; padding: 0.3rem 0.9rem; border-radius: 20px;
    font-size: 0.82rem; font-weight: 600; margin-top: 0.3rem;
  }
  .confidence-high     { background: rgba(0,200,83,.15); color:#00c853; border:1px solid #00c853; }
  .confidence-moderate { background: rgba(255,152,0,.15); color:#ff9800; border:1px solid #ff9800; }
  .confidence-low      { background: rgba(255,75,75,.15); color:#ff4b4b; border:1px solid #ff4b4b; }

  /* Sentence blocks */
  .sentence-ai      { background: rgba(255,75,75,.12); border-left:4px solid #ff4b4b; padding:.6rem 1rem; margin:.3rem 0; border-radius:4px; }
  .sentence-human   { background: rgba(0,200,83,.10);  border-left:4px solid #00c853; padding:.6rem 1rem; margin:.3rem 0; border-radius:4px; }
  .sentence-uncertain{ background: rgba(255,152,0,.10); border-left:4px solid #ff9800; padding:.6rem 1rem; margin:.3rem 0; border-radius:4px; }

  /* Token badges */
  .token-indicators { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.3rem; }
  .token-ai-badge    { background:rgba(255,75,75,.2); color:#ff6b6b; padding:2px 8px; border-radius:12px; font-size:.72rem; border:1px solid rgba(255,75,75,.3); }
  .token-human-badge { background:rgba(0,200,83,.15); color:#69f0ae; padding:2px 8px; border-radius:12px; font-size:.72rem; border:1px solid rgba(0,200,83,.3); }

  /* Word highlights */
  .word-ai   { background-color: rgba(255,75,75,var(--intensity,.3));  border-radius:3px; padding:0 2px; }
  .word-human{ background-color: rgba(0,200,83,var(--intensity,.3));   border-radius:3px; padding:0 2px; }

  /* Warning / info banners */
  .warning-banner {
    background:rgba(255,152,0,.12); border:1px solid #ff9800;
    border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; font-size:.88rem;
  }
  .evasion-high     { background:rgba(255,75,75,.12); border:1px solid #ff4b4b; border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }
  .evasion-medium   { background:rgba(255,152,0,.12); border:1px solid #ff9800; border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }
  .evasion-low      { background:rgba(0,200,83,.10); border:1px solid #00c853; border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }

  /* Domain pill */
  .domain-pill {
    display:inline-block; background:rgba(124,58,237,.15); color:#a78bfa;
    border:1px solid rgba(124,58,237,.3); border-radius:12px; padding:2px 10px;
    font-size:.8rem; font-weight:600;
  }

  /* Uncertainty bar */
  .uncertainty-bar-bg { background:#2d2d42; border-radius:4px; height:8px; margin:4px 0; }
  .uncertainty-bar-fill { border-radius:4px; height:8px; transition:width 0.4s; }

  /* Fingerprint bar */
  .fp-bar-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
  .fp-bar-label { width:80px; font-size:.82rem; color:#d1d5db; }
  .fp-bar-bg { flex:1; background:#2d2d42; border-radius:4px; height:8px; }
  .fp-bar-fill { height:8px; border-radius:4px; background:linear-gradient(90deg,#7c3aed,#a78bfa); }
  .fp-bar-pct { width:36px; font-size:.8rem; color:#9ca3af; text-align:right; }

  /* Suggestion diff blocks */
  .suggestion-block { background:#1e1e2e; border-radius:10px; padding:1rem; margin-bottom:.75rem; border:1px solid #2d2d42; }
  .suggestion-original { color:#ff6b6b; padding:.5rem; background:rgba(255,75,75,.08); border-radius:6px; margin-bottom:.5rem; font-size:.88rem; }
  .suggestion-rewritten{ color:#69f0ae; padding:.5rem; background:rgba(0,200,83,.08); border-radius:6px; font-size:.88rem; }
  .suggestion-label { font-size:.72rem; font-weight:700; margin-bottom:.25rem; opacity:.7; }

  /* Connective pills */
  .conn-pill { display:inline-block; background:rgba(255,75,75,.15); color:#ff6b6b; border:1px solid rgba(255,75,75,.3); border-radius:12px; padding:2px 10px; font-size:.78rem; margin:3px; }

  /* Metric card */
  .metric-card { background:#1e1e2e; border-radius:12px; padding:1.5rem; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ── Cached Resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_detector():
    """Load the ensemble detector (cached across reruns)."""
    llm_best_model = Path("models/detector/llm_detector/best")
    if llm_best_model.exists():
        return EnsembleDetector(classifier_path=str(llm_best_model))
    best_model = Path("models/detector/best")
    if best_model.exists():
        return EnsembleDetector(classifier_path=str(best_model))
    return EnsembleDetector()


@st.cache_resource
def load_history():
    return HistoryManager()


@st.cache_resource
def load_report_generator():
    return ReportGenerator()


@st.cache_resource
def load_adversarial_tester():
    from detector.adversarial import AdversarialTester
    return AdversarialTester()


@st.cache_resource
def load_explainer():
    from detector.explainer import OllamaExplainer
    return OllamaExplainer()


# ── Chart Helpers ─────────────────────────────────────────────────────────────

def render_gauge(probability: float, uncertainty_std: float | None = None):
    """Render a circular gauge with optional uncertainty band."""
    pct = probability * 100
    bar_color = "#ff4b4b" if probability > 0.5 else "#00c853"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "AI Probability", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 30],  "color": "rgba(0,200,83,0.18)"},
                {"range": [30, 55], "color": "rgba(255,152,0,0.12)"},
                {"range": [55, 75], "color": "rgba(255,152,0,0.22)"},
                {"range": [75, 100],"color": "rgba(255,75,75,0.18)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))

    # Add uncertainty band if MC Dropout is available
    if uncertainty_std is not None and uncertainty_std > 0:
        low  = max(0, pct - uncertainty_std * 100)
        high = min(100, pct + uncertainty_std * 100)
        fig.add_annotation(
            x=0.5, y=0.25, xref="paper", yref="paper",
            text=f"±{uncertainty_std*100:.1f}% uncertainty",
            font=dict(size=11, color="#ff9800"),
            showarrow=False,
        )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def render_perplexity_chart(sentences: list[dict]):
    if not sentences:
        return None
    labels = [f"S{i+1}" for i in range(len(sentences))]
    ppls = [s["perplexity"] for s in sentences]
    colors = [
        "#ff4b4b" if s["ai_probability"] > 0.6
        else "#00c853" if s["ai_probability"] < 0.4
        else "#ff9800"
        for s in sentences
    ]
    fig = go.Figure(go.Bar(
        x=labels, y=ppls, marker_color=colors,
        text=[f"{p:.1f}" for p in ppls], textposition="auto",
    ))
    fig.update_layout(
        title="Sentence-Level Perplexity Distribution",
        xaxis_title="Sentence", yaxis_title="Perplexity",
        height=320, margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def render_fingerprint_chart(fingerprint: dict):
    """Render horizontal bar chart for multi-LLM fingerprint."""
    items = fingerprint.get("ai_source_ranking", [])
    if not items:
        return None
    labels = [i["model"] for i in items]
    probs  = [i["probability"] * 100 for i in items]
    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation="h",
        marker=dict(
            color=probs,
            colorscale=[[0,"#1e1b4b"],[0.5,"#7c3aed"],[1,"#a78bfa"]],
        ),
        text=[f"{p:.1f}%" for p in probs], textposition="auto",
    ))
    fig.update_layout(
        title="Source Model Probability Distribution",
        height=280, margin=dict(l=10, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(range=[0,100], title="Probability (%)"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ── Sentence Rendering ────────────────────────────────────────────────────────

def _highlight_words(text: str, word_attributions: list) -> str:
    if not word_attributions:
        return text
    attr_map = {w.lower(): s for w, s in word_attributions}
    words = text.split()
    parts = []
    for word in words:
        clean = word.lower().strip(".,;:!?\"'()[]{}") 
        score = attr_map.get(clean, 0.0)
        if abs(score) < 0.05:
            parts.append(word)
        elif score > 0:
            intensity = min(0.6, abs(score) * 0.6)
            parts.append(f'<span class="word-ai" style="--intensity:{intensity:.2f}">{word}</span>')
        else:
            intensity = min(0.6, abs(score) * 0.6)
            parts.append(f'<span class="word-human" style="--intensity:{intensity:.2f}">{word}</span>')
    return " ".join(parts)


def render_sentence_highlighting(sentences: list[dict]):
    for i, s in enumerate(sentences):
        prob = s["ai_probability"]
        if prob > 0.6:
            css_class, label, icon = "sentence-ai", f"AI ({prob:.0%})", "🔴"
        elif prob < 0.4:
            css_class, label, icon = "sentence-human", f"Human ({1-prob:.0%})", "🟢"
        else:
            css_class, label, icon = "sentence-uncertain", f"Uncertain ({prob:.0%})", "🟡"

        word_attrs = s.get("word_attributions", [])
        highlighted_text = _highlight_words(s["text"], word_attrs)

        token_html = ""
        top_ai = s.get("top_ai_tokens", [])
        top_human = s.get("top_human_tokens", [])
        if top_ai or top_human:
            badges = []
            for w, score in top_ai[:3]:
                badges.append(f'<span class="token-ai-badge">🔴 {w} ({score:+.2f})</span>')
            for w, score in top_human[:3]:
                badges.append(f'<span class="token-human-badge">🟢 {w} ({score:+.2f})</span>')
            token_html = f'<div class="token-indicators">{" ".join(badges)}</div>'

        st.markdown(
            f'<div class="{css_class}">'
            f'<small><b>{icon} {label}</b> | PPL: {s["perplexity"]:.1f}</small><br/>'
            f'{highlighted_text}{token_html}</div>',
            unsafe_allow_html=True,
        )


def render_confidence_badge(confidence: str, category: str):
    st.markdown(
        f'<div style="text-align:center">'
        f'<span class="confidence-badge confidence-{confidence}">{category}</span></div>',
        unsafe_allow_html=True,
    )


# ── File Extraction ───────────────────────────────────────────────────────────

def extract_text_from_upload(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    elif filename.endswith(".pdf"):
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            st.error("PDF support requires `pdfplumber`. Install: `pip install pdfplumber`")
    elif filename.endswith(".docx"):
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(uploaded_file.read()))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            st.error("DOCX support requires `python-docx`. Install: `pip install python-docx`")
    else:
        st.error(f"Unsupported file type: {filename}")
    return ""


# ── History Dashboard (#10) ───────────────────────────────────────────────────

def render_history_dashboard(history: HistoryManager):
    st.subheader("📊 Analysis History & Trends")
    stats = history.get_stats()

    if stats.get("total_count", 0) == 0:
        st.info("No history yet. Analyze some text to start tracking trends!")
        return

    total = stats["total_count"]
    verdict_bd = stats.get("verdict_breakdown", {})
    avg_prob = stats.get("avg_ai_probability", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyses", f"{total:,}")
    col2.metric("Avg AI Probability", f"{avg_prob:.0%}")
    col3.metric("AI Verdicts", verdict_bd.get("AI", 0))
    col4.metric("Human Verdicts", verdict_bd.get("Human", 0))

    # Verdict breakdown pie
    if verdict_bd:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Verdict Distribution**")
            verdict_colors = {"AI": "#ff4b4b", "Human": "#00c853", "Mixed": "#ff9800", "Unknown": "#6b7280"}
            fig_pie = go.Figure(go.Pie(
                labels=list(verdict_bd.keys()),
                values=list(verdict_bd.values()),
                marker_colors=[verdict_colors.get(k, "#6b7280") for k in verdict_bd.keys()],
                hole=0.45,
            ))
            fig_pie.update_layout(
                height=250, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("**Score Histogram**")
            hist = stats.get("score_histogram", {})
            if hist:
                buckets = sorted(hist.keys())
                counts = [hist[b] for b in buckets]
                labels = [f"{b}-{b+10}%" for b in buckets]
                colors_h = ["#00c853" if b < 40 else "#ff9800" if b < 70 else "#ff4b4b" for b in buckets]
                fig_hist = go.Figure(go.Bar(x=labels, y=counts, marker_color=colors_h))
                fig_hist.update_layout(
                    height=250, margin=dict(l=30, r=10, t=10, b=30),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"), xaxis_title="AI Probability",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # Time series
    ts = stats.get("time_series", [])
    if len(ts) > 1:
        st.markdown("**Daily Analysis Volume (last 30 days)**")
        days = [r["day"] for r in ts]
        counts = [r["count"] for r in ts]
        avg_probs = [r["avg_prob"] for r in ts]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(x=days, y=counts, name="Analyses", marker_color="#7c3aed", opacity=0.7))
        fig_ts.add_trace(go.Scatter(
            x=days, y=[p * 100 for p in avg_probs],
            name="Avg AI %", yaxis="y2",
            line=dict(color="#ff9800", width=2),
        ))
        fig_ts.update_layout(
            height=280, margin=dict(l=40, r=40, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(title="Count"),
            yaxis2=dict(title="Avg AI %", overlaying="y", side="right", range=[0,100]),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Drift
    drift = stats.get("drift", [])
    if len(drift) > 1:
        st.markdown("**Weekly AI Probability Drift (last 8 weeks)**")
        weeks = [r["week"] for r in drift]
        drift_probs = [r["avg_prob"] * 100 for r in drift]
        fig_drift = go.Figure(go.Scatter(
            x=weeks, y=drift_probs, mode="lines+markers",
            line=dict(color="#a78bfa", width=2),
            marker=dict(size=8, color="#7c3aed"),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
        ))
        fig_drift.add_hline(y=50, line_dash="dash", line_color="#6b7280",
                            annotation_text="50% threshold", annotation_position="bottom right")
        fig_drift.update_layout(
            height=240, margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), yaxis=dict(range=[0, 100]),
            yaxis_title="Avg AI Probability %",
        )
        st.plotly_chart(fig_drift, use_container_width=True)

    # Domain breakdown
    domain_bd = stats.get("domain_breakdown", {})
    if domain_bd:
        st.markdown("**Domain Breakdown**")
        d_labels = list(domain_bd.keys())
        d_counts = list(domain_bd.values())
        d_colors = ["#7c3aed","#4c1d95","#a78bfa","#6d28d9","#8b5cf6","#c4b5fd"]
        fig_domain = go.Figure(go.Bar(x=d_labels, y=d_counts, marker_color=d_colors[:len(d_labels)]))
        fig_domain.update_layout(
            height=220, margin=dict(l=40, r=20, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_domain, use_container_width=True)

    # Model fingerprint breakdown
    model_bd = stats.get("model_breakdown", {})
    if model_bd:
        st.markdown("**Detected Source Models**")
        m_labels = list(model_bd.keys())
        m_counts = list(model_bd.values())
        st.bar_chart(dict(zip(m_labels, m_counts)))

    # Recent history table
    with st.expander("📋 Recent Analyses"):
        recent = history.get_recent(25)
        if recent:
            import pandas as pd
            df = pd.DataFrame(recent)
            df["ts"] = pd.to_datetime(df["ts"], unit="s")
            df = df[["ts", "verdict", "ai_probability", "confidence", "word_count", "domain"]].copy()
            df.columns = ["Time", "Verdict", "AI Prob", "Confidence", "Words", "Domain"]
            df["AI Prob"] = df["AI Prob"].apply(lambda x: f"{x:.0%}")
            df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Clear History", type="secondary"):
        history.clear()
        st.success("History cleared!")
        st.rerun()


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    detector = None
    history = None
    report_gen = None

    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🛡️ SentinelAI — AI Content Detector")
    st.caption("Perplexity · Burstiness · Neural Classifier · Stylometry · Domain-Aware · Bayesian Uncertainty")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Loading detection models..."):
        detector = load_detector()
        history = load_history()
        report_gen = load_report_generator()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ SentinelAI Engine")
        st.markdown("**Signals active:**")

        sig_ppl   = "✅" if True else "❌"
        sig_burst = "✅" if True else "❌"
        sig_clf   = "✅" if detector.classifier.is_fine_tuned else "⚠️"
        sig_stylo = "✅" if detector._stylo is not None else "⚠️"
        sig_fp    = "✅" if detector._fingerprinter.is_available else "⚠️"

        st.markdown(f"""
        {sig_ppl} Perplexity (GPT-2)
        {sig_burst} Burstiness
        {sig_clf} {"DeBERTa Classifier" if detector.classifier.is_fine_tuned else "Classifier (not trained)"}
        {sig_stylo} {"Stylometry" if detector._stylo is not None else "Stylometry (install spaCy)"}
        {sig_fp} {"Multi-LLM Fingerprint" if detector._fingerprinter.is_available else "Fingerprint (train model)"}
        """)

        st.divider()

        st.markdown("**Ensemble Weights:**")
        for k, v in detector.weights.items():
            pct = int(v * 100)
            st.progress(pct, text=f"{k.capitalize()}: {pct}%")

        st.divider()

        # Adversarial robustness toggle
        run_adversarial = st.toggle(
            "🛡️ Adversarial Robustness Test",
            value=False,
            help="Generate paraphrases and re-score to measure evasion risk. Slower but more thorough."
        )

        # Contrastive explanation toggle
        run_explainer = st.toggle(
            "✏️ Contrastive Explanations",
            value=False,
            help="Use local Ollama LLM to suggest human-sounding rewrites. Requires Ollama running."
        )

        st.divider()

        # Browser extension instructions
        with st.expander("🌐 Browser Extension"):
            st.markdown("""
            **Load in Chrome:**
            1. Go to `chrome://extensions/`
            2. Enable Developer Mode
            3. Click "Load unpacked"
            4. Select the `browser_extension/` folder
            5. Right-click selected text → **Check with SentinelAI**

            *API must be running: `python api.py`*
            """)

        st.divider()
        st.caption("Detects: ChatGPT, GPT-4, Gemini, Claude, Llama, Mistral, DeepSeek")

    # ── Main Tabs ─────────────────────────────────────────────────────────────
    main_tab, history_tab = st.tabs(["🔍 Analyze", "📊 History"])

    with history_tab:
        render_history_dashboard(history)

    with main_tab:
        # Input tabs
        tab1, tab2 = st.tabs(["Paste Text", "Upload File"])
        text_input = ""

        with tab1:
            text_input = st.text_area(
                "Paste text to analyze:",
                height=220,
                placeholder="Enter at least 2-3 sentences for accurate detection...",
                max_chars=50000,
            )
            char_count = len(text_input) if text_input else 0
            st.caption(f"{char_count:,}/50,000 characters")

        with tab2:
            uploaded_file = st.file_uploader(
                "Upload a document",
                type=["txt", "pdf", "docx"],
                help="Supported formats: TXT, PDF, DOCX"
            )
            if uploaded_file:
                text_input = extract_text_from_upload(uploaded_file)
                if text_input:
                    st.success(f"Extracted {len(text_input):,} characters from {uploaded_file.name}")
                    with st.expander("Preview extracted text"):
                        st.text(text_input[:2000] + ("..." if len(text_input) > 2000 else ""))

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_btn = st.button("🔍 Analyze Text", use_container_width=True, type="primary")

        if analyze_btn and text_input and text_input.strip():
            # ── STREAMING ANALYSIS (#5) ────────────────────────────────────
            result_placeholder = st.empty()
            progress_bar = st.progress(0, text="Initializing...")

            with st.spinner("Analyzing text..."):
                t0 = time.time()
                progress_bar.progress(10, text="Preprocessing text...")
                result = detector.analyze(text_input)
                progress_bar.progress(100, text="Done!")
                elapsed = time.time() - t0

            progress_bar.empty()

            # Log to history (#10)
            try:
                history.log(text_input, result)
            except Exception:
                pass

            # ── Verdict ────────────────────────────────────────────────────
            verdict   = result["verdict"]
            prob      = result["ai_probability"]
            confidence = result.get("confidence", "low")
            conf_cat  = result.get("confidence_category", "")
            domain    = result.get("domain", "general")
            domain_info = result.get("domain_info", {})
            mc        = result.get("uncertainty")

            if verdict == "AI":
                st.markdown(f'<div class="verdict-ai">🤖 Verdict: AI-Generated ({prob:.0%})</div>', unsafe_allow_html=True)
            elif verdict == "Human":
                st.markdown(f'<div class="verdict-human">✅ Verdict: Human-Written ({1-prob:.0%})</div>', unsafe_allow_html=True)
            elif verdict == "Mixed":
                st.markdown(f'<div class="verdict-mixed">⚡ Verdict: Mixed Authorship</div>', unsafe_allow_html=True)
            else:
                st.info("Could not determine authorship. Please provide more text.")

            render_confidence_badge(confidence, conf_cat)

            # Domain badge (#8)
            _conf_html = (
                f"&nbsp;&nbsp;<small style=\"color:#6b7280;\">confidence: {domain_info.get('confidence','')}</small>"
                if domain_info.get("confidence") else ""
            )
            st.markdown(
                f'<div style="text-align:center; margin:8px 0;">'
                f'<span class="domain-pill">📁 Domain: {domain.capitalize()}</span>'
                f'{_conf_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


            st.write("")

            # ── Warnings ───────────────────────────────────────────────────
            metadata = result.get("metadata", {})
            if metadata.get("homoglyphs_detected"):
                st.markdown(
                    '<div class="warning-banner">⚠️ <b>Homoglyph characters detected.</b> '
                    'Unicode lookalike characters have been normalized for analysis.</div>',
                    unsafe_allow_html=True,
                )
            if not metadata.get("classifier_trained", False):
                st.markdown(
                    '<div class="warning-banner">ℹ️ DeBERTa classifier is not fine-tuned. '
                    'Results rely on perplexity and burstiness.</div>',
                    unsafe_allow_html=True,
                )

            # ── Component Breakdown ────────────────────────────────────────
            st.subheader("📊 Component Breakdown")
            comp = result["components"]

            num_cols = 3 + (1 if "stylometry" in comp else 0)
            cols = st.columns(num_cols)

            with cols[0]:
                st.metric("Perplexity", f'{comp["perplexity"]["global_ppl"]:.1f}',
                          help="Lower = more AI-like. AI: 7-15, Human: 25+")
                ppl_prob = comp["perplexity"]["ai_probability"]
                icon = "🔴" if ppl_prob > 0.6 else "🟢" if ppl_prob < 0.4 else "🟡"
                st.caption(f'{icon} AI Prob: {ppl_prob:.0%}')

            with cols[1]:
                st.metric("Burstiness", f'{comp["burstiness"]["score"]:.3f}',
                          help="Lower = more AI-like (flat writing pattern)")
                burst_prob = comp["burstiness"]["ai_probability"]
                icon = "🔴" if burst_prob > 0.6 else "🟢" if burst_prob < 0.4 else "🟡"
                st.caption(f'{icon} AI Prob: {burst_prob:.0%}')

            with cols[2]:
                clf_prob = comp["classifier"]["ai_probability"]
                is_ft = comp["classifier"]["is_fine_tuned"]
                st.metric("Classifier", f'{clf_prob:.0%}',
                          help="Neural network prediction. Most accurate when fine-tuned.")
                icon = "🔴" if clf_prob > 0.6 else "🟢" if clf_prob < 0.4 else "🟡"
                status = "Fine-tuned" if is_ft else "Not trained"
                st.caption(f'{icon} {clf_prob:.0%} ({status})')

            if "stylometry" in comp and len(cols) > 3:
                with cols[3]:
                    stylo_prob = comp["stylometry"]["ai_probability"]
                    st.metric("Stylometry", f'{stylo_prob:.0%}',
                              help="Writing style analysis (connectives, passive voice, sentence length)")
                    icon = "🔴" if stylo_prob > 0.6 else "🟢" if stylo_prob < 0.4 else "🟡"
                    st.caption(f'{icon} AI Prob: {stylo_prob:.0%}')

            # ── Gauge (#5, #7) ─────────────────────────────────────────────
            st.subheader("Overall Score")
            uncertainty_std = mc.get("std_uncertainty") if mc else None
            gauge_fig = render_gauge(prob, uncertainty_std)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # MC Dropout details (#7)
            if mc and mc.get("std_uncertainty", 0) > 0:
                with st.expander(f"🎲 MC Dropout Uncertainty — {mc['uncertainty_level'].capitalize()} uncertainty"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean AI Prob", f'{mc["mean_ai_probability"]:.0%}')
                    c2.metric("Std Dev (±σ)", f'{mc["std_uncertainty"]:.3f}')
                    c3.metric("Uncertainty", mc["uncertainty_level"].capitalize())
                    if mc.get("samples"):
                        fig_mc = go.Figure(go.Histogram(
                            x=mc["samples"], nbinsx=10,
                            marker_color="#7c3aed", opacity=0.8,
                        ))
                        fig_mc.update_layout(
                            title="MC Dropout Sample Distribution",
                            height=200, margin=dict(l=30, r=10, t=40, b=30),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white"), xaxis_title="AI Probability",
                        )
                        st.plotly_chart(fig_mc, use_container_width=True)

            # ── Multi-LLM Fingerprinting (#1) ──────────────────────────────
            fingerprint = result.get("fingerprint")
            if fingerprint:
                st.subheader("🔬 Source Model Fingerprint")
                top_lbl = fingerprint["top_label"]
                top_prob = fingerprint["top_probability"]
                fp_color = "#ff4b4b" if top_lbl != "Human" else "#00c853"
                st.markdown(
                    f'<p style="font-size:1.1rem;font-weight:700;color:{fp_color};">'
                    f'Most likely source: {top_lbl} ({top_prob:.0%})</p>',
                    unsafe_allow_html=True,
                )
                fp_fig = render_fingerprint_chart(fingerprint)
                if fp_fig:
                    st.plotly_chart(fp_fig, use_container_width=True)
            else:
                st.info("💡 **Multi-LLM Fingerprinting**: Train the fingerprint model on Colab to identify *which* AI wrote this. Run `training/train_fingerprint.py`.")

            # ── Stylometry Details (#3) ─────────────────────────────────────
            if "stylometry" in comp:
                stylo = comp["stylometry"]
                flagged = stylo.get("flagged_connectives", [])
                if flagged:
                    with st.expander(f"✍️ Stylometric Analysis — {len(flagged)} AI phrase(s) detected"):
                        st.markdown("**AI-characteristic connective phrases found:**")
                        pills_html = " ".join(f'<span class="conn-pill">{c}</span>' for c in flagged)
                        st.markdown(f'<div style="margin:8px 0">{pills_html}</div>', unsafe_allow_html=True)

                        feats = stylo.get("features", {})
                        if feats:
                            f1, f2, f3 = st.columns(3)
                            f1.metric("Connective Rate", f'{feats.get("connective_rate", 0):.2f}/100w')
                            f2.metric("Passive Voice", f'{feats.get("passive_ratio", 0):.0%}')
                            f3.metric("Avg Sent Len", f'{feats.get("avg_sentence_length", 0):.0f} words')

            # ── Perplexity Chart ────────────────────────────────────────────
            if result["sentences"]:
                ppl_fig = render_perplexity_chart(result["sentences"])
                if ppl_fig:
                    st.plotly_chart(ppl_fig, use_container_width=True)

            # ── Sentence Highlighting ───────────────────────────────────────
            st.subheader("Sentence-Level Analysis")
            st.caption("🟢 Human | 🟡 Uncertain | 🔴 AI — individual words highlighted by attribution score")
            if result["sentences"]:
                render_sentence_highlighting(result["sentences"])
            else:
                st.info("No sentences to analyze. Provide longer text.")

            # ── Contrastive Explanation (#9) ───────────────────────────────
            if run_explainer:
                st.subheader("✏️ Humanization Suggestions")
                explainer = load_explainer()
                with st.spinner("Generating suggestions with Ollama..."):
                    expl = explainer.explain(result["sentences"])

                if not expl.get("available"):
                    st.warning(f"Ollama not available: {expl.get('reason', '')}")
                elif not expl.get("explanations"):
                    st.info(expl.get("message", "No strongly AI-flagged sentences to explain."))
                else:
                    st.caption("Below are suggested rewrites for the most AI-flagged sentences:")
                    for e in expl["explanations"]:
                        st.markdown(
                            f'<div class="suggestion-block">'
                            f'<div class="suggestion-label">🔴 ORIGINAL ({e["ai_probability"]:.0%} AI) — {e["change_level"]}</div>'
                            f'<div class="suggestion-original">{e["original"]}</div>'
                            f'<div class="suggestion-label">🟢 SUGGESTED REWRITE</div>'
                            f'<div class="suggestion-rewritten">{e["rewritten"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # ── Adversarial Robustness (#2) ────────────────────────────────
            if run_adversarial:
                st.subheader("🛡️ Adversarial Robustness Test")
                adv_tester = load_adversarial_tester()
                with st.spinner("Generating paraphrases and re-scoring..."):
                    adv = adv_tester.test(text_input, result, detector)

                if not adv.get("enabled"):
                    st.warning(f"Adversarial test disabled: {adv.get('error', 'disabled in config')}")
                elif adv.get("skipped"):
                    st.info(adv.get("reason", "Skipped"))
                else:
                    risk = adv["evasion_risk"]
                    risk_class = {"High": "evasion-high", "Medium": "evasion-medium", "Low": "evasion-low"}[risk]
                    risk_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk]
                    st.markdown(
                        f'<div class="{risk_class}"><b>{risk_icon} Evasion Risk: {risk}</b><br/>'
                        f'{adv["evasion_risk_description"]}</div>',
                        unsafe_allow_html=True,
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Evasion Rate", f'{adv["evasion_rate"]:.0%}',
                              help="% of paraphrases that flip to Human verdict")
                    c2.metric("Prob Drop", f'{adv["probability_drop"]:.0%}',
                              help="How much AI probability drops after paraphrasing")
                    c3.metric("Paraphrases Tested", adv["num_paraphrases_tested"])

                    with st.expander("See paraphrase results"):
                        for i, pr in enumerate(adv.get("paraphrase_results", [])):
                            v_icon = "🔴" if pr["verdict"] == "AI" else "🟢"
                            st.markdown(f"**Paraphrase {i+1}:** {v_icon} {pr['verdict']} ({pr['ai_probability']:.0%})")
                            st.caption(pr["text_preview"])

            # ── PDF/HTML Export (#6) ────────────────────────────────────────
            st.divider()
            st.subheader("📥 Export Report")
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                html_bytes = report_gen.render_html(text_input, result)
                st.download_button(
                    label="⬇️ Download HTML Report",
                    data=html_bytes,
                    file_name="sentinelai_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

            with exp_col2:
                pdf_bytes = report_gen.render_pdf(text_input, result)
                if pdf_bytes:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name="sentinelai_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "⬇️ PDF (install reportlab)",
                        disabled=True,
                        use_container_width=True,
                        help="pip install reportlab",
                    )

            # ── Document Info ──────────────────────────────────────────────
            if metadata:
                with st.expander("📋 Document Info"):
                    info_cols = st.columns(4)
                    info_cols[0].metric("Characters", f'{metadata.get("original_length", 0):,}')
                    info_cols[1].metric("Sentences", metadata.get("num_sentences", 0))
                    info_cols[2].metric("Language", metadata.get("language", "unknown").upper())
                    info_cols[3].metric("Domain", domain.capitalize())
                    st.caption(f"Analysis completed in {elapsed:.2f}s")

            with st.expander("🔧 Raw Detection Data"):
                st.json(result)

        elif analyze_btn:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
