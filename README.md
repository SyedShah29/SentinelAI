# SentinelAI — AI Content Detection Platform

## Quick Start

```bash
# 1. Install all dependencies (run this ONCE)
py -3.11 -m pip install -r requirements.txt

# 2. Download spaCy language model (run this ONCE)
py -3.11 -m spacy download en_core_web_sm

# 3. Run the web app
py -3.11 -m streamlit run app.py
```

---

## Detection Capabilities

### ✅ Works immediately (no setup)
| Feature | Status |
|---------|--------|
| Stylometry Analysis | ✅ After `spacy download en_core_web_sm` |
| Streaming Analysis | ✅ Built into app |
| PDF/HTML Export | ✅ After `pip install reportlab jinja2` |
| MC Dropout Uncertainty | ✅ Works after classifier is fine-tuned |
| Domain-Aware Detection | ✅ Always active |
| History Dashboard | ✅ Always active (stored in `data/history.db`) |

### ⚙️ Requires extra setup
| Feature | What to do |
|---------|-----------| 
| Contrastive Explanations | `winget install Ollama.Ollama` then `ollama pull llama3.2` |
| Browser Extension | Load `browser_extension/` in Chrome (see below) |

### 🏋️ Requires Colab training
| Feature | Training Script |
|---------|---------------|
| Multi-LLM Fingerprinting | `training/train_fingerprint.py` on Colab T4 GPU |
| Adversarial Robustness | Auto-downloads T5 model on first use |

---

## Installing Ollama (for Contrastive Explanations)

```powershell
winget install Ollama.Ollama
# After install, open a NEW terminal:
ollama pull llama3.2
# Ollama starts automatically as a background service
```

---

## Loading the Browser Extension

1. Open Chrome → `chrome://extensions/`
2. Toggle **Developer Mode** (top right)
3. Click **Load unpacked**
4. Select the `browser_extension/` folder
5. The extension icon appears in the toolbar
6. **Right-click any selected text** → "🛡️ Check with SentinelAI"

> **Note**: The REST API must be running for the extension to work:
> ```bash
> py -3.11 api.py
> ```

---

## Training Multi-LLM Fingerprinter — Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook → Runtime → **Change runtime type → T4 GPU**
3. Upload `training/train_fingerprint.py`
4. Run all cells (~3-4 hours on free T4)
5. Download `fingerprint_best.zip`
6. Extract to: `models/fingerprint/best/`
7. Restart the app — fingerprinting activates automatically

---

## Project Structure

```
SentinelAI/
├── detector/
│   ├── ensemble.py          ← Wires ALL signals together
│   ├── perplexity.py        ← Domain-aware thresholds
│   ├── classifier.py        ← MC Dropout enabled
│   ├── stylometry.py        ← Writing style signal
│   ├── domain_classifier.py ← Domain detection
│   ├── adversarial.py       ← Evasion risk testing
│   ├── explainer.py         ← Ollama humanization suggestions
│   └── fingerprint.py       ← Multi-LLM source identification
├── store/
│   └── history.py           ← SQLite history tracking
├── export/
│   ├── report_generator.py  ← PDF/HTML export
│   └── templates/report.html
├── browser_extension/       ← Chrome extension
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── popup.html / popup.js
│   └── styles.css
├── training/
│   └── train_fingerprint.py ← Colab training script
├── app.py                   ← Main Streamlit UI
├── api.py                   ← FastAPI REST server
├── config.py                ← All settings
└── requirements.txt         ← All dependencies
```
