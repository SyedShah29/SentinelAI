// content.js — Injected into every page to show detection results inline

const PANEL_ID = "ai-detector-panel";

// Remove existing panel
function removePanel() {
  const existing = document.getElementById(PANEL_ID);
  if (existing) existing.remove();
}

// Create and inject the results panel
function createPanel() {
  removePanel();
  const panel = document.createElement("div");
  panel.id = PANEL_ID;
  panel.className = "ai-detector-panel";
  document.body.appendChild(panel);
  return panel;
}

// Build color class from verdict
function verdictClass(verdict) {
  return { AI: "ai", Human: "human", Mixed: "mixed" }[verdict] || "unknown";
}

// Build probability bar HTML
function probBar(prob, label) {
  const color = prob > 0.6 ? "#ff4b4b" : prob < 0.4 ? "#00c853" : "#ff9800";
  const pct = Math.round(prob * 100);
  return `
    <div class="ai-d-bar-row">
      <span class="ai-d-bar-label">${label}</span>
      <div class="ai-d-bar-bg">
        <div class="ai-d-bar-fill" style="width:${pct}%;background:${color};"></div>
      </div>
      <span class="ai-d-bar-pct">${pct}%</span>
    </div>`;
}

// Build sentence items HTML
function buildSentenceItems(sentences) {
  if (!sentences || sentences.length === 0) return "";
  const items = sentences.slice(0, 8).map(s => {
    const p = s.ai_probability;
    const cls = p > 0.6 ? "ai" : p < 0.4 ? "human" : "mixed";
    const icon = p > 0.6 ? "🔴" : p < 0.4 ? "🟢" : "🟡";
    const label = p > 0.6 ? `AI (${Math.round(p*100)}%)` : p < 0.4 ? `Human (${Math.round((1-p)*100)}%)` : `Uncertain`;
    return `<div class="ai-d-sentence ${cls}">
      <div class="ai-d-sent-label">${icon} ${label}</div>
      <div class="ai-d-sent-text">${escHtml(s.text.slice(0, 120))}${s.text.length > 120 ? "…" : ""}</div>
    </div>`;
  });
  return items.join("");
}

function escHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// Show loading state
function showLoading(text) {
  const panel = createPanel();
  panel.innerHTML = `
    <div class="ai-d-header">
      <span class="ai-d-title">🔍 AI Text Detector</span>
      <button class="ai-d-close" onclick="document.getElementById('${PANEL_ID}').remove()">✕</button>
    </div>
    <div class="ai-d-loading">
      <div class="ai-d-spinner"></div>
      <div>Analyzing ${text.split(" ").length} words...</div>
    </div>`;
}

// Show result
function showResult(result, text) {
  const panel = createPanel();
  const vc = verdictClass(result.verdict);
  const prob = result.ai_probability;
  const pct = Math.round(prob * 100);
  const comp = result.components || {};
  const ppl = comp.perplexity?.ai_probability || 0;
  const burst = comp.burstiness?.ai_probability || 0;
  const clf = comp.classifier?.ai_probability || 0;

  const fingerprintHTML = result.fingerprint
    ? `<div class="ai-d-section-title">Source Model</div>
       <div class="ai-d-fingerprint">${result.fingerprint.top_label} <span class="ai-d-fp-prob">${Math.round(result.fingerprint.top_probability * 100)}%</span></div>`
    : "";

  const domainHTML = result.domain
    ? `<span class="ai-d-domain-pill">📁 ${result.domain}</span>`
    : "";

  panel.innerHTML = `
    <div class="ai-d-header">
      <span class="ai-d-title">🔍 AI Text Detector</span>
      <button class="ai-d-close" onclick="document.getElementById('${PANEL_ID}').remove()">✕</button>
    </div>
    <div class="ai-d-verdict ${vc}">
      ${result.verdict === "AI" ? "🤖 AI-Generated" : result.verdict === "Human" ? "✅ Human-Written" : result.verdict === "Mixed" ? "⚡ Mixed" : "❓ Unknown"}
      — ${pct}%
    </div>
    <div class="ai-d-conf">${result.confidence_category || ""} ${domainHTML}</div>

    <div class="ai-d-section-title">Component Scores</div>
    ${probBar(prob, "Overall")}
    ${probBar(ppl, "Perplexity")}
    ${probBar(burst, "Burstiness")}
    ${probBar(clf, "Classifier")}

    ${fingerprintHTML}

    <div class="ai-d-section-title">Sentence Analysis</div>
    <div class="ai-d-sentences">
      ${buildSentenceItems(result.sentences)}
    </div>

    <div class="ai-d-footer">
      ${text.split(" ").length} words &nbsp;|&nbsp; ${result.metadata?.num_sentences || 0} sentences
    </div>`;
}

// Show error
function showError(message) {
  const panel = createPanel();
  panel.innerHTML = `
    <div class="ai-d-header">
      <span class="ai-d-title">🔍 AI Text Detector</span>
      <button class="ai-d-close" onclick="document.getElementById('${PANEL_ID}').remove()">✕</button>
    </div>
    <div class="ai-d-error">${escHtml(message)}</div>`;
}

// Listen for messages from background
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SHOW_LOADING") showLoading(msg.text);
  else if (msg.type === "SHOW_RESULT") showResult(msg.result, msg.text);
  else if (msg.type === "SHOW_ERROR") showError(msg.message);
});
