// popup.js — Extension popup logic

// Tab switching
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
  });
});

// Check API status
async function checkApiStatus() {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const { apiUrl } = await chrome.storage.sync.get("apiUrl");
    const base = apiUrl || "http://localhost:8000";
    const resp = await chrome.runtime.sendMessage({ type: "GET_API_STATUS" });
    if (resp?.ok) {
      dot.classList.add("ok");
      dot.classList.remove("err");
      const clf = resp.data.classifier_trained ? "Fine-tuned" : "Base model";
      text.textContent = `API ready — ${clf}`;
    } else {
      throw new Error("Not available");
    }
  } catch {
    dot.classList.add("err");
    dot.classList.remove("ok");
    document.getElementById("statusText").textContent = "API offline — run: python api.py";
  }
}

// Render history
async function renderHistory() {
  const { history = [] } = await chrome.storage.local.get("history");
  const list = document.getElementById("historyList");

  if (history.length === 0) {
    list.innerHTML = '<div class="empty-state">No detections yet.<br/>Right-click selected text to analyze.</div>';
    return;
  }

  list.innerHTML = history.map(item => {
    const vc = { AI: "ai", Human: "human", Mixed: "mixed" }[item.verdict] || "unknown";
    const pct = Math.round(item.ai_probability * 100);
    const time = new Date(item.timestamp).toLocaleString();
    const icon = { AI: "🔴", Human: "🟢", Mixed: "🟡" }[item.verdict] || "❓";
    return `<div class="history-item ${vc}">
      <div class="hist-verdict ${vc}">${icon} ${item.verdict} — ${pct}%</div>
      <div class="hist-text">${escHtml(item.textPreview)}</div>
      <div class="hist-time">${time}</div>
    </div>`;
  }).join("");
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Clear history
document.getElementById("clearHistory").addEventListener("click", async () => {
  await chrome.storage.local.set({ history: [] });
  renderHistory();
});

// Load settings
async function loadSettings() {
  const { apiKey, apiUrl } = await chrome.storage.sync.get(["apiKey", "apiUrl"]);
  document.getElementById("apiKeyInput").value = apiKey || "";
  document.getElementById("apiUrlInput").value = apiUrl || "http://localhost:8000";
}

// Save settings
document.getElementById("saveSettings").addEventListener("click", async () => {
  const apiKey = document.getElementById("apiKeyInput").value.trim();
  const apiUrl = document.getElementById("apiUrlInput").value.trim();
  await chrome.storage.sync.set({ apiKey, apiUrl });
  const ok = document.getElementById("saveOk");
  ok.style.display = "block";
  setTimeout(() => { ok.style.display = "none"; }, 2000);
});

// Init
checkApiStatus();
renderHistory();
loadSettings();
