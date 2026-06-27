// background.js — Service worker for SentinelAI browser extension

const API_BASE = "http://localhost:8000/api/v1";

// Create right-click context menu on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "ai-detect-selection",
    title: "🛡️ Check with SentinelAI",
    contexts: ["selection"],
  });
});

// Handle context menu click
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "ai-detect-selection") return;

  const selectedText = info.selectionText?.trim();
  if (!selectedText || selectedText.length < 20) {
    chrome.tabs.sendMessage(tab.id, {
      type: "SHOW_ERROR",
      message: "Please select at least 20 characters of text.",
    });
    return;
  }

  // Tell content script to show loading state
  chrome.tabs.sendMessage(tab.id, {
    type: "SHOW_LOADING",
    text: selectedText,
  });

  // Get API key from storage
  const { apiKey } = await chrome.storage.sync.get("apiKey");

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey || "dev-key",
      },
      body: JSON.stringify({
        text: selectedText,
        include_sentences: true,
        include_metadata: true,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    chrome.tabs.sendMessage(tab.id, {
      type: "SHOW_RESULT",
      result,
      text: selectedText,
    });

    // Save to extension history
    saveToHistory(selectedText, result);

  } catch (error) {
    chrome.tabs.sendMessage(tab.id, {
      type: "SHOW_ERROR",
      message: `Detection failed: ${error.message}. Make sure the API is running (python api.py).`,
    });
  }
});

// Save recent detection to chrome.storage
async function saveToHistory(text, result) {
  const { history = [] } = await chrome.storage.local.get("history");
  history.unshift({
    timestamp: Date.now(),
    textPreview: text.slice(0, 100) + (text.length > 100 ? "..." : ""),
    verdict: result.verdict,
    ai_probability: result.ai_probability,
    confidence: result.confidence,
  });
  // Keep last 50 entries
  if (history.length > 50) history.splice(50);
  await chrome.storage.local.set({ history });
}

// Handle messages from popup
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "GET_API_STATUS") {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(data => sendResponse({ ok: true, data }))
      .catch(() => sendResponse({ ok: false }));
    return true; // async
  }
});
