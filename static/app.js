const listEl = document.getElementById("conv-list");
const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const PLACEHOLDER_TITLE = "新對話";
let currentId = null;
let streaming = false;

function toLogin() {
  window.location.href = "/login.html";
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (res.status === 401) {
    toLogin();
    throw new Error("unauthorized");
  }
  return res;
}

function detailText(data, status) {
  const d = data && data.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => e.msg || JSON.stringify(e)).join("\n");
  return "發生錯誤（" + status + "）";
}

function addBubble(role, text) {
  const el = document.createElement("div");
  el.className = "bubble " + role;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function setStreaming(on) {
  streaming = on;
  sendEl.disabled = on;
}

async function loadList() {
  const res = await api("/api/conversations");
  const items = await res.json();
  listEl.textContent = "";
  for (const c of items) {
    const li = document.createElement("li");
    li.className = "conv-item" + (c.id === currentId ? " active" : "");
    const title = document.createElement("span");
    title.className = "conv-title";
    title.textContent = c.title || PLACEHOLDER_TITLE;
    const del = document.createElement("button");
    del.className = "conv-delete";
    del.textContent = "刪除";
    del.addEventListener("click", (ev) => {
      ev.stopPropagation();
      removeConversation(c.id);
    });
    li.addEventListener("click", () => selectConversation(c.id));
    li.append(title, del);
    listEl.appendChild(li);
  }
}

async function selectConversation(id) {
  if (streaming) return;
  currentId = id;
  messagesEl.textContent = "";
  const res = await api("/api/conversations/" + id);
  if (!res.ok) {
    currentId = null;
    await loadList();
    return;
  }
  const conv = await res.json();
  for (const m of conv.messages) addBubble(m.role, m.content);
  await loadList();
}

async function removeConversation(id) {
  if (streaming) return;
  await api("/api/conversations/" + id, { method: "DELETE" });
  if (id === currentId) {
    currentId = null;
    messagesEl.textContent = "";
  }
  await loadList();
}

async function newConversation() {
  if (streaming) return;
  const res = await api("/api/conversations", { method: "POST" });
  const conv = await res.json();
  await selectConversation(conv.id);
}

async function sendMessage() {
  const text = inputEl.value;
  if (streaming || !text.trim()) return;
  setStreaming(true);
  try {
    if (currentId === null) {
      const res = await api("/api/conversations", { method: "POST" });
      currentId = (await res.json()).id;
      messagesEl.textContent = "";
    }
    inputEl.value = "";
    addBubble("user", text);
    const bubble = addBubble("assistant", "");
    try {
      const res = await api("/api/conversations/" + currentId + "/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: text }),
      });
      if (!res.ok) {
        let data = null;
        try { data = await res.json(); } catch (e) { /* no body */ }
        bubble.textContent = detailText(data, res.status);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          bubble.textContent += decoder.decode(value, { stream: true });
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
        bubble.textContent += decoder.decode();
      } catch (e) {
        bubble.textContent += "（回覆中斷）";
      }
    } catch (e) {
      if (e.message !== "unauthorized") bubble.textContent = "無法連線到伺服器";
    }
  } catch (e) {
    /* unauthorized: already redirected */
  } finally {
    setStreaming(false);
    await loadList().catch(() => {});
  }
}

document.getElementById("composer").addEventListener("submit", (ev) => {
  ev.preventDefault();
  sendMessage();
});
inputEl.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    sendMessage();
  }
});
document.getElementById("new-chat").addEventListener("click", newConversation);
document.getElementById("logout").addEventListener("click", async () => {
  await fetch("/api/auth/logout", { method: "POST" });
  toLogin();
});

(async function init() {
  try {
    const res = await api("/api/me");
    document.getElementById("user-name").textContent = (await res.json()).username;
    await loadList();
  } catch (e) { /* redirected on 401 */ }
})();
