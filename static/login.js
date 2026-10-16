const form = document.getElementById("form");
const errorEl = document.getElementById("error");
const titleEl = document.getElementById("title");
const submitEl = document.getElementById("submit");
const toggleEl = document.getElementById("toggle");
let mode = "login";

toggleEl.addEventListener("click", () => {
  mode = mode === "login" ? "register" : "login";
  const isLogin = mode === "login";
  titleEl.textContent = isLogin ? "登入" : "註冊";
  submitEl.textContent = isLogin ? "登入" : "註冊";
  toggleEl.textContent = isLogin ? "沒有帳號？註冊" : "已有帳號？登入";
  errorEl.textContent = "";
});

function errorText(data, status) {
  const d = data && data.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => e.msg || JSON.stringify(e)).join("\n");
  return "發生錯誤（" + status + "）";
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch (e) { /* no body */ }
  return { res, data };
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  errorEl.textContent = "";
  submitEl.disabled = true;
  const creds = {
    username: document.getElementById("username").value,
    password: document.getElementById("password").value,
  };
  try {
    if (mode === "register") {
      const r = await post("/api/auth/register", creds);
      if (!r.res.ok) { errorEl.textContent = errorText(r.data, r.res.status); return; }
    }
    const l = await post("/api/auth/login", creds);
    if (!l.res.ok) { errorEl.textContent = errorText(l.data, l.res.status); return; }
    window.location.href = "/";
  } catch (e) {
    errorEl.textContent = "無法連線到伺服器";
  } finally {
    submitEl.disabled = false;
  }
});
