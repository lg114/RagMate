"use strict";
/* RagMate 前端：单页应用，无构建、无依赖。
布局：侧栏导航切三个视图（对话 / 文档 / 设置），历史对话只在对话视图下；
接口：POST /chat/stream（SSE，debug:true 以取回检索轨迹）、
GET /chat/sessions、GET /chat/sessions/{id}、DELETE /chat/sessions/{id}、
GET /documents、POST /documents/upload、DELETE /documents/{name}、
POST /ingest、GET /ingest/status、GET /ready。
界面中英双语（localStorage ragmate_lang，默认跟浏览器语言）。
引用只含文件名与页码（后端 public_citations 的口径）；检索详情里的
片段预览来自 debug 轨迹（每块截断 240 字），默认折叠。 */
const AUTH_TOKEN_KEY = "ragmate_auth_token";
const SESSION_KEY = "ragmate_session_id";
const LANG_KEY = "ragmate_lang";

const I18N = {
zh: {
navChat: "对话", navDocuments: "文档", navSettings: "设置",
newChat: "新对话", history: "历史对话", historyEmpty: "暂无历史对话",
emptyHint: "还没有提问。直接在下面输入。",
placeholder: "基于已入库的文档提问", send: "发送", stop: "停止",
docsTitle: "已入库的文档", docsEmpty: "还没有入库的文档，先上传一份。",
dropzone: "拖入文件或点这里上传新文档",
uploading: "正在上传", ingesting: "正在入库", ingestDone: "入库完成",
settingsTitle: "设置", tokenPlaceholder: "API 令牌（默认空，不用填）",
tokenSave: "保存", tokenNote: "令牌只保存在本浏览器，用于 API 鉴权；服务端默认不设鉴权，不用填。",
tokenSaved: "令牌已保存（仅本浏览器）", tokenSaveFail: "浏览器不允许保存",
readinessTitle: "服务状态", readyOk: "就绪", readyBad: "异常", readyUnknown: "未连接",
modelLabel: "模型", deployLabel: "部署", modelMissing: "未配置",
deployLocal: "本地 · 文档不出本机", deployRemote: "远程 API · 检索片段会发给服务商",
llmCheck: "模型服务",
llmNotConfigured: "没有配置模型（LLM_MODEL 为空）",
llmUnreachable: "连不上模型端点，确认服务已启动",
llmHttpError: "模型端点返回错误，检查 API Key 与地址",
llmModelMissing: "端点在线，但该模型没有拉取或不存在",
sourceTitle: "来源", retrievalTitle: "检索详情",
retrievalSummary: "{candidates} 个候选，{returned} 个留下",
retrievalMeta: "最高分 {top} · 阈值 {threshold}",
retrievalDropped: "另有 {n} 个候选被过滤",
pageLabel: "第 {n} 页", chunksLabel: "{n} 块",
statusDone: "回答已完成",
statusHedged: "低置信回答，请对照引用来源核对后再采信",
statusRefused: "没有足够资料，已拒答",
statusFailed: "这次没有生成完", statusStopped: "已停止生成",
unsupportedNote: "注意：部分说法未能在检索资料中找到依据，请以上方引用来源为准。",
noReply: "没有收到回复",
err401: "需要 API 令牌。请在设置里填写后重试。",
err413: "文件超过大小限制", err429: "请求过于频繁，请稍后再试。",
errNetwork: "网络错误，上传失败", errUploadCancelled: "上传已取消",
errGeneric: "请求失败，请重试", errConnLost: "连接意外断开",
deleteDocConfirm: "删除“{name}”？文档与向量都会删除。",
replaceConfirm: "“{name}”已存在。用新文件替换它？",
deleteSessionConfirm: "删除这段对话？",
docIngested: "已入库", docUploaded: "待入库", docIngesting: "入库中", docFailed: "入库失败",
pwdDesc: "此实例已设置访问密码，请输入以继续。",
pwdPlaceholder: "请输入实例访问密码",
pwdUnlockBtn: "解锁访问",
pwdRequired: "请输入密码",
pwdIncorrect: "密码错误，请重新输入",
pwdUnlocked: "实例解锁成功",
},
en: {
navChat: "Chat", navDocuments: "Documents", navSettings: "Settings",
newChat: "New chat", history: "History", historyEmpty: "No conversations yet",
emptyHint: "No questions yet. Type below to ask.",
placeholder: "Ask about your indexed documents", send: "Send", stop: "Stop",
docsTitle: "Documents", docsEmpty: "No documents yet — upload one to begin.",
dropzone: "Drop files here or click to upload",
uploading: "Uploading", ingesting: "Ingesting", ingestDone: "Ingest complete",
settingsTitle: "Settings", tokenPlaceholder: "API token (optional, empty by default)",
tokenSave: "Save", tokenNote: "The token is stored only in this browser, for API auth. The server sets no auth by default — leave it empty.",
tokenSaved: "Token saved (this browser only)", tokenSaveFail: "Browser refused to save",
readinessTitle: "Service status", readyOk: "Ready", readyBad: "Unavailable", readyUnknown: "Not connected",
modelLabel: "Model", deployLabel: "Deployment", modelMissing: "Not configured",
deployLocal: "Local · documents stay on this machine", deployRemote: "Remote API · retrieved passages are sent to the provider",
llmCheck: "Model service",
llmNotConfigured: "No model configured (LLM_MODEL is empty)",
llmUnreachable: "Model endpoint unreachable — is the service running?",
llmHttpError: "Model endpoint returned an error — check the API key and base URL",
llmModelMissing: "Endpoint is up, but the model is not available there",
sourceTitle: "Sources", retrievalTitle: "Retrieval details",
retrievalSummary: "{candidates} candidates, {returned} kept",
retrievalMeta: "Top score {top} · threshold {threshold}",
retrievalDropped: "{n} more candidates filtered out",
pageLabel: "Page {n}", chunksLabel: "{n} chunks",
statusDone: "Done",
statusHedged: "Low-confidence answer — check it against the cited sources",
statusRefused: "Not enough evidence to answer reliably",
statusFailed: "Generation did not finish", statusStopped: "Stopped",
unsupportedNote: "Note: some claims were not found in the retrieved sources — rely on the cited sources above.",
noReply: "No reply received",
err401: "API token required. Add it in Settings and retry.",
err413: "File exceeds the size limit", err429: "Too many requests — try again later.",
errNetwork: "Network error — upload failed", errUploadCancelled: "Upload cancelled",
errGeneric: "Request failed — please retry", errConnLost: "Connection lost unexpectedly",
deleteDocConfirm: "Delete “{name}”? The document and its vectors will be removed.",
replaceConfirm: "“{name}” already exists. Replace it with the new file?",
deleteSessionConfirm: "Delete this conversation?",
docIngested: "Ingested", docUploaded: "Pending", docIngesting: "Ingesting", docFailed: "Failed",
pwdDesc: "Instance password required to access this server.",
pwdPlaceholder: "Instance password",
pwdUnlockBtn: "Unlock Instance",
pwdRequired: "Please enter the instance password",
pwdIncorrect: "Incorrect instance password",
pwdUnlocked: "Instance unlocked successfully",
},
};

let LANG = "zh";
try {
const saved = localStorage.getItem(LANG_KEY);
LANG = saved === "en" || saved === "zh"? saved
: (navigator.language || "").toLowerCase().startsWith("zh")? "zh": "en";
} catch { /* private mode */}
function t(key, vars) {
let s = (I18N[LANG] && I18N[LANG][key]) || I18N.zh[key] || key;
if (vars) for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, String(v));
return s;
}
function applyI18n() {
document.documentElement.lang = LANG === "zh"? "zh-CN": "en";
document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n);});
document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh);});
setSendMode(streaming? "streaming": "idle");
}

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const scrollBox = $("scroll");
const input = $("input");
const sendBtn = $("send");
const toastEl = $("toast");
let toastTimer = null;
let streaming = false;
let aborter = null;
let lastDocs = [];
let lastSessions = [];
let lastReady = null;
let currentView = "chat";

function toast(text) {
toastEl.textContent = text;
toastEl.classList.add("show");
clearTimeout(toastTimer);
toastTimer = setTimeout(() => toastEl.classList.remove("show"), 3000);
}
function confirm(message) {
return new Promise((resolve) => {
  const overlay = $("modal-overlay");
  const msgEl = $("modal-message");
  const cancelBtn = $("modal-cancel");
  const confirmBtn = $("modal-confirm");
  msgEl.textContent = message;
  overlay.hidden = false;
  const cleanup = () => {
    overlay.hidden = true;
    cancelBtn.removeEventListener("click", onCancel);
    confirmBtn.removeEventListener("click", onConfirm);
  };
  const onCancel = () => { cleanup(); resolve(false); };
  const onConfirm = () => { cleanup(); resolve(true); };
  cancelBtn.addEventListener("click", onCancel);
  confirmBtn.addEventListener("click", onConfirm);
});
}
function esc(s) {
return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function scrollToBottom() {
scrollBox.scrollTop = scrollBox.scrollHeight;
}
function getAuthToken() {
  try { return (sessionStorage.getItem(AUTH_TOKEN_KEY) || "").trim(); } catch { return ""; }
}
function setAuthToken(val) {
  try {
    if (val) sessionStorage.setItem(AUTH_TOKEN_KEY, val);
    else sessionStorage.removeItem(AUTH_TOKEN_KEY);
  } catch {}
}
function authHeaders(extra) {
  const headers = { ...(extra || {}) };
  const token = getAuthToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}
function errorFrom(status, data) {
if (status === 401) return t("err401");
if (status === 413) return (data && data.detail) || t("err413");
if (status === 429) return t("err429");
const detail = data && (data.detail || data.message);
return detail || t("errGeneric");
}
async function request(method, path, body) {
const opts = { method, headers: authHeaders()};
if (body instanceof FormData) opts.body = body;
else if (body!== undefined) {
opts.headers["Content-Type"] = "application/json";
opts.body = JSON.stringify(body);
}
const res = await fetch(path, opts);
const data = await res.json().catch(() => ({}));
if (!res.ok) {
  if (res.status === 401) {
    showUnlockOverlay("Session expired or password required.");
  }
  throw new Error(errorFrom(res.status, data));
}
return data;
}
function peekSessionId() {
try { return sessionStorage.getItem(SESSION_KEY) || null;} catch { return null;}
}
function getSessionId() {
let sid = peekSessionId();
if (!sid) {
sid = (crypto.randomUUID && crypto.randomUUID()) || `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;
try { sessionStorage.setItem(SESSION_KEY, sid);} catch { /* ignore */}
}
return sid;
}
function setSessionId(sid) {
try { sessionStorage.setItem(SESSION_KEY, sid);} catch { /* ignore */}
}

/* ---- 最小 Markdown：转义 → 代码块 → 行内代码 → 加粗 → 链接 → 分段 ---- */
function renderMarkdown(src) {
const parts = src.split(/(```[\s\S]*?```)/g);
const html = parts.map((part) => {
if (part.startsWith("```")) {
const body = part.replace(/^```[^\n]*\n?/, "").replace(/```$/, "");
return `<pre><code>${esc(body)}</code></pre>`;
}
let x = esc(part);
x = x.replace(/`([^`\n]+)`/g, "<code>$1</code>");
x = x.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
x = x.replace(/https?:\/\/[^\s<>"')\]]+/g, (url) => {
const clean = url.replace(/[.,;:!?]+$/, "");
const tail = url.slice(clean.length);
return `<a href="${clean}" target="_blank" rel="noopener noreferrer">${clean}</a>${tail}`;
});
return x.split(/\n{2,}/).map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`).join("");
});
return html.join("");
}
function splitSourceLine(text) {
const m = text.match(/\n\s*数据来源：([^\n]*)\s*$/);
if (!m) return { body: text, citations: []};
const cites = [];
for (const chunk of m[1].split("、")) {
const mm = chunk.match(/【(.+?)(?:[,，]\s*第(\d+)页)?】/);
if (mm) cites.push({ source: mm[1], page: mm[2]? Number(mm[2]): null});
}
return { body: text.slice(0, m.index).trimEnd(), citations: cites};
}

/* ---- 问答 ---- */
function addUserMsg(text) {
$("empty-hint")?.remove();
const div = document.createElement("div");
div.className = "msg msg-user";
const b = document.createElement("div");
b.className = "bubble-user";
b.textContent = text;
div.appendChild(b);
thread.appendChild(div);
scrollToBottom();
syncEmptyState();
}
function renderSources(cites, anchor) {
const box = document.createElement("div");
box.className = "sources";
const h = document.createElement("h3");
h.textContent = t("sourceTitle");
box.appendChild(h);
cites.forEach((c, i) => {
const card = document.createElement("div");
card.className = "source-card";
card.id = `src-${anchor}-${i + 1}`;
const head = document.createElement("div");
head.className = "source-head";
const name = document.createElement("span");
name.className = "source-name";
name.textContent = c.source;
head.appendChild(name);
if (c.page) {
const loc = document.createElement("span");
loc.className = "source-loc";
loc.textContent = t("pageLabel", { n: c.page});
head.appendChild(loc);
}
card.appendChild(head);
box.appendChild(card);
});
return box;
}
function renderRetrievalDebug(entries) {
if (!entries ||!entries.length) return null;
const details = document.createElement("details");
details.className = "retrieval";
const keptTotal = entries.reduce((n, e) => n + (e.chunks || []).filter((c) => c.kept).length, 0);
const candTotal = entries.reduce((n, e) => n + (e.candidates || (e.chunks || []).length), 0);
const summary = document.createElement("summary");
summary.textContent = `${t("retrievalTitle")} · ${t("retrievalSummary", { candidates: candTotal, returned: keptTotal})}`;
details.appendChild(summary);
entries.forEach((entry) => {
const meta = document.createElement("p");
meta.className = "retrieval-meta";
meta.textContent = t("retrievalMeta", { top: entry.top_score?? "–", threshold: entry.threshold?? "–"});
details.appendChild(meta);
const ul = document.createElement("ul");
ul.className = "retrieval-chunks";
(entry.chunks || []).filter((c) => c.kept).forEach((c) => {
const li = document.createElement("li");
const head = document.createElement("div");
head.className = "r-head";
const rank = document.createElement("span");
rank.textContent = `#${c.rank} · ${c.score}`;
const src = document.createElement("span");
src.className = "mono";
src.textContent = c.source + (c.page? ` · ${t("pageLabel", { n: c.page})}`: "");
head.append(rank, src);
li.appendChild(head);
if (c.preview) {
const pv = document.createElement("p");
pv.className = "r-preview";
pv.textContent = c.preview;
li.appendChild(pv);
}
ul.appendChild(li);
});
details.appendChild(ul);
const dropped = (entry.chunks || []).filter((c) =>!c.kept).length;
if (dropped > 0) {
const dp = document.createElement("p");
dp.className = "retrieval-dropped";
dp.textContent = t("retrievalDropped", { n: dropped});
details.appendChild(dp);
}
});
return details;
}
function attachCites(body, cites, wrap, anchor) {
if (!cites.length) return;
cites.forEach((c, i) => {
const b = document.createElement("button");
b.type = "button";
b.className = "cite";
b.textContent = String(i + 1);
b.setAttribute("aria-label", `${c.source}${c.page? ` · ${t("pageLabel", { n: c.page})}`: ""}`);
b.addEventListener("click", () => {
const card = document.getElementById(`src-${anchor}-${i + 1}`);
if (!card) return;
card.scrollIntoView({ behavior: "smooth", block: "center"});
card.classList.add("flash");
setTimeout(() => card.classList.remove("flash"), 1600);
});
const ps = body.querySelectorAll("p");
const last = ps[ps.length - 1] || body;
last.appendChild(document.createTextNode(" "));
last.appendChild(b);
});
wrap.appendChild(renderSources(cites, anchor));
}
function renderAssistant(text, doneData) {
const wrap = document.createElement("div");
wrap.className = "msg";
const body = document.createElement("div");
body.className = "bubble-ai";
wrap.appendChild(body);
const { body: cleanBody, citations: lineCites} = splitSourceLine(text || "");
const cites = (doneData && doneData.citations && doneData.citations.length)? doneData.citations: lineCites;
body.innerHTML = cleanBody? renderMarkdown(cleanBody): `<p>${esc(t("noReply"))}</p>`;
const anchor = `m${Date.now()}${Math.floor(Math.random() * 1000)}`;
attachCites(body, cites, wrap, anchor);
if (doneData && doneData.retrieval_debug) {
const dbg = renderRetrievalDebug(doneData.retrieval_debug);
if (dbg) wrap.appendChild(dbg);
}
if (doneData) {
const status = document.createElement("p");
status.className = "answer-status";
if (doneData.grounding === "refused") status.textContent = t("statusRefused");
else if (doneData.grounding === "hedged") status.textContent = t("statusHedged");
else if (doneData.unsupported_claims && doneData.unsupported_claims.length) status.textContent = t("unsupportedNote");
else status.textContent = t("statusDone");
body.appendChild(status);
}
thread.appendChild(wrap);
return wrap;
}
const ICON_SEND = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>`;
const ICON_STOP = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5"/></svg>`;
/* 圆钮是图标按钮，没有可见文字，所以 i18n 落在 aria-label / title 上。
   空闲态下输入框为空就置灰——没有内容可发，不该让按钮看起来可点。 */
function setSendMode(mode) {
if (mode === "streaming") {
sendBtn.innerHTML = ICON_STOP;
sendBtn.disabled = false;
sendBtn.setAttribute("aria-label", t("stop"));
sendBtn.title = t("stop");
} else {
sendBtn.innerHTML = ICON_SEND;
sendBtn.disabled = !input.value.trim();
sendBtn.setAttribute("aria-label", t("send"));
sendBtn.title = t("send");
}
}
async function ask(question) {
if (streaming) return;
const q = (question?? input.value).trim();
if (!q) return;
input.value = "";
streaming = true;
setSendMode("streaming");
addUserMsg(q);

const wrap = document.createElement("div");
wrap.className = "msg";
const body = document.createElement("div");
body.className = "bubble-ai";
const p = document.createElement("p");
body.appendChild(p);
wrap.appendChild(body);
thread.appendChild(wrap);
scrollToBottom();

aborter = new AbortController();
let fullText = "";
let doneData = null;
let failed = null;
try {
const res = await fetch("/chat/stream", {
method: "POST",
headers: authHeaders({ "Content-Type": "application/json"}),
body: JSON.stringify({ message: q, session_id: getSessionId(), replace_last: false, debug: true}),
signal: aborter.signal,
});
if (!res.ok) {
const data = await res.json().catch(() => ({}));
if (res.status === 401) {
  showUnlockOverlay("Session expired or password required.");
}
throw new Error(errorFrom(res.status, data));
}
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
const handleBlock = (block) => {
for (const line of block.split("\n")) {
if (!line.startsWith("data: ")) continue;
let data;
try { data = JSON.parse(line.slice(6));} catch { continue;}
if (data.done) { doneData = data; return true;}
if (data.error) { failed = String(data.error); return true;}
if (data.token) {
fullText += data.token;
p.textContent = fullText;
scrollToBottom();
}
}
return false;
};
while (true) {
const { done, value} = await reader.read();
if (done) break;
buffer += decoder.decode(value, { stream: true});
const lines = buffer.split("\n");
buffer = lines.pop() || "";
if (handleBlock(lines.join("\n"))) break;
}
if (!doneData &&!failed && buffer.trim()) handleBlock(buffer);
if (!doneData &&!failed) failed = t("errConnLost");
} catch (err) {
if (err && err.name === "AbortError") failed = t("statusStopped");
else failed = (err && err.message) || t("errGeneric");
} finally {
streaming = false;
aborter = null;
setSendMode("idle");
}

if (failed) {
p.textContent = fullText || t("statusFailed");
const st = document.createElement("p");
st.className = "answer-status";
st.textContent = failed;
body.appendChild(st);
toast(failed);
input.focus();
return;
}

const text = (doneData.response?? fullText) || "";
if (doneData.session_id) setSessionId(doneData.session_id);
wrap.remove();
renderAssistant(text, doneData);
scrollToBottom();
input.focus();
loadSessions();
}

/* ---- 历史对话 ---- */
const PINNED_KEY = "ragmate_pinned_sessions";
function getPinned() {
  try { return JSON.parse(localStorage.getItem(PINNED_KEY) || "[]"); } catch { return []; }
}
function setPinned(ids) {
  try { localStorage.setItem(PINNED_KEY, JSON.stringify(ids)); } catch {}
}
function renderHistoryItem(s, list, pinned) {
const li = document.createElement("li");
const current = peekSessionId();
if (s.session_id === current) li.classList.add("is-active");
const title = document.createElement("span");
title.className = "h-title";
const isPinned = pinned.includes(s.session_id);
title.textContent = s.first_message || s.session_id.slice(0, 8);
title.title = s.first_message || s.session_id;
const moreBtn = document.createElement("button");
moreBtn.type = "button";
moreBtn.className = "h-more";
moreBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/></svg>`;
moreBtn.setAttribute("aria-label", "More options");
moreBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  document.querySelectorAll(".h-menu.show").forEach((m) => m.classList.remove("show"));
  const menu = li.querySelector(".h-menu");
  const rect = moreBtn.getBoundingClientRect();
  menu.style.right = (window.innerWidth - rect.right) + "px";
  menu.style.top = (rect.bottom + 4) + "px";
  menu.classList.add("show");
  const onKey = (ev) => {
    if (ev.key === "Escape") { menu.classList.remove("show"); document.removeEventListener("keydown", onKey); return; }
    if (ev.key.toLowerCase() === "p") { menu.querySelector(".h-menu-item").click(); document.removeEventListener("keydown", onKey); }
    if (ev.key.toLowerCase() === "r") { menu.querySelectorAll(".h-menu-item")[1].click(); document.removeEventListener("keydown", onKey); }
    if (ev.key.toLowerCase() === "d") { menu.querySelectorAll(".h-menu-item")[2].click(); document.removeEventListener("keydown", onKey); }
  };
  document.addEventListener("keydown", onKey);
});
const menu = document.createElement("div");
menu.className = "h-menu";
const makeItem = (label, shortcut, cls, handler) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "h-menu-item" + (cls ? " " + cls : "");
  btn.innerHTML = `<span>${label}</span><span class="h-menu-shortcut">${shortcut}</span>`;
  btn.addEventListener("click", (e) => { e.stopPropagation(); menu.classList.remove("show"); handler(e); });
  return btn;
};
menu.appendChild(makeItem(isPinned ? "Unpin" : "Pin", "P", "", () => {
  const pinned = getPinned();
  if (pinned.includes(s.session_id)) {
    setPinned(pinned.filter(id => id !== s.session_id));
  } else {
    setPinned([...pinned, s.session_id]);
  }
  renderHistory();
}));
menu.appendChild(makeItem("Rename", "R", "", () => {}));
menu.appendChild(makeItem("Delete", "D", "danger", async () => {
  if (!await confirm(t("deleteSessionConfirm"))) return;
  try {
    await request("DELETE", `/chat/sessions/${encodeURIComponent(s.session_id)}`);
    if (peekSessionId() === s.session_id) startNewChat();
    await loadSessions();
  } catch (err) { toast(err.message);}
}));
li.append(title, moreBtn, menu);
li.addEventListener("click", () => loadSession(s.session_id));
list.appendChild(li);
}
function renderHistory() {
const list = $("history-list");
const pinnedList = $("pinned-list");
list.replaceChildren();
pinnedList.replaceChildren();
$("history-empty").hidden = lastSessions.length > 0;
const pinned = getPinned();
const pinnedSessions = lastSessions.filter(s => pinned.includes(s.session_id));
const unpinnedSessions = lastSessions.filter(s => !pinned.includes(s.session_id));
$("pinned-block").hidden = pinnedSessions.length === 0;
pinnedSessions.forEach((s) => renderHistoryItem(s, pinnedList, pinned));
unpinnedSessions.forEach((s) => renderHistoryItem(s, list, pinned));
syncNavActive();
}
async function loadSessions() {
try {
const data = await request("GET", "/chat/sessions");
lastSessions = data.sessions || [];
} catch {
lastSessions = [];
}
renderHistory();
}
async function loadSession(sessionId) {
try {
const data = await request("GET", `/chat/sessions/${encodeURIComponent(sessionId)}`);
setSessionId(sessionId);
thread.replaceChildren();
(data.messages || []).forEach((m) => {
if (m.role === "user") addUserMsg(m.content || "");
else renderAssistant(m.content || "", null);
});
if (!data.messages ||!data.messages.length) showEmptyHint();
syncEmptyState();
setView("chat");
renderHistory();
closeSidebar();
scrollToBottom();
} catch (err) { toast(err.message);}
}
function showEmptyHint() {
thread.replaceChildren();
const p = document.createElement("p");
p.className = "hint";
p.id = "empty-hint";
p.textContent = t("emptyHint");
thread.appendChild(p);
syncEmptyState();
}
/* 空态判定：线程里还挂着 #empty-hint 就说明一句都没问过。
   空态时给 #view-chat 加 is-empty，hero 与输入框收成垂直居中的一整块；
   一旦有消息，is-empty 摘掉，输入框回到常规的底部位置。 */
function syncEmptyState() {
$("view-chat").classList.toggle("is-empty", !!$("empty-hint"));
}
function startNewChat() {
try { sessionStorage.removeItem(SESSION_KEY);} catch { /* ignore */}
setView("chat");
thread.replaceChildren();
showEmptyHint();
renderHistory();
closeSidebar();
input.focus();
}

/* ---- 文档一览与上传 ---- */
const DOC_STATUS = { ingested: "docIngested", uploaded: "docUploaded", ingesting: "docIngesting", failed: "docFailed"};
function renderDocs() {
const totalChunks = lastDocs.reduce((n, d) => n + (d.chunk_count || 0), 0);
$("docs-count").textContent = lastDocs.length? ` · ${lastDocs.length} · ${t("chunksLabel", { n: totalChunks})}`: "";
const list = $("doc-list");
list.replaceChildren();
$("doc-empty").hidden = lastDocs.length > 0;
lastDocs.forEach((d) => {
const li = document.createElement("li");
const name = document.createElement("span");
name.className = "mono";
name.textContent = d.filename;
const meta = document.createElement("span");
meta.className = "count";
meta.textContent = `${t(DOC_STATUS[d.status] || "docUploaded")} · ${t("chunksLabel", { n: d.chunk_count || 0})}`;
const del = document.createElement("button");
del.type = "button";
del.className = "doc-del";
del.textContent = "×";
del.setAttribute("aria-label", t("deleteDocConfirm", { name: d.filename}));
del.addEventListener("click", async () => {
if (!await confirm(t("deleteDocConfirm", { name: d.filename}))) return;
try {
await request("DELETE", `/documents/${encodeURIComponent(d.filename)}`);
await loadDocuments();
} catch (err) { toast(err.message);}
});
li.append(name, meta, del);
list.appendChild(li);
});
}
async function loadDocuments() {
try {
const data = await request("GET", "/documents");
lastDocs = data.documents || [];
} catch (err) {
toast(err.message || t("errGeneric"));
lastDocs = [];
}
renderDocs();
updateDocsMeta();
return lastDocs;
}
function uploadDocument(file) {
return new Promise((resolve, reject) => {
const xhr = new XMLHttpRequest();
xhr.open("POST", "/documents/upload");
const token = getAuthToken();
if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
xhr.onload = () => {
let data = {};
try { data = JSON.parse(xhr.responseText || "{}");} catch { /* ignore */}
if (xhr.status >= 200 && xhr.status < 300) resolve(data);
else reject(new Error(errorFrom(xhr.status, data)));
};
xhr.onerror = () => reject(new Error(t("errNetwork")));
xhr.onabort = () => reject(new Error(t("errUploadCancelled")));
const body = new FormData();
body.append("file", file);
xhr.send(body);
});
}
async function pollIngest(dropzone) {
for (let i = 0; i < 150; i++) {
await new Promise((r) => setTimeout(r, 2000));
try {
const st = await request("GET", "/ingest/status");
if (st.status === "idle") {
dropzone.textContent = t("dropzone");
dropzone.classList.remove("busy");
await loadDocuments();
toast(t("ingestDone"));
return;
}
dropzone.textContent = `${t("ingesting")}${st.message? ` · ${st.message}`: ""}`;
} catch { break;}
}
dropzone.textContent = t("dropzone");
dropzone.classList.remove("busy");
await loadDocuments();
}
async function handleFiles(files) {
const dropzone = $("dropzone");
const docs = await loadDocuments();
for (const file of files) {
if (docs.some((d) => d.filename === file.name)) {
if (!await confirm(t("replaceConfirm", { name: file.name }))) continue;
// 覆盖上传：先删后传（与旧工作台一致）
try { await request("DELETE", `/documents/${encodeURIComponent(file.name)}`); } catch (err) { toast(err.message); continue; }
}
dropzone.classList.add("busy");
dropzone.textContent = `${t("uploading")} ${file.name}…`;
try {
await uploadDocument(file);
dropzone.textContent = t("ingesting");
await request("POST", "/ingest", { filenames: [file.name]});
await pollIngest(dropzone);
} catch (err) {
dropzone.textContent = t("dropzone");
dropzone.classList.remove("busy");
toast(err.message || t("errGeneric"));
}
await loadDocuments();
}
}



/* ---- 统一认证与解锁遮罩 ---- */
function showUnlockError(message) {
  const errBox = $("unlock-error");
  const errText = $("unlock-error-text");
  const pwdInput = $("unlock-password");
  const card = $("unlock-card");

  if (errText) errText.textContent = message;
  if (errBox) errBox.hidden = false;
  if (pwdInput) {
    pwdInput.classList.add("error");
    pwdInput.focus();
  }
  if (card) {
    card.classList.remove("shake");
    void card.offsetWidth;
    card.classList.add("shake");
  }
}

function clearUnlockError() {
  const errBox = $("unlock-error");
  const pwdInput = $("unlock-password");
  if (errBox) errBox.hidden = true;
  if (pwdInput) pwdInput.classList.remove("error");
}

function showUnlockOverlay(msg) {
  const overlay = $("unlock-overlay");
  const input = $("unlock-password");
  clearUnlockError();
  if (overlay) overlay.hidden = false;
  if (msg) {
    showUnlockError(msg);
  }
  if (input) {
    input.value = "";
    setTimeout(() => input.focus(), 60);
  }
}

function hideUnlockOverlay() {
  const overlay = $("unlock-overlay");
  clearUnlockError();
  if (overlay) overlay.hidden = true;
}

async function checkAuthStatus() {
  try {
    const res = await fetch("/auth/status", { headers: authHeaders() });
    const data = await res.json().catch(() => ({}));
    const logoutBtn = $("btn-logout");
    if (data.remote_disabled) {
      showUnlockOverlay("Remote access is disabled because API_AUTH_TOKEN is not configured on the server.");
      return false;
    }
    if (data.auth_required) {
      if (logoutBtn) logoutBtn.hidden = false;
      if (!data.authenticated) {
        showUnlockOverlay();
        return false;
      }
    } else {
      if (logoutBtn) logoutBtn.hidden = true;
    }
    hideUnlockOverlay();
    return true;
  } catch {
    return true;
  }
}

async function updateDocsMeta() {
  try {
    const res = await fetch("/settings", { headers: authHeaders() });
    if (!res.ok) return;
    const data = await res.json();
    if (data.supported_extensions) {
      const el = $("docs-meta-types");
      if (el) el.textContent = data.supported_extensions.join(", ");
    }
    if (data.settings && data.settings.MAX_UPLOAD_SIZE) {
      const el = $("docs-meta-limit");
      const mb = Math.round(Number(data.settings.MAX_UPLOAD_SIZE) / (1024 * 1024));
      if (el) el.textContent = `${mb} MB`;
    }
  } catch {}
}

/* ---- 设置管理 (Settings Management) ---- */
const SECRET_MASK = "••••••••••••••••";
let currentSettingsData = null;
let currentSettingsVersion = null;
let configuredSecrets = new Set();
let currentFieldSources = {};
let currentIndexLockedKeys = [];
let currentEditableKeys = [];
let connectionTestResults = { llm: null, embedding: null, reranker: null };

const FIELD_DEFS = {
  // Chat Model
  "LLM_MODEL": { type: "text" },
  "LLM_API_BASE_URL": { type: "text" },
  "LLM_API_KEY": { type: "secret" },
  "LLM_TEMPERATURE": { type: "number", min: 0.0, max: 2.0 },
  "LLM_REASONING_EFFORT": { type: "select" },
  // Embedding Model
  "EMBEDDING_PROVIDER": { type: "select" },
  "EMBEDDING_MODEL": { type: "text" },
  "EMBEDDING_API_BASE_URL": { type: "text" },
  "EMBEDDING_API_KEY": { type: "secret" },
  // Reranker Model
  "RERANKER_PROVIDER": { type: "select" },
  "RERANKER_MODEL": { type: "text" },
  "RERANKER_API_BASE_URL": { type: "text" },
  "RERANKER_API_KEY": { type: "secret" },
  // LangSmith Observability
  "LANGSMITH_TRACING": { type: "boolean" },
  "LANGSMITH_API_KEY": { type: "secret" },
  "LANGSMITH_PROJECT": { type: "text" },
  "LANGSMITH_ENDPOINT": { type: "text" },
};

function updateSettingsLinkage() {
  const embProvider = $("cfg-EMBEDDING_PROVIDER")?.value || "remote";
  const isEmbRemote = embProvider === "remote" || embProvider === "openai";
  const wrapEmbUrl = $("wrap-EMBEDDING_API_BASE_URL");
  const wrapEmbKey = $("wrap-EMBEDDING_API_KEY");
  if (wrapEmbUrl) wrapEmbUrl.style.display = isEmbRemote ? "flex" : "none";
  if (wrapEmbKey) wrapEmbKey.style.display = isEmbRemote ? "flex" : "none";

  const rerankProvider = $("cfg-RERANKER_PROVIDER")?.value || "api";
  const isRerankApi = rerankProvider === "api";
  const wrapRerankUrl = $("wrap-RERANKER_API_BASE_URL");
  const wrapRerankKey = $("wrap-RERANKER_API_KEY");
  if (wrapRerankUrl) wrapRerankUrl.style.display = isRerankApi ? "flex" : "none";
  if (wrapRerankKey) wrapRerankKey.style.display = isRerankApi ? "flex" : "none";

  const tracingOn = $("cfg-LANGSMITH_TRACING")?.checked;
  const wrapTracing = $("wrap-LANGSMITH_CONFIG");
  if (wrapTracing) wrapTracing.style.opacity = tracingOn ? "1" : "0.4";
}

async function loadSettings() {
  try {
    const res = await fetch("/settings", { headers: authHeaders() });
    if (res.status === 401) {
      showUnlockOverlay("Session expired. Please enter instance password.");
      return;
    }
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      toast(errData.detail || "Failed to load settings");
      return;
    }
    const data = await res.json();
    currentSettingsData = data.settings || {};
    currentSettingsVersion = data.version || null;
    configuredSecrets = new Set(data.secret_keys || []);
    currentFieldSources = data.field_sources || {};
    currentIndexLockedKeys = data.index_status?.locked_keys || [];
    currentEditableKeys = data.editable_keys || [];

    // Populate dynamic specs
    const specChunking = $("spec-chunking");
    if (specChunking) {
      const strat = currentSettingsData.CHUNKING_STRATEGY || "title_path";
      const parent = currentSettingsData.CHUNK_SIZE_PARENT || 2500;
      const child = currentSettingsData.CHUNK_SIZE || 1000;
      specChunking.innerHTML = `<code>${strat}</code> with Small-to-Big parent expansion (parent ${parent} chars, child ${child} chars).`;
    }
    const specHybrid = $("spec-hybrid");
    if (specHybrid) {
      const k = currentSettingsData.RRF_K || 60;
      specHybrid.textContent = `BM25 text rank fused with 1024-dim HNSW vector search via Reciprocal Rank Fusion (k=${k}).`;
    }
    const specRerank = $("spec-reranker");
    if (specRerank) {
      const model = currentSettingsData.RERANKER_MODEL || "BAAI/bge-reranker-v2-m3";
      const cand = currentSettingsData.RERANK_CANDIDATES || 30;
      const thresh = currentSettingsData.RERANK_SCORE_THRESHOLD || 0.3;
      const finalK = currentSettingsData.FINAL_CONTEXT_K || 15;
      specRerank.textContent = `Top ${cand} candidates re-scored via ${model}; scores < ${thresh} filtered out; top ${finalK} retained.`;
    }
    const specGrounding = $("spec-grounding");
    if (specGrounding) {
      const faith = currentSettingsData.FAITHFULNESS_CHECK;
      specGrounding.textContent = (faith ? "Faithfulness check enabled. " : "") + "Automatically hedges or refuses when retrieved passage evidence is insufficient.";
    }

    // Populate fields
    for (const [key, def] of Object.entries(FIELD_DEFS)) {
      const el = $(`cfg-${key}`);
      if (!el) continue;
      el.disabled = false; // default enabled for all editable keys
      const val = currentSettingsData[key];

      if (def.type === "boolean") {
        el.checked = (val === true || val === "true" || val === 1 || val === "1");
      } else if (def.type === "secret") {
        const isConf = Boolean(currentSettingsData[key + "_configured"]);
        const tag = $(`tag-${key}`);
        if (isConf) {
          el.value = SECRET_MASK;
          el.dataset.configured = "true";
          el.placeholder = "Configured (enter new value to change)";
          if (tag) {
            tag.textContent = "Configured";
            tag.classList.add("configured");
          }
        } else {
          el.value = "";
          el.dataset.configured = "false";
          el.placeholder = "Not set";
          if (tag) {
            tag.textContent = "Not set";
            tag.classList.remove("configured");
          }
        }
      } else {
        el.value = val !== undefined && val !== null ? String(val) : "";
      }
    }

    // 1. Mark deployment environment-managed fields
    const sources = data.field_sources || {};
    for (const [key, src] of Object.entries(sources)) {
      if (src === "environment") {
        const el = $(`cfg-${key}`);
        if (el) el.disabled = true;
        const tag = $(`tag-${key}`);
        if (tag) {
          tag.textContent = "Deployment Managed";
          tag.classList.add("configured");
        }
      }
    }

    // 2. Lock embedding fields if documents are already indexed in vector store
    const indexStatus = data.index_status || {};
    if (indexStatus.has_indexed_docs) {
      const lockedKeys = indexStatus.locked_keys || ["EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_API_BASE_URL"];
      lockedKeys.forEach((key) => {
        const el = $(`cfg-${key}`);
        if (el) el.disabled = true;
        const tag = $(`tag-${key}`);
        if (tag) {
          tag.textContent = "Locked (docs indexed)";
          tag.classList.add("configured");
        }
      });
    }

    // 3. Display pending restart banner if there are unapplied changes on disk
    const pending = data.pending_restart || [];
    const restartBanner = $("settings-restart-banner");
    const restartText = $("settings-restart-text");
    if (restartBanner) {
      if (pending.length > 0) {
        restartBanner.hidden = false;
        if (restartText) {
          restartText.innerHTML = `Pending restart for: <strong>${pending.join(", ")}</strong>. Service restart is required to take effect.`;
        }
      } else {
        restartBanner.hidden = true;
      }
    }

    updateSettingsLinkage();
  } catch (err) {
    console.error("Failed to load settings:", err);
  }
}

async function saveAllSettings() {
  const saveBtn = $("btn-save-settings");
  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.textContent = "Saving...";
  }

  const updates = {};
  let hasError = false;

  // Clear previous errors
  document.querySelectorAll(".field-error").forEach((el) => el.classList.remove("field-error"));

  for (const [key, def] of Object.entries(FIELD_DEFS)) {
    // Only send editable keys; skip deployment managed or locked fields
    if (currentEditableKeys.length > 0 && !currentEditableKeys.includes(key)) continue;

    const el = $(`cfg-${key}`);
    if (!el || el.disabled) continue;
    if (currentFieldSources[key] === "environment") continue;
    if (currentIndexLockedKeys.includes(key)) continue;

    let changed = false;
    let val = null;

    if (def.type === "boolean") {
      val = el.checked;
      const oldVal = Boolean(currentSettingsData[key]);
      if (val !== oldVal) changed = true;
    } else if (def.type === "secret") {
      const sVal = el.value.trim();
      if (sVal && sVal !== SECRET_MASK) {
        val = sVal;
        changed = true;
      }
    } else if (def.type === "number") {
      const raw = el.value.trim();
      if (raw !== "") {
        const num = Number(raw);
        if (isNaN(num) || (def.min !== undefined && num < def.min) || (def.max !== undefined && num > def.max)) {
          el.classList.add("field-error");
          toast(`${key} is out of valid range (${def.min ?? "-"} ~ ${def.max ?? "-"})`);
          hasError = true;
          break;
        }
        val = num;
        const oldNum = Number(currentSettingsData[key]);
        if (val !== oldNum) changed = true;
      }
    } else {
      val = el.value.trim();
      const oldStr = (currentSettingsData[key] ?? "").toString().trim();
      if (val !== oldStr) changed = true;
    }

    if (changed) {
      updates[key] = val;
    }
  }

  if (hasError) {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = "Save Settings";
    }
    return;
  }

  if (Object.keys(updates).length === 0) {
    toast("No changes to save.");
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = "Save Settings";
    }
    return;
  }

  try {
    const res = await fetch("/settings", {
      method: "PUT",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ settings: updates, version: currentSettingsVersion }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "保存失败");
    }

    const result = await res.json();
    const pendingAfter = result.pending_restart || [];
    const banner = $("settings-restart-banner");
    const restartText = $("settings-restart-text");
    if (result.restart_required || pendingAfter.length > 0) {
      if (banner) {
        banner.hidden = false;
        if (restartText && pendingAfter.length > 0) {
          restartText.innerHTML = `Settings saved! Pending restart for: <strong>${pendingAfter.join(", ")}</strong>. Core changes require service restart.`;
        }
      }
      toast("Settings saved! A restart is required for changes to take effect.");
    } else {
      if (banner) banner.hidden = true;
      toast("Settings saved successfully!");
    }

    // Sync token to browser localStorage if API_AUTH_TOKEN was updated
    if (updates["API_AUTH_TOKEN"]) {
      try { localStorage.setItem(TOKEN_KEY, updates["API_AUTH_TOKEN"]); } catch {}
    }

    await loadSettings();
    await loadReady();
  } catch (err) {
    toast(err.message || "Failed to save settings");
  } finally {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = "Save Settings";
    }
  }
}

async function discardUnsavedChanges() {
  await loadSettings();
  toast("Unsaved changes discarded.");
}


async function runConnectionTest(component, btnId, resultId) {
  const btn = $(btnId);
  const resultEl = $(resultId);
  if (!btn || !resultEl) return;
  btn.disabled = true;
  const oldText = btn.textContent;
  btn.textContent = "Testing...";
  resultEl.className = "test-conn-result";
  resultEl.textContent = "Connecting to service...";

  // Collect form input values to test current inputs even before saving
  const getSecretVal = (id) => {
    const v = $(id)?.value?.trim();
    return (v && v !== SECRET_MASK) ? v : null;
  };

  let testPayload = {};
  if (component === "llm") {
    testPayload = {
      model: $("cfg-LLM_MODEL")?.value?.trim() || null,
      base_url: $("cfg-LLM_API_BASE_URL")?.value?.trim() || null,
      api_key: getSecretVal("cfg-LLM_API_KEY"),
    };
  } else if (component === "embedding") {
    testPayload = {
      provider: $("cfg-EMBEDDING_PROVIDER")?.value || null,
      model: $("cfg-EMBEDDING_MODEL")?.value?.trim() || null,
      base_url: $("cfg-EMBEDDING_API_BASE_URL")?.value?.trim() || null,
      api_key: getSecretVal("cfg-EMBEDDING_API_KEY"),
    };
  } else if (component === "reranker") {
    testPayload = {
      provider: $("cfg-RERANKER_PROVIDER")?.value || null,
      model: $("cfg-RERANKER_MODEL")?.value?.trim() || null,
      base_url: $("cfg-RERANKER_API_BASE_URL")?.value?.trim() || null,
      api_key: getSecretVal("cfg-RERANKER_API_KEY"),
    };
  }

  try {
    const res = await fetch(`/settings/test/${component}`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(testPayload),
    });
    if (res.status === 403) {
      resultEl.className = "test-conn-result bad";
      resultEl.textContent = "✖ Management API disabled on server (API_AUTH_TOKEN not set).";
      return;
    }
    if (res.status === 401) {
      resultEl.className = "test-conn-result bad";
      resultEl.textContent = "✖ Unauthorized: API token required. Please enter token in Access & Limits.";
      return;
    }
    const data = await res.json();
    if (data.ok) {
      connectionTestResults[component] = "ok";
      resultEl.className = "test-conn-result ok";
      resultEl.textContent = `✔ ${data.detail || "Success"}`;
    } else {
      connectionTestResults[component] = "bad";
      resultEl.className = "test-conn-result bad";
      resultEl.textContent = `✖ ${data.detail || "Connection failed"}`;
    }
    loadReady();
  } catch (err) {
    resultEl.className = "test-conn-result bad";
    resultEl.textContent = `✖ ${err.message || "Network error"}`;
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
}

function initSettingsEvents() {
  // Secret input mask interaction
  ["LLM_API_KEY", "EMBEDDING_API_KEY", "RERANKER_API_KEY", "LANGSMITH_API_KEY"].forEach((k) => {
    const el = $(`cfg-${k}`);
    if (!el) return;
    el.addEventListener("focus", () => {
      if (el.value === SECRET_MASK) {
        el.value = "";
      }
    });
    el.addEventListener("blur", () => {
      if (!el.value.trim() && el.dataset.configured === "true") {
        el.value = SECRET_MASK;
      }
    });
  });

  $("cfg-EMBEDDING_PROVIDER")?.addEventListener("change", updateSettingsLinkage);
  $("cfg-RERANKER_PROVIDER")?.addEventListener("change", updateSettingsLinkage);
  $("cfg-LANGSMITH_TRACING")?.addEventListener("change", updateSettingsLinkage);
  $("btn-save-settings")?.addEventListener("click", saveAllSettings);
  $("btn-discard-settings")?.addEventListener("click", discardUnsavedChanges);
  $("btn-refresh-ready")?.addEventListener("click", loadReady);
  $("settings-restart-close")?.addEventListener("click", () => {
    const banner = $("settings-restart-banner");
    if (banner) banner.hidden = true;
  });

  // Test connection event listeners
  $("btn-test-llm")?.addEventListener("click", () => runConnectionTest("llm", "btn-test-llm", "res-test-llm"));
  $("btn-test-embedding")?.addEventListener("click", () => runConnectionTest("embedding", "btn-test-embedding", "res-test-embedding"));
  $("btn-test-reranker")?.addEventListener("click", () => runConnectionTest("reranker", "btn-test-reranker", "res-test-reranker"));

  // Unlock form submission & input cleanup
  $("unlock-password")?.addEventListener("input", clearUnlockError);

  $("unlock-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const pwd = $("unlock-password")?.value || "";
    const btn = $("btn-unlock");

    clearUnlockError();

    // 1. Client-side empty validation (no browser native tooltip bubble)
    if (!pwd.trim()) {
      showUnlockError(t("pwdRequired"));
      return;
    }

    if (btn) btn.disabled = true;
    try {
      const res = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: pwd }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        const detail = data.detail || "";
        const isWrong = detail.toLowerCase().includes("incorrect");
        showUnlockError(isWrong ? t("pwdIncorrect") : (detail || t("pwdIncorrect")));
        return;
      }
      if (data.token) {
        setAuthToken(data.token);
      }
      hideUnlockOverlay();
      const logoutBtn = $("btn-logout");
      if (logoutBtn) logoutBtn.hidden = false;
      toast(t("pwdUnlocked"));
      initApp();
    } catch (ex) {
      showUnlockError(ex.message || t("errNetwork"));
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  // Logout button
  $("btn-logout")?.addEventListener("click", async () => {
    try {
      await fetch("/auth/logout", { method: "POST", headers: authHeaders() });
    } catch {}
    setAuthToken("");
    toast("Session locked");
    showUnlockOverlay("Session locked. Enter instance password to unlock.");
  });
}

/* ---- 设置：令牌与服务状态 ---- */
/* 后端 llm_detail 是原因码，映射到 i18n 键；未知码不显示提示，免得露出原始字符串。 */
const LLM_HINTS = {
not_configured: "llmNotConfigured",
unreachable: "llmUnreachable",
http_error: "llmHttpError",
model_missing: "llmModelMissing",
};
function addReadyRow(list, label, value, cls, hint, dataDest) {
  const li = document.createElement("li");
  const left = document.createElement("div");
  left.className = "ready-row-left";

  const n = document.createElement("span");
  n.textContent = label;
  n.style.fontWeight = "500";
  left.appendChild(n);

  if (dataDest) {
    const dest = document.createElement("span");
    dest.className = "data-dest-hint";
    dest.textContent = dataDest;
    left.appendChild(dest);
  }

  const v = document.createElement("span");
  v.className = cls;
  v.textContent = value;
  if (hint) { li.title = hint; v.title = hint; }

  li.append(left, v);
  list.appendChild(li);
}

async function loadReady() {
  const list = $("ready-list");
  try {
    const res = await fetch("/ready", { headers: authHeaders() });
    const data = await res.json().catch(() => ({}));
    lastReady = data;
    const chip = $("model-chip");
    if (data.model) {
      chip.textContent = data.model;
      chip.title = data.model;
      chip.hidden = false;
    }
    const checks = data.checks || {};
    list.replaceChildren();

    // Overall status header
    const title = document.createElement("li");
    const tn = document.createElement("span");
    tn.textContent = "Service Status";
    const tv = document.createElement("span");
    tv.className = res.ok && data.status === "ready" ? "ok" : "bad";
    tv.textContent = res.ok && data.status === "ready" ? "Ready" : "Unavailable";
    title.append(tn, tv);
    list.appendChild(title);

    // 1. PostgreSQL
    addReadyRow(list, "PostgreSQL Database", checks.postgresql ? "Ready" : "Unavailable", checks.postgresql ? "ok" : "bad", null, "Stores relational schema, document catalog, and chat sessions");

    // 2. pgvector
    addReadyRow(list, "pgvector Extension", checks.pgvector ? "Ready" : "Unavailable", checks.pgvector ? "ok" : "bad", null, "Vector index with 1024-dim HNSW cosine/inner-product support");

    // 3. Chat Model
    const llmOk = checks.llm;
    const modelName = data.model || $("cfg-LLM_MODEL")?.value || "Unconfigured";
    const llmHint = data.llm_detail ? `Status: ${data.llm_detail}` : null;
    addReadyRow(list, `Chat Model (${modelName})`, llmOk ? "Ready" : "Unavailable", llmOk ? "ok" : "bad", llmHint, "Receives user questions and retrieved context passages");

    // 4. Embedding Model
    const embModel = $("cfg-EMBEDDING_MODEL")?.value || "BAAI/bge-m3";
    const embProv = $("cfg-EMBEDDING_PROVIDER")?.value || "remote";
    let embStatus = "Untested";
    let embClass = "untested";
    if (connectionTestResults.embedding === "ok") {
      embStatus = "Ready";
      embClass = "ok";
    } else if (connectionTestResults.embedding === "bad") {
      embStatus = "Failed";
      embClass = "bad";
    }
    addReadyRow(list, `Embedding Model (${embModel})`, embStatus, embClass, null, `Receives document text during indexing and search queries (${embProv})`);

    // 5. Reranker Model
    const rerankModel = $("cfg-RERANKER_MODEL")?.value || "BAAI/bge-reranker-v2-m3";
    const rerankProv = $("cfg-RERANKER_PROVIDER")?.value || "api";
    let rerankStatus = "Untested";
    let rerankClass = "untested";
    if (connectionTestResults.reranker === "ok") {
      rerankStatus = "Ready";
      rerankClass = "ok";
    } else if (connectionTestResults.reranker === "bad") {
      rerankStatus = "Failed";
      rerankClass = "bad";
    }
    addReadyRow(list, `Reranker Model (${rerankModel})`, rerankStatus, rerankClass, null, `Receives search queries and candidate chunks (${rerankProv})`);
  } catch {
    lastReady = null;
    list.replaceChildren();
    const li = document.createElement("li");
    const n = document.createElement("span");
    n.textContent = "Service Status";
    const v = document.createElement("span");
    v.className = "bad";
    v.textContent = "Not connected";
    li.append(n, v);
    list.appendChild(li);
  }
}

/* ---- 视图切换：侧栏导航（对话 / 文档 / 设置） ---- */
/* 侧栏的历史对话列表在所有视图下都常驻——用户要能从文档页 / 设置页直接点回某段对话，
   点了会由 loadSession() 切回对话视图。只有 Pinned 区块按“是否有内容”显隐。 */
/* 「New Chat」是动作按钮，不是视图标签页：当前已经打开着某段会话时它不点亮，
   否则会和历史列表里的选中项重复高亮，看起来像是刚被点了 New Chat。
   高亮状态由「当前视图 + 是否已选中会话」推导，任何一处改了这两个值都要重算。 */
function syncNavActive() {
const hasSession = !!peekSessionId();
document.querySelectorAll("#side-nav button").forEach((b) => {
const view = b.dataset.view;
b.classList.toggle("is-active", view === currentView && !(view === "chat" && hasSession));
});
}
function setView(name) {
currentView = name;
["chat", "documents", "settings"].forEach((v) => { $(`view-${v}`).hidden = v !== name;});
syncNavActive();
closeSidebar();
if (name === "documents") loadDocuments();
if (name === "settings") { loadReady(); loadSettings(); }
}

/* ---- 侧栏（移动端） ---- */
function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("sidebar-backdrop")?.classList.remove("active");
}

/* ---- 接线 ---- */
sendBtn.addEventListener("click", () => {
if (streaming) { aborter?.abort(); return;}
ask();
});
input.addEventListener("keydown", (e) => {
if (e.key === "Enter" &&!e.shiftKey) { e.preventDefault(); if (!streaming) ask();}
});
input.addEventListener("input", () => {
input.style.height = "auto";
input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
if (!streaming) sendBtn.disabled = !input.value.trim();
});
$("side-nav").addEventListener("click", (e) => {
const btn = e.target.closest("button[data-view]");
if (!btn) return;
if (btn.dataset.view === "chat") startNewChat();
else setView(btn.dataset.view);
});
$("sidebar-collapse").addEventListener("click", () => {
  if (window.innerWidth <= 860) {
    closeSidebar();
  } else {
    $("sidebar").classList.toggle("collapsed");
  }
});
$("sidebar-expand").addEventListener("click", () => {
  if (window.innerWidth <= 860) {
    $("sidebar").classList.add("open");
    $("sidebar-backdrop")?.classList.add("active");
  } else {
    $("sidebar").classList.remove("collapsed");
  }
});
$("sidebar-backdrop")?.addEventListener("click", closeSidebar);
$("chats-toggle").addEventListener("click", () => {
  $("chats-toggle").classList.toggle("collapsed");
  $("history-content").classList.toggle("collapsed");
});
$("pinned-toggle").addEventListener("click", () => {
  $("pinned-toggle").classList.toggle("collapsed");
  $("pinned-content").classList.toggle("collapsed");
});
document.addEventListener("click", () => {
  document.querySelectorAll(".h-menu.show").forEach((m) => m.classList.remove("show"));
});
const dropzone = $("dropzone");
const fileInput = $("file-input");
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click();}
});
["dragover", "dragenter"].forEach((name) => dropzone.addEventListener(name, (e) => e.preventDefault()));
dropzone.addEventListener("drop", (e) => {
e.preventDefault();
if (e.dataTransfer.files.length) handleFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener("change", () => {
if (fileInput.files.length) handleFiles([...fileInput.files]);
fileInput.value = "";
});


async function initApp() {
  loadReady();
  loadSessions();
  loadDocuments();
  if (currentView === "settings") loadSettings();
  const current = peekSessionId();
  if (current) {
    try {
      const data = await request("GET", `/chat/sessions/${encodeURIComponent(current)}`);
      if (data.messages && data.messages.length) {
        thread.replaceChildren();
        data.messages.forEach((m) => {
          if (m.role === "user") addUserMsg(m.content || "");
          else renderAssistant(m.content || "", null);
        });
        renderHistory();
        scrollToBottom();
      }
    } catch { /* 会话已删或后端未就绪：留在空态 */}
  }
}

/* ---- 启动 ---- */
applyI18n();
syncEmptyState();
initSettingsEvents();
checkAuthStatus().then((isAuthed) => {
  if (isAuthed) {
    initApp();
  }
});
