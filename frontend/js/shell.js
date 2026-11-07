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
const TOKEN_KEY = "ragmate_api_token";
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
function getToken() {
try { return (localStorage.getItem(TOKEN_KEY) || "").trim();} catch { return "";}
}
function authHeaders(extra) {
const headers = {...(extra || {})};
const token = getToken();
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
if (!res.ok) throw new Error(errorFrom(res.status, data));
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
return lastDocs;
}
function uploadDocument(file) {
return new Promise((resolve, reject) => {
const xhr = new XMLHttpRequest();
xhr.open("POST", "/documents/upload");
const token = getToken();
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


/* ---- 设置管理 (Settings Management) ---- */
let currentSettingsData = null;
let configuredSecrets = new Set();

const FIELD_DEFS = {
  // text / password / select inputs
  "LLM_MODEL": { type: "text", required: true },
  "LLM_API_KEY": { type: "secret" },
  "LLM_API_BASE_URL": { type: "text" },
  "LLM_REASONING_EFFORT": { type: "select" },
  "EMBEDDING_PROVIDER": { type: "select" },
  "EMBEDDING_MODEL": { type: "text" },
  "EMBEDDING_API_BASE_URL": { type: "text" },
  "EMBEDDING_API_KEY": { type: "secret" },
  "RERANKER_PROVIDER": { type: "select" },
  "RERANKER_MODEL": { type: "text" },
  "RERANKER_API_BASE_URL": { type: "text" },
  "RERANKER_API_KEY": { type: "secret" },
  "API_AUTH_TOKEN": { type: "secret" },
  "CORS_ORIGINS": { type: "text" },
  "LANGSMITH_API_KEY": { type: "secret" },
  "LANGSMITH_PROJECT": { type: "text" },
  "LANGSMITH_ENDPOINT": { type: "text" },
  // numbers
  "FINAL_CONTEXT_K": { type: "number", min: 1, max: 50 },
  "RERANK_SCORE_THRESHOLD": { type: "number", min: 0, max: 1 },
  "RERANK_CANDIDATES": { type: "number", min: 1, max: 200 },
  "RRF_K": { type: "number", min: 1, max: 200 },
  "LLM_TEMPERATURE": { type: "number", min: 0, max: 2 },
  "MAX_HISTORY_TURNS": { type: "number", min: 1, max: 50 },
  "AGENT_TIMEOUT": { type: "number", min: 5, max: 600 },
  "RATE_LIMIT_MAX": { type: "number", min: 1, max: 1000 },
  "RATE_LIMIT_WINDOW": { type: "number", min: 1, max: 3600 },
  "CHUNK_SIZE": { type: "number", min: 50, max: 5000 },
  "CHUNK_OVERLAP": { type: "number", min: 0, max: 2000 },
  "CHUNK_SIZE_PARENT": { type: "number", min: 100, max: 10000 },
  "CHUNK_OVERLAP_PARENT": { type: "number", min: 0, max: 3000 },
  // booleans
  "QUERY_CONTEXTUALIZE": { type: "boolean" },
  "FAITHFULNESS_CHECK": { type: "boolean" },
  "LANGSMITH_TRACING": { type: "boolean" },
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
    if (!res.ok) {
      console.warn("Settings API not available or error:", res.status);
      return;
    }
    const data = await res.json();
    currentSettingsData = data.settings || {};
    configuredSecrets = new Set(data.secret_keys || []);

    // Populate fields
    for (const [key, def] of Object.entries(FIELD_DEFS)) {
      const el = $(`cfg-${key}`);
      if (!el) continue;
      const val = currentSettingsData[key];

      if (def.type === "boolean") {
        el.checked = Boolean(val);
      } else if (def.type === "secret") {
        el.value = "";
        const isConf = Boolean(currentSettingsData[key + "_configured"]);
        const tag = $(`tag-${key}`);
        if (isConf) {
          el.placeholder = "Configured (enter new value to change)";
          if (tag) {
            tag.textContent = "Configured";
            tag.classList.add("configured");
          }
        } else {
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

    // Special field: MAX_UPLOAD_SIZE (bytes to MB)
    const uploadMbEl = $("cfg-MAX_UPLOAD_SIZE_MB");
    if (uploadMbEl && currentSettingsData.MAX_UPLOAD_SIZE) {
      const bytes = Number(currentSettingsData.MAX_UPLOAD_SIZE) || 0;
      uploadMbEl.value = Math.round(bytes / (1024 * 1024));
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
    const el = $(`cfg-${key}`);
    if (!el) continue;

    if (def.type === "boolean") {
      updates[key] = el.checked;
    } else if (def.type === "secret") {
      const val = el.value.trim();
      if (val) {
        updates[key] = val; // only send if newly entered
      }
    } else if (def.type === "number") {
      const raw = el.value.trim();
      if (raw === "") continue;
      const num = Number(raw);
      if (isNaN(num) || (def.min !== undefined && num < def.min) || (def.max !== undefined && num > def.max)) {
        el.classList.add("field-error");
        toast(`${key} is out of valid range (${def.min ?? "-"} ~ ${def.max ?? "-"})`);
        hasError = true;
        break;
      }
      updates[key] = num;
    } else {
      updates[key] = el.value.trim();
    }
  }

  // Handle MAX_UPLOAD_SIZE_MB conversion
  const uploadMbEl = $("cfg-MAX_UPLOAD_SIZE_MB");
  if (uploadMbEl && uploadMbEl.value.trim() !== "") {
    const mb = Number(uploadMbEl.value.trim());
    if (mb > 0) updates["MAX_UPLOAD_SIZE"] = mb * 1024 * 1024;
  }

  if (hasError) {
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
      body: JSON.stringify({ settings: updates }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "保存失败");
    }

    const result = await res.json();
    if (result.restart_required) {
      const banner = $("settings-restart-banner");
      if (banner) banner.hidden = false;
      toast("Settings saved! A restart is required for changes to take effect.");
    } else {
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

async function resetAllSettings() {
  const ok = await confirm("Reset all settings to default values? (Existing DATABASE_URL will be preserved).");
  if (!ok) return;

  try {
    const res = await fetch("/settings/reset", {
      method: "POST",
      headers: authHeaders(),
    });
    if (!res.ok) throw new Error("重置失败");
    toast("Settings reset to defaults. A server restart is required.");
    const banner = $("settings-restart-banner");
    if (banner) banner.hidden = false;
    await loadSettings();
    await loadReady();
  } catch (err) {
    toast(err.message || "Failed to reset settings");
  }
}

function initSettingsEvents() {
  $("cfg-EMBEDDING_PROVIDER")?.addEventListener("change", updateSettingsLinkage);
  $("cfg-RERANKER_PROVIDER")?.addEventListener("change", updateSettingsLinkage);
  $("cfg-LANGSMITH_TRACING")?.addEventListener("change", updateSettingsLinkage);
  $("btn-save-settings")?.addEventListener("click", saveAllSettings);
  $("btn-reset-settings")?.addEventListener("click", resetAllSettings);
  $("btn-refresh-ready")?.addEventListener("click", loadReady);
  $("settings-restart-close")?.addEventListener("click", () => {
    const banner = $("settings-restart-banner");
    if (banner) banner.hidden = true;
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
function addReadyRow(list, label, value, cls, hint) {
const li = document.createElement("li");
const n = document.createElement("span");
n.textContent = label;
const v = document.createElement("span");
v.className = cls;
v.textContent = value;
if (hint) { li.title = hint; v.title = hint; }
li.append(n, v);
list.appendChild(li);
}
async function loadReady() {
const list = $("ready-list");
try {
const res = await fetch("/ready", { headers: authHeaders()});
const data = await res.json().catch(() => ({}));
lastReady = data;
/* 模型名没配置就整块隐藏——宁可空着，也不显示一个没在用的模型。 */
const chip = $("model-chip");
if (data.model) {
chip.textContent = data.model;
chip.title = data.model;
chip.hidden = false;
}
const checks = data.checks || {};
list.replaceChildren();
const title = document.createElement("li");
const tn = document.createElement("span");
tn.textContent = t("readinessTitle");
const tv = document.createElement("span");
tv.className = res.ok && data.status === "ready"? "ok": "bad";
tv.textContent = res.ok && data.status === "ready"? t("readyOk"): t("readyBad");
title.append(tn, tv);
list.appendChild(title);
Object.entries(checks).forEach(([name, ok]) => {
const label = name === "llm"? t("llmCheck"): name;
const hintKey = name === "llm" && !ok? LLM_HINTS[data.llm_detail]: null;
addReadyRow(list, label, ok? t("readyOk"): t("readyBad"), ok? "ok": "bad", hintKey? t(hintKey): "");
});
/* 模型名与部署位置放在服务状态里，而不是塞进输入框的 chip——
   输入框保持安静，需要判断"数据会不会出本机"时到这里看。 */
if (data.model) {
addReadyRow(list, t("modelLabel"), data.model, "ok");
const local = data.model_source === "local";
addReadyRow(list, t("deployLabel"), local? t("deployLocal"): t("deployRemote"), "ok");
} else {
addReadyRow(list, t("modelLabel"), t("modelMissing"), "bad");
}
} catch {
lastReady = null;
list.replaceChildren();
const li = document.createElement("li");
const n = document.createElement("span");
n.textContent = t("readinessTitle");
const v = document.createElement("span");
v.className = "bad";
v.textContent = t("readyUnknown");
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
function closeSidebar() { $("sidebar").classList.remove("open");}

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
$("sidebar-collapse").addEventListener("click", () => $("sidebar").classList.toggle("collapsed"));
$("sidebar-expand").addEventListener("click", () => $("sidebar").classList.remove("collapsed"));
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
if ($("token-input")) $("token-input").value = getToken();
$("token-save")?.addEventListener("click", () => {
const value = $("token-input")?.value?.trim() || "";
try {
if (value) localStorage.setItem(TOKEN_KEY, value);
else localStorage.removeItem(TOKEN_KEY);
toast(t("tokenSaved"));
} catch {
toast(t("tokenSaveFail"));
}
});

/* ---- 启动 ---- */
applyI18n();
syncEmptyState();
loadReady();
initSettingsEvents();
loadSettings();
loadDocuments();
loadSessions();
(async () => {
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
})();
