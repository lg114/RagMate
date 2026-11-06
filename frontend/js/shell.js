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
$("lang-toggle").textContent = LANG === "zh"? "EN": "中文";
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

function toast(text) {
toastEl.textContent = text;
toastEl.classList.add("show");
clearTimeout(toastTimer);
toastTimer = setTimeout(() => toastEl.classList.remove("show"), 3000);
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
function setSendMode(mode) {
if (mode === "streaming") {
sendBtn.textContent = t("stop");
sendBtn.disabled = false;
} else {
sendBtn.textContent = t("send");
sendBtn.disabled = false;
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
function renderHistory() {
const list = $("history-list");
list.replaceChildren();
$("history-empty").hidden = lastSessions.length > 0;
const current = peekSessionId();
lastSessions.forEach((s) => {
const li = document.createElement("li");
if (s.session_id === current) li.classList.add("is-active");
const title = document.createElement("span");
title.className = "h-title";
title.textContent = s.first_message || s.session_id.slice(0, 8);
title.title = s.first_message || s.session_id;
const del = document.createElement("button");
del.type = "button";
del.className = "h-del";
del.textContent = "×";
del.setAttribute("aria-label", t("deleteSessionConfirm"));
del.addEventListener("click", async (e) => {
e.stopPropagation();
if (!window.confirm(t("deleteSessionConfirm"))) return;
try {
await request("DELETE", `/chat/sessions/${encodeURIComponent(s.session_id)}`);
if (peekSessionId() === s.session_id) startNewChat();
await loadSessions();
} catch (err) { toast(err.message);}
});
li.append(title, del);
li.addEventListener("click", () => loadSession(s.session_id));
list.appendChild(li);
});
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
}
function startNewChat() {
try { sessionStorage.removeItem(SESSION_KEY);} catch { /* ignore */}
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
if (!window.confirm(t("deleteDocConfirm", { name: d.filename}))) return;
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
if (!window.confirm(t("replaceConfirm", { name: file.name }))) continue;
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

/* ---- 设置：令牌与服务状态 ---- */
async function loadReady() {
const list = $("ready-list");
try {
const res = await fetch("/ready", { headers: authHeaders()});
const data = await res.json().catch(() => ({}));
lastReady = data;
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
const li = document.createElement("li");
const n = document.createElement("span");
n.textContent = name;
const v = document.createElement("span");
v.className = ok? "ok": "bad";
v.textContent = ok? t("readyOk"): t("readyBad");
li.append(n, v);
list.appendChild(li);
});
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
let currentView = "chat";
function setView(name) {
currentView = name;
["chat", "documents", "settings"].forEach((v) => { $(`view-${v}`).hidden = v !== name;});
document.querySelectorAll("#side-nav button").forEach((b) => b.classList.toggle("is-active", b.dataset.view === name));
$("history-block").hidden = name !== "chat";
closeSidebar();
if (name === "documents") loadDocuments();
if (name === "settings") loadReady();
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
});
$("menu").addEventListener("click", () => $("sidebar").classList.toggle("open"));
$("side-nav").addEventListener("click", (e) => {
const btn = e.target.closest("button[data-view]");
if (btn) setView(btn.dataset.view);
});
$("new-chat").addEventListener("click", startNewChat);
$("lang-toggle").addEventListener("click", () => {
LANG = LANG === "zh"? "en": "zh";
try { localStorage.setItem(LANG_KEY, LANG);} catch { /* ignore */}
applyI18n();
renderDocs();
renderHistory();
loadReady();
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
$("token-input").value = getToken();
$("token-save").addEventListener("click", () => {
const value = $("token-input").value.trim();
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
loadReady();
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
