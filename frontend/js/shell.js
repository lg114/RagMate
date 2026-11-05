"use strict";
/* RagMate demo 壳：单文件前端逻辑，无构建、无依赖。
接口与旧工作台一致：POST /chat/stream（SSE）、GET /documents、
POST /documents/upload、POST /ingest、GET /ingest/status、GET /ready。
引用只含文件名与页码（后端 public_citations 的口径），无片段正文；
片段预览只在 debug 轨迹里有，壳默认不开 debug，所以来源卡只列文件。 */
const TOKEN_KEY = "ragmate_api_token";
const SESSION_KEY = "ragmate_session_id";

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const input = $("input");
const sendBtn = $("send");
const toastEl = $("toast");
let toastTimer = null;
let streaming = false;
let aborter = null;

function toast(text) {
toastEl.textContent = text;
toastEl.classList.add("show");
clearTimeout(toastTimer);
toastTimer = setTimeout(() => toastEl.classList.remove("show"), 3000);
}
function esc(s) {
return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
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
if (status === 401) return "需要 API 令牌。请在连接栏填写后重试。";
if (status === 413) return (data && data.detail) || "文件超过大小限制";
if (status === 429) return "请求过于频繁，请稍后再试。";
const detail = data && (data.detail || data.message);
return detail || `请求失败（${status}）`;
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
function getSessionId() {
try {
let sid = sessionStorage.getItem(SESSION_KEY);
if (!sid) {
sid = (crypto.randomUUID && crypto.randomUUID()) || `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;
sessionStorage.setItem(SESSION_KEY, sid);
}
return sid;
} catch {
return `s-${Date.now()}`;
}
}

/* ---- 最小 Markdown：转义 → 代码块 → 行内代码 → 加粗 → 链接 → 分段 ---- */
function renderMarkdown(src) {
const parts = src.split(/(```[\s\S]*?```)/g);
const html = parts.map((part) => {
if (part.startsWith("```")) {
const body = part.replace(/^```[^\n]*\n?/, "").replace(/```$/, "");
return `<pre><code>${esc(body)}</code></pre>`;
}
let t = esc(part);
t = t.replace(/`([^`\n]+)`/g, "<code>$1</code>");
t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
t = t.replace(/https?:\/\/[^\s<>"')\]]+/g, (url) => {
const clean = url.replace(/[.,;:!?]+$/, "");
const tail = url.slice(clean.length);
return `<a href="${clean}" target="_blank" rel="noopener noreferrer">${clean}</a>${tail}`;
});
return t.split(/\n{2,}/).map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`).join("");
});
return html.join("");
}
function splitSourceLine(text) {
const m = text.match(/\n\s*数据来源：([^\n]*)\s*$/);
if (!m) return { body: text, citations: []};
const cites = [];
for (const chunk of m[1].split("、")) {
const mm = chunk.match(/【(.+?)(?:[,，]\s*第(\d+)页)?】/);
if (mm) cites.push({ source: mm[1], page: mm[2] ? Number(mm[2]) : null });
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
div.scrollIntoView({ behavior: "smooth", block: "end"});
}
function renderSources(cites, anchor) {
const box = document.createElement("div");
box.className = "sources";
box.innerHTML = "<h3>来源</h3>";
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
loc.textContent = `第 ${c.page} 页`;
head.appendChild(loc);
}
card.appendChild(head);
box.appendChild(card);
});
return box;
}
function setSendMode(mode) {
if (mode === "streaming") {
sendBtn.textContent = "停止";
sendBtn.disabled = false;
} else {
sendBtn.textContent = "发送";
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
wrap.scrollIntoView({ behavior: "smooth", block: "end"});

aborter = new AbortController();
let fullText = "";
let doneData = null;
let failed = null;
try {
const res = await fetch("/chat/stream", {
method: "POST",
headers: authHeaders({ "Content-Type": "application/json"}),
body: JSON.stringify({ message: q, session_id: getSessionId(), replace_last: false, debug: false}),
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
wrap.scrollIntoView({ behavior: "smooth", block: "end"});
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
if (!doneData &&!failed) failed = "连接意外断开";
} catch (err) {
if (err && err.name === "AbortError") {
failed = "已停止生成";
} else {
failed = (err && err.message) || "请求失败，请重试";
}
} finally {
streaming = false;
aborter = null;
setSendMode("idle");
}

if (failed) {
if (!fullText) {
p.textContent = "这次没有生成完";
} else {
p.textContent = fullText;
}
const st = document.createElement("p");
st.className = "answer-status";
st.textContent = failed;
body.appendChild(st);
toast(failed);
input.focus();
return;
}

const text = (doneData.response?? fullText) || "";
if (doneData.session_id) {
try { sessionStorage.setItem(SESSION_KEY, doneData.session_id);} catch { /* ignore */}
}
const { body: cleanBody, citations: lineCites} = splitSourceLine(text);
const cites = (doneData.citations && doneData.citations.length)? doneData.citations: lineCites;
body.innerHTML = cleanBody? renderMarkdown(cleanBody): "<p>没有收到回复</p>";
if (cites.length) {
const anchor = `m${Date.now()}`;
cites.forEach((c, i) => {
const b = document.createElement("button");
b.type = "button";
b.className = "cite";
b.textContent = String(i + 1);
b.setAttribute("aria-label", `来源 ${c.source}${c.page? ` 第 ${c.page} 页`: ""}`);
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
const status = document.createElement("p");
status.className = "answer-status";
if (doneData.grounding === "refused") status.textContent = "没有足够资料，已拒答";
else if (doneData.grounding === "hedged") status.textContent = "低置信回答，请对照引用来源核对后再采信";
else if (doneData.unsupported_claims && doneData.unsupported_claims.length) status.textContent = "注意：部分说法未能在检索资料中找到依据，请以上方引用来源为准。";
else status.textContent = "回答已完成";
body.appendChild(status);
wrap.scrollIntoView({ behavior: "smooth", block: "end"});
input.focus();
}

/* ---- 文档一览与上传 ---- */
const STATUS_LABEL = { ingested: "已入库", uploaded: "待入库", ingesting: "入库中", failed: "入库失败"};
async function loadDocuments() {
const pill = $("corpus-pill");
try {
const data = await request("GET", "/documents");
const docs = data.documents || [];
const totalChunks = docs.reduce((n, d) => n + (d.chunk_count || 0), 0);
pill.textContent = docs.length? `${docs.length} 篇文档 · ${totalChunks} 块`: "暂无文档";
$("docs-count").textContent = docs.length? ` · ${docs.length} 篇 · ${totalChunks} 块`: "";
const list = $("doc-list");
list.replaceChildren();
$("doc-empty").hidden = docs.length > 0;
docs.forEach((d) => {
const li = document.createElement("li");
const name = document.createElement("span");
name.className = "mono";
name.textContent = d.filename;
const meta = document.createElement("span");
meta.className = "count";
meta.textContent = `${STATUS_LABEL[d.status] || d.status} · ${d.chunk_count || 0} 块`;
li.append(name, meta);
list.appendChild(li);
});
return docs;
} catch (err) {
pill.textContent = "文档加载失败";
toast(err.message || "文档加载失败");
return [];
}
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
xhr.onerror = () => reject(new Error("网络错误，上传失败"));
xhr.onabort = () => reject(new Error("上传已取消"));
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
dropzone.textContent = "拖入文件或点这里上传新文档";
dropzone.classList.remove("busy");
await loadDocuments();
toast("入库完成");
return;
}
dropzone.textContent = `入库中${st.message? ` · ${st.message}`: ""}`;
} catch {
break;
}
}
dropzone.textContent = "拖入文件或点这里上传新文档";
dropzone.classList.remove("busy");
await loadDocuments();
}
async function handleFiles(files) {
const dropzone = $("dropzone");
const docs = await loadDocuments();
for (const file of files) {
if (docs.some((d) => d.filename === file.name)) {
if (!window.confirm(`“${file.name}” 已存在。用新文件替换它？`)) continue;
try { await request("DELETE", `/documents/${encodeURIComponent(file.name)}`);} catch (err) { toast(err.message); continue;}
}
dropzone.classList.add("busy");
dropzone.textContent = `正在上传 ${file.name}…`;
try {
await uploadDocument(file);
dropzone.textContent = "正在入库…";
await request("POST", "/ingest", { filenames: [file.name]});
await pollIngest(dropzone);
} catch (err) {
dropzone.textContent = "拖入文件或点这里上传新文档";
dropzone.classList.remove("busy");
toast(err.message || "上传失败");
}
await loadDocuments();
}
}

/* ---- 状态与令牌 ---- */
async function loadReady() {
const pill = $("ready-pill");
try {
const res = await fetch("/ready", { headers: authHeaders()});
const data = await res.json().catch(() => ({}));
if (res.ok && (data.status === "ready" ||!data.status)) {
pill.textContent = "就绪";
pill.classList.add("ok");
} else {
pill.textContent = "服务异常";
}
} catch {
pill.textContent = "未连接";
}
}

/* ---- 接线 ---- */
$("chips").addEventListener("click", (e) => {
const chip = e.target.closest(".chip");
if (chip &&!streaming) ask(chip.dataset.q);
});
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
toast("令牌已保存（仅本浏览器）");
} catch {
toast("浏览器不允许保存");
}
});
loadReady();
loadDocuments();
