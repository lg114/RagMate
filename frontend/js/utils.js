export const GROUNDING_COPY = {
  hedge: '检索置信度较低，以下内容可能不完整或不准确，请对照引用来源核对后再采信。',
  refusal: '没有找到足够相关的资料，无法可靠回答这个问题。请换个问法，或确认相关文档已经入库。',
  note: '注意：部分说法未能在检索资料中找到依据，请以上方引用来源为准。',
};

const FILE_CITE = String.raw`[^】]+\.(?:pdf|docx?|xlsx?|xls|txt|md)`;
const CITE_RE = new RegExp(`【(${FILE_CITE})(?:[,，]\\s*第(\\d+)页)?】`, 'gi');

export function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function formatFileSize(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(isoStr) {
  if (!isoStr) return '—';
  const d = new Date(isoStr);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function formatTime(isoStr) {
  if (!isoStr) return '';
  const d = new Date(isoStr);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

export function getDateKey(isoStr) {
  if (!isoStr) return '';
  const d = new Date(isoStr);
  if (Number.isNaN(d.getTime())) return '';
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

export function formatDateSeparator(isoStr) {
  if (!isoStr) return '';
  const d = new Date(isoStr);
  if (Number.isNaN(d.getTime())) return '';
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const target = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const diff = Math.floor((today - target) / 86400000);
  if (diff === 0) return '今天';
  if (diff === 1) return '昨天';
  if (diff === 2) return '前天';
  const month = d.getMonth() + 1;
  const day = d.getDate();
  return d.getFullYear() === now.getFullYear()
    ? `${month}月${day}日`
    : `${d.getFullYear()}年${month}月${day}日`;
}

export function statusLabel(status) {
  return {
    uploaded: '未入库',
    ingesting: '入库中',
    ingested: '已入库',
    failed: '失败',
  }[status] || status || '未知';
}

export function confidenceLabel(level) {
  return { high: '高置信', medium: '中置信', low: '低置信' }[level] || level || '';
}

export function dropReasonLabel(reason) {
  return {
    threshold: '低于阈值',
    per_source: '同源过多',
    score_gap: '分数断崖',
    top_k: '超出上限',
  }[reason] || '未保留';
}

export function fileExt(name) {
  const part = String(name || '').split('.').pop();
  return (part || '').toLowerCase();
}

export function normalizeCitations(text) {
  const source = String(text || '');
  const matches = [];
  const seen = new Set();
  let count = 0;
  CITE_RE.lastIndex = 0;
  let match;
  while ((match = CITE_RE.exec(source)) !== null) {
    count += 1;
    if (seen.has(match[1])) continue;
    seen.add(match[1]);
    matches.push({ source: match[1], page: match[2] ? Number(match[2]) : null });
  }
  if (count < 2) return source;

  let cleaned = source.replace(/^\s*数据来源[:：].*$/gm, '');
  matches.forEach((item) => {
    const escaped = item.source.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    cleaned = cleaned.replace(new RegExp(`【${escaped}(?:[,，]\\s*第\\d+页)?】`, 'g'), '');
  });
  cleaned = cleaned
    .replace(/[ \t]+([，。；：、,.!?！？])/g, '$1')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  const sourceLine = `数据来源：${matches.map((item) => (
    item.page ? `【${item.source}, 第${item.page}页】` : `【${item.source}】`
  )).join('、')}`;
  return cleaned ? `${cleaned}\n\n${sourceLine}` : sourceLine;
}

export function citationsFromText(text) {
  const citations = [];
  const seen = new Set();
  CITE_RE.lastIndex = 0;
  let match;
  const source = String(text || '');
  while ((match = CITE_RE.exec(source)) !== null) {
    const item = { source: match[1] };
    if (match[2]) item.page = Number(match[2]);
    const key = `${item.source}#${item.page || ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    citations.push(item);
  }
  return citations;
}

export function extractSourceLine(text) {
  const source = String(text || '');
  const line = source.match(/^\s*数据来源[:：]\s*(.*)$/m);
  if (!line) return { body: source.trim(), citations: citationsFromText(source) };
  const citations = citationsFromText(line[1]);
  const body = source.replace(/^\s*数据来源[:：].*$/gm, '').trim();
  return { body, citations: citations.length ? citations : citationsFromText(body) };
}

export function splitAnswerFrame(text, grounding) {
  let body = String(text || '').trim();
  let hedge = grounding === 'hedged';
  let refused = grounding === 'refused';
  if (body.startsWith(GROUNDING_COPY.hedge)) {
    hedge = true;
    body = body.slice(GROUNDING_COPY.hedge.length).trim();
  }
  if (body.includes(GROUNDING_COPY.note)) {
    body = body.split(GROUNDING_COPY.note).join('').trim();
  }
  if (refused || body === GROUNDING_COPY.refusal) {
    refused = true;
    body = '';
  }
  return { body, hedge, refused };
}

export function mergeCitations(primary, secondary) {
  const out = [];
  const seen = new Set();
  [...(primary || []), ...(secondary || [])].forEach((item) => {
    if (!item || !item.source) return;
    const page = item.page == null ? '' : item.page;
    const key = `${item.source}#${page}`;
    if (seen.has(key)) return;
    seen.add(key);
    const next = { source: item.source };
    if (item.page != null) next.page = item.page;
    out.push(next);
  });
  const paged = new Set(out.filter((item) => item.page != null).map((item) => item.source));
  return out.filter((item) => item.page != null || !paged.has(item.source));
}

function decodeHtml(html) {
  const el = document.createElement('textarea');
  el.innerHTML = html;
  return el.value;
}

function encodeBase64(value) {
  return btoa(unescape(encodeURIComponent(value)));
}

export function decodeBase64(value) {
  return decodeURIComponent(escape(atob(value)));
}

export function renderMarkdown(text) {
  const raw = window.marked.parse(text || '', { gfm: true, breaks: true });
  let html = window.DOMPurify.sanitize(raw);
  html = html.replace(/<pre><code(?: class="language-([\w-]+)")?>([\s\S]*?)<\/code><\/pre>/g, (_, lang, code) => {
    const label = lang || 'text';
    let decoded = code;
    try { decoded = decodeHtml(code); } catch { /* keep raw */ }
    let encoded = '';
    try { encoded = encodeBase64(decoded); } catch { encoded = ''; }
    return `<div class="code-block"><div class="code-block-header"><span>${escapeHtml(label)}</span><button type="button" class="btn-copy-code" data-code-b64="${encoded}">复制</button></div><pre><code>${code}</code></pre></div>`;
  });
  return html;
}

export function bindCodeCopy(root) {
  root.querySelectorAll('.btn-copy-code').forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = '1';
    btn.addEventListener('click', async () => {
      let code = '';
      try { code = decodeBase64(btn.dataset.codeB64 || ''); } catch { code = ''; }
      try {
        await navigator.clipboard.writeText(code);
        btn.textContent = '已复制';
        setTimeout(() => { btn.textContent = '复制'; }, 1400);
      } catch {
        btn.textContent = '复制失败';
      }
    });
  });
}

export function linkifyCitations(root, onActivate) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    const text = node.nodeValue;
    if (!text || !text.includes('【')) return;
    const parent = node.parentElement;
    if (!parent || parent.closest('button, a, pre, code')) return;
    const re = new RegExp(CITE_RE.source, 'gi');
    const frag = document.createDocumentFragment();
    let last = 0;
    let match;
    let any = false;
    while ((match = re.exec(text)) !== null) {
      any = true;
      if (match.index > last) frag.append(document.createTextNode(text.slice(last, match.index)));
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'cite-inline';
      btn.textContent = match[0];
      const citation = { source: match[1] };
      if (match[2]) citation.page = Number(match[2]);
      btn.setAttribute('aria-label', citation.page ? `查看来源 ${citation.source} 第 ${citation.page} 页` : `查看来源 ${citation.source}`);
      btn.addEventListener('click', () => onActivate(citation, btn));
      frag.append(btn);
      last = match.index + match[0].length;
    }
    if (!any) return;
    if (last < text.length) frag.append(document.createTextNode(text.slice(last)));
    node.parentNode.replaceChild(frag, node);
  });
}
