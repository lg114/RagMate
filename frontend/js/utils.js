export function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function truncate(str, len) {
  return str.length > len ? str.slice(0, len - 3) + '...' : str;
}

export function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export function formatDate(isoStr) {
  if (!isoStr) return '-';
  const d = new Date(isoStr);
  return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

export function formatTime(isoStr) {
  if (!isoStr) return '';
  return new Date(isoStr).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

export function getDateKey(isoStr) {
  if (!isoStr) return '';
  const d = new Date(isoStr);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

export function formatDateSeparator(isoStr) {
  if (!isoStr) return '';
  const d = new Date(isoStr);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const target = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const diff = Math.floor((today - target) / 86400000);
  if (diff === 0) return '今天';
  if (diff === 1) return '昨天';
  if (diff === 2) return '前天';
  const isSameYear = d.getFullYear() === now.getFullYear();
  const month = d.getMonth() + 1;
  const day = d.getDate();
  return isSameYear ? `${month}月${day}日` : `${d.getFullYear()}年${month}月${day}日`;
}

export function normalizeCitations(text) {
  const citationRe = /【([^】]+\.(?:pdf|docx?|xlsx?|xls|txt|md))(?:[,，]\s*第\d+页)?】/gi;
  const sources = [];
  let citationCount = 0;
  let match;
  while ((match = citationRe.exec(text)) !== null) {
    citationCount += 1;
    if (!sources.includes(match[1])) sources.push(match[1]);
  }
  if (citationCount < 2) return text;

  let cleaned = text.replace(/^\s*数据来源[:：].*$/gm, '');
  sources.forEach(source => {
    cleaned = cleaned.replace(new RegExp(`【${source.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:[,，]\\s*第\\d+页)?】`, 'g'), '');
  });
  cleaned = cleaned
    .replace(/[ \t]+([，。；：、,.!?！？])/g, '$1')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  const sourceLine = '数据来源：' + sources.map(source => `【${source}】`).join('、');
  return cleaned ? `${cleaned}\n\n${sourceLine}` : sourceLine;
}

export function renderAssistantMarkdown(text) {
  const displayText = normalizeCitations(text);
  const withSourceChips = displayText.replace(/^数据来源[:：]\s*(.*)$/m, (_, rawSources) => {
    const sources = [];
    rawSources.replace(/【([^】]+)】/g, (_m, source) => {
      if (!sources.includes(source)) sources.push(source);
      return '';
    });
    if (sources.length === 0) return escapeHtml(`数据来源：${rawSources}`);
    const chips = sources
      .map(source => `<span class="source-chip" title="${escapeHtml(source)}">${escapeHtml(source)}</span>`)
      .join('');
    return `<div class="source-row"><span class="source-label">来源</span>${chips}</div>`;
  });
  let html = DOMPurify.sanitize(marked.parse(withSourceChips));
  html = html.replace(/<pre><code(?: class="language-(\w+)")?>([\s\S]*?)<\/code><\/pre>/g, (_, lang, code) => {
    const label = lang || 'code';
    const encoded = btoa(unescape(encodeURIComponent(code)));
    return `<div class="code-block-header"><span>${escapeHtml(label)}</span><button class="btn-copy-code" data-code-b64="${encoded}">复制</button></div><pre><code>${code}</code></pre>`;
  });
  return html;
}

export function statusLabel(s) {
  const map = { uploaded: '未入库', ingesting: '入库中', ingested: '已入库', failed: '失败' };
  return map[s] || s;
}
