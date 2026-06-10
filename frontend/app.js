/* ══════════════════════════════════════════
   RagMate Frontend — app.js
   ══════════════════════════════════════════ */

// ── API Client ──
const API = {
  async request(method, path, body) {
    const opts = { method };
    if (body instanceof FormData) {
      opts.body = body;
    } else if (body !== undefined) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
    return data;
  },

  async chatStream(message, sessionId, onToken, onDone, onError, replaceLast) {
    try {
      const res = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId, replace_last: !!replaceLast }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Error ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.done) { onDone(data); return; }
            if (data.error) { onError(data.error); return; }
            if (data.token) onToken(data.token);
          } catch (e) {
            // 忽略无法解析的行
          }
        }
      }
      // 处理 buffer 中残留的数据
      if (buffer.trim()) {
        for (const line of buffer.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.done) { onDone(data); return; }
            if (data.error) { onError(data.error); return; }
            if (data.token) onToken(data.token);
          } catch (e) { /* ignore */ }
        }
      }
      onError('连接意外断开');
    } catch (err) {
      onError(err.message || '请求失败，请重试');
    }
  },

  getDocuments() {
    return this.request('GET', '/documents');
  },

  uploadDocument(file) {
    const form = new FormData();
    form.append('file', file);
    return this.request('POST', '/documents/upload', form);
  },

  deleteDocument(filename) {
    return this.request('DELETE', `/documents/${encodeURIComponent(filename)}`);
  },

  startIngest(filenames) {
    return filenames ? this.request('POST', '/ingest', { filenames }) : this.request('POST', '/ingest');
  },

  getIngestStatus() {
    return this.request('GET', '/ingest/status');
  },

  getSessions() {
    return this.request('GET', '/chat/sessions');
  },

  getHistory(sessionId) {
    return this.request('GET', `/chat/sessions/${encodeURIComponent(sessionId)}`);
  },

  deleteSession(sessionId) {
    return this.request('DELETE', `/chat/sessions/${encodeURIComponent(sessionId)}`);
  }
};

// ── Helpers ──
function showConfirm(message) {
  return new Promise((resolve) => {
    const overlay = document.getElementById('modal-overlay');
    document.getElementById('modal-message').textContent = message;
    overlay.classList.remove('hidden');
    const cleanup = (result) => { overlay.classList.add('hidden'); resolve(result); };
    document.getElementById('modal-confirm').onclick = () => cleanup(true);
    document.getElementById('modal-cancel').onclick = () => cleanup(false);
  });
}

function showProgressModal(title, total) {
  const overlay = document.getElementById('modal-overlay');
  const box = overlay.querySelector('.modal-box');
  const origHTML = box.innerHTML;

  box.innerHTML = `
    <p class="modal-progress-text">${escapeHtml(title)} <span id="prog-count">0/${total}</span></p>
    <div class="modal-progress-bar-wrap"><div class="modal-progress-bar" id="prog-bar"></div></div>
    <div class="modal-progress-current" id="prog-current"></div>
    <div class="modal-actions"><button class="btn-modal btn-modal-cancel" id="prog-cancel">取消</button></div>
  `;
  overlay.classList.remove('hidden');

  let cancelled = false;
  document.getElementById('prog-cancel').onclick = () => { cancelled = true; };

  return {
    get cancelled() { return cancelled; },
    update(index, filename) {
      document.getElementById('prog-count').textContent = `${index + 1}/${total}`;
      document.getElementById('prog-bar').style.width = `${((index + 1) / total) * 100}%`;
      document.getElementById('prog-current').textContent = filename;
    },
    done(title, items, detail) {
      // items: { ok: [{name}], skipped: [{name, reason}], fail: [{name, error}] }
      let html = `<p style="margin-bottom:16px;font-size:14px;font-weight:600;">${escapeHtml(title)}</p>`;
      if (items.ok && items.ok.length > 0) {
        html += `<p class="modal-result-item modal-result-ok">✓ 已入库：${items.ok.length} 个</p>`;
        items.ok.forEach(f => {
          if (f.name) html += `<p class="modal-result-detail">- ${escapeHtml(f.name)}</p>`;
        });
      }
      if (items.skipped && items.skipped.length > 0) {
        html += `<p class="modal-result-item modal-result-skip">⊘ 跳过：${items.skipped.length} 个</p>`;
        items.skipped.forEach(f => {
          html += `<p class="modal-result-detail">- ${escapeHtml(f.name)}（${escapeHtml(f.reason || '内容重复')}）</p>`;
        });
      }
      if (items.fail && items.fail.length > 0) {
        html += `<p class="modal-result-item modal-result-fail">✗ 失败：${items.fail.length} 个</p>`;
        items.fail.forEach(f => {
          html += `<p class="modal-result-detail">- ${escapeHtml(f.name)}（${escapeHtml(f.error)}）</p>`;
        });
      }
      if (detail) html += `<p class="modal-result-detail" style="margin-top:8px;color:var(--text-muted);">${escapeHtml(detail)}</p>`;
      html += `<div class="modal-actions" style="margin-top:20px"><button class="btn-modal btn-modal-confirm" id="prog-close">关闭</button></div>`;
      box.innerHTML = html;
      document.getElementById('prog-close').onclick = () => {
        overlay.classList.add('hidden');
        box.innerHTML = origHTML;
      };
    },
    close() {
      overlay.classList.add('hidden');
      box.innerHTML = origHTML;
    }
  };
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatDate(isoStr) {
  if (!isoStr) return '-';
  const d = new Date(isoStr);
  return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function truncate(str, len) {
  return str.length > len ? str.slice(0, len - 3) + '...' : str;
}

function normalizeCitations(text) {
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
    // 移除所有匹配该文件名的引用（可能带页码）
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

function renderAssistantMarkdown(text) {
  const displayText = normalizeCitations(text);
  const withSourceChips = displayText.replace(/^数据来源[:：]\s*(.*)$/m, (_, rawSources) => {
    const sources = [];
    rawSources.replace(/【([^】]+)】/g, (_match, source) => {
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
  // Add code block headers with copy buttons
  html = html.replace(/<pre><code(?: class="language-(\w+)")?>([\s\S]*?)<\/code><\/pre>/g, (_, lang, code) => {
    const label = lang || 'code';
    const encoded = btoa(unescape(encodeURIComponent(code)));
    return `<div class="code-block-header"><span>${escapeHtml(label)}</span><button class="btn-copy-code" data-code-b64="${encoded}">复制</button></div><pre><code>${code}</code></pre>`;
  });
  return html;
}

// ── Session ID ──
function getSessionId() {
  let sid = sessionStorage.getItem('ragmate_session_id');
  if (!sid) {
    sid = crypto.randomUUID?.() || (Date.now().toString(36) + Math.random().toString(36).slice(2));
    sessionStorage.setItem('ragmate_session_id', sid);
  }
  return sid;
}

function newSession() {
  sessionStorage.removeItem('ragmate_session_id');
}

// ═══════════════════════ Chat Panel ═══════════════════════

const ChatPanel = {
  messagesEl: document.getElementById('chat-messages'),
  formEl: document.getElementById('chat-form'),
  textareaEl: document.getElementById('chat-textarea'),
  sendBtn: document.getElementById('btn-send'),
  errorEl: document.getElementById('chat-error'),
  loading: false,

  init() {
    this.formEl.addEventListener('submit', (e) => { e.preventDefault(); this.send(); });
    document.getElementById('btn-new-chat').addEventListener('click', () => {
      this.clear();
      HistoryPanel.activeId = null;
      HistoryPanel.load();
    });

    // Textarea 自动增高 + Enter 发送 / Shift+Enter 换行
    this.textareaEl.addEventListener('input', () => {
      this.textareaEl.style.height = 'auto';
      const fontSize = parseFloat(getComputedStyle(this.textareaEl).fontSize);
      const lineHeight = parseFloat(getComputedStyle(this.textareaEl).lineHeight) || fontSize * 1.5;
      this.textareaEl.style.height = Math.min(this.textareaEl.scrollHeight, lineHeight * 5) + 'px';
    });
    this.textareaEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.send();
      }
    });
  },

  addMessage(role, text) {
    const empty = this.messagesEl.querySelector('.empty-state');
    if (empty) empty.remove();

    // Add divider between messages
    const hasMessages = this.messagesEl.querySelector('.msg');
    if (hasMessages) {
      const divider = document.createElement('div');
      divider.className = 'msg-divider';
      this.messagesEl.appendChild(divider);
    }

    const div = document.createElement('div');
    div.className = 'msg ' + (role === 'user' ? 'msg-user' : 'msg-assistant');

    const userIcon = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M6 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/></svg>`;
    const aiIcon = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>`;

    if (role === 'assistant') {
      div.innerHTML = `<div class="msg-body"><div class="msg-role"><span class="msg-role-icon ai-icon">${aiIcon}</span>RagMate</div><div class="msg-content">${renderAssistantMarkdown(text)}</div></div>`;
    } else {
      div.innerHTML = `<div class="msg-body"><div class="msg-role"><span class="msg-role-icon user-icon">${userIcon}</span>You</div><div class="msg-content">${DOMPurify.sanitize(text)}</div></div>`;
    }
    this.messagesEl.appendChild(div);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    return div;
  },

  showLoading() {
    const div = document.createElement('div');
    div.className = 'msg-loading typing-dots';
    div.innerHTML = '思考中<span>.</span><span>.</span><span>.</span>';
    div.id = 'msg-loading';
    this.messagesEl.appendChild(div);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  },

  hideLoading() {
    const el = document.getElementById('msg-loading');
    if (el) el.remove();
  },

  showError(msg) {
    this.errorEl.innerHTML = '';
    this.errorEl.appendChild(document.createTextNode(msg + ' '));
    const retryBtn = document.createElement('button');
    retryBtn.className = 'btn-retry';
    retryBtn.textContent = '重试';
    retryBtn.addEventListener('click', () => {
      if (this._lastUserText && !this.loading) {
        this.hideError();
        // 移除最后一条空的 assistant 消息和 divider
        const lastMsg = this.messagesEl.querySelector('.msg-assistant:last-of-type');
        if (lastMsg) {
          const prev = lastMsg.previousElementSibling;
          if (prev && prev.classList.contains('msg-divider')) prev.remove();
          lastMsg.remove();
        }
        // 移除最后一条用户消息和 divider
        const userMsg = this.messagesEl.querySelector('.msg-user:last-of-type');
        if (userMsg) {
          const prev = userMsg.previousElementSibling;
          if (prev && prev.classList.contains('msg-divider')) prev.remove();
          userMsg.remove();
        }
        this.textareaEl.value = this._lastUserText;
        this.send(true);
      }
    });
    this.errorEl.appendChild(retryBtn);
    this.errorEl.classList.remove('hidden');
  },

  hideError() {
    this.errorEl.classList.add('hidden');
  },

  setDisabled(disabled) {
    this.loading = disabled;
    this.sendBtn.disabled = disabled;
    this.textareaEl.disabled = disabled;
  },

  async send(replaceLast) {
    const text = this.textareaEl.value.trim();
    if (!text || this.loading) return;

    this.textareaEl.value = '';
    this.textareaEl.style.height = 'auto';
    this.hideError();
    this._lastUserText = text;
    this.addMessage('user', text);
    this.setDisabled(true);

    const sid = getSessionId();
    const streamDiv = this.startStreamMessage();
    let fullText = '';

    await API.chatStream(
      text,
      sid,
      (token) => { fullText += token; this.appendStreamToken(streamDiv, fullText); },
      (doneData) => {
        this.finalizeStreamMessage(streamDiv, fullText, doneData);
        sessionStorage.setItem('ragmate_session_id', doneData.session_id);
        this.setDisabled(false);
        this.textareaEl.focus();
        HistoryPanel.load();
      },
      (errMsg) => {
        this.finalizeStreamMessage(streamDiv, fullText || '');
        this.showError(errMsg);
        this.setDisabled(false);
        this.textareaEl.focus();
        HistoryPanel.load();
      },
      replaceLast,
    );
  },

  startStreamMessage() {
    const empty = this.messagesEl.querySelector('.empty-state');
    if (empty) empty.remove();

    // Add divider
    const hasMessages = this.messagesEl.querySelector('.msg');
    if (hasMessages) {
      const divider = document.createElement('div');
      divider.className = 'msg-divider';
      this.messagesEl.appendChild(divider);
    }

    const aiIcon = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>`;

    const div = document.createElement('div');
    div.className = 'msg msg-assistant';
    div.innerHTML = `<div class="msg-body"><div class="msg-role"><span class="msg-role-icon ai-icon">${aiIcon}</span>RagMate</div><div class="msg-content"><div class="msg-loading typing-dots">检索与生成中<span>.</span><span>.</span><span>.</span></div></div></div>`;
    this.messagesEl.appendChild(div);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    return div;
  },

  appendStreamToken(div, fullText) {
    const content = div.querySelector('.msg-content');
    if (div._streamDone) return;
    if (!div._rafPending) {
      div._rafPending = true;
      requestAnimationFrame(() => {
        if (div._streamDone) return;
        content.innerHTML = DOMPurify.sanitize(marked.parse(fullText));
        // Smart scroll: only auto-scroll if user is near bottom
        const el = this.messagesEl;
        const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
        if (isNearBottom) el.scrollTop = el.scrollHeight;
        div._rafPending = false;
      });
    }
  },

  finalizeStreamMessage(div, fullText, doneData = {}) {
    div._streamDone = true;
    const content = div.querySelector('.msg-content');
    content.innerHTML = renderAssistantMarkdown(fullText || '没有收到回复');

    // 置信度指示器
    if (doneData.confidence) {
      const level = doneData.confidence.level;
      const label = { high: '高置信', medium: '中置信', low: '低置信' }[level] || level;
      const badge = document.createElement('div');
      badge.className = `confidence-badge confidence-${level}`;
      badge.textContent = label;
      badge.title = `最高相关度: ${doneData.confidence.score}, 引用片段: ${doneData.confidence.chunks}`;
      content.appendChild(badge);
    }

    // 忠诚度警告
    if (doneData.unsupported_claims && doneData.unsupported_claims.length > 0) {
      const warn = document.createElement('div');
      warn.className = 'faithfulness-warning';
      warn.innerHTML = `⚠ 以下声明可能缺乏文献支撑：<ul>${doneData.unsupported_claims.map(c => `<li>${escapeHtml(c.claim)}</li>`).join('')}</ul>`;
      content.appendChild(warn);
    }

    // Add action buttons
    const actionsHtml = `<div class="msg-actions">
      <button class="msg-action-btn" data-action="copy" title="复制">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        复制
      </button>
      <button class="msg-action-btn" data-action="regenerate" title="重新生成">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
        重新生成
      </button>
    </div>`;
    content.insertAdjacentHTML('beforeend', actionsHtml);

    // Bind action handlers
    const msgBody = div.querySelector('.msg-body') || content.parentElement;
    const copyBtn = msgBody.querySelector('[data-action="copy"]');
    const regenBtn = msgBody.querySelector('[data-action="regenerate"]');

    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        const textToCopy = fullText || '';
        navigator.clipboard.writeText(textToCopy).then(() => {
          copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> 已复制`;
          setTimeout(() => {
            copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> 复制`;
          }, 1500);
        });
      });
    }

    if (regenBtn) {
      regenBtn.addEventListener('click', () => {
        if (this._lastUserText && !this.loading) {
          const lastText = this._lastUserText;
          // Remove the last AI message and its divider
          const lastMsg = div;
          const prevSibling = lastMsg.previousElementSibling;
          if (prevSibling && prevSibling.classList.contains('msg-divider')) prevSibling.remove();
          lastMsg.remove();
          // Remove the last user message and its divider
          const userMsg = this.messagesEl.querySelector('.msg-user:last-of-type');
          if (userMsg) {
            const userDivider = userMsg.previousElementSibling;
            if (userDivider && userDivider.classList.contains('msg-divider')) userDivider.remove();
            userMsg.remove();
          }
          this.textareaEl.value = lastText;
          this.send(true);
        }
      });
    }

    // Bind code copy buttons
    content.querySelectorAll('.btn-copy-code').forEach(btn => {
      btn.addEventListener('click', () => {
        const code = decodeURIComponent(escape(atob(btn.dataset.codeB64)));
        navigator.clipboard.writeText(code).then(() => {
          btn.textContent = '已复制';
          btn.classList.add('copied');
          setTimeout(() => {
            btn.textContent = '复制';
            btn.classList.remove('copied');
          }, 1500);
        });
      });
    });
  },

  clear() {
    newSession();
    this.messagesEl.innerHTML = `
      <div id="hero-empty" class="hero-empty">
        <canvas id="hero-canvas" class="hero-canvas"></canvas>
        <div class="hero-content">
          <h1 class="hero-title">你的<span class="accent">知识库</span> AI 助手</h1>
          <p class="hero-subtitle">上传文档，建立专属知识库。提问即检索，精准溯源，拒绝幻觉。</p>
          <div class="hero-suggestions">
            <button class="hero-suggestion" data-q="这个知识库包含哪些文档？">这个知识库包含哪些文档？</button>
            <button class="hero-suggestion" data-q="帮我总结一下核心要点">帮我总结核心要点</button>
            <button class="hero-suggestion" data-q="有哪些常见问题和解决方案？">常见问题和解决方案</button>
          </div>
        </div>
      </div>`;
    initHeroCanvas();
    document.querySelectorAll('.hero-suggestion').forEach(btn => {
      btn.addEventListener('click', () => {
        const q = btn.dataset.q;
        if (q) {
          this.textareaEl.value = q;
          this.textareaEl.dispatchEvent(new Event('input'));
          this.formEl.requestSubmit();
        }
      });
    });
    this.hideError();
  },

  loadMessages(messages) {
    this.messagesEl.innerHTML = '';
    messages.forEach(m => this.addMessage(m.role, m.content));
  }
};

// ═══════════════════════ History Panel ═══════════════════════

const HistoryPanel = {
  listEl: document.getElementById('history-list'),
  activeId: null,

  init() {
    document.getElementById('btn-refresh-sessions').addEventListener('click', () => this.load());
    this.load();
  },

  async load() {
    // Show skeleton while loading
    this.listEl.innerHTML = Array.from({length: 3}, () =>
      `<div class="skeleton-history"><div class="skeleton skeleton-bar w-40"></div><div class="skeleton skeleton-bar w-20"></div></div>`
    ).join('');
    try {
      const data = await API.getSessions();
      this.render(data.sessions || []);
    } catch (err) {
      this.render([]);
    }
  },

  render(sessions) {
    this.listEl.innerHTML = '';
    if (sessions.length === 0) {
      this.listEl.innerHTML = '<div class="history-empty">暂无记录</div>';
      return;
    }
    sessions.forEach(s => {
      const btn = document.createElement('button');
      btn.className = 'history-item' + (s.session_id === this.activeId ? ' active' : '');
      btn.setAttribute('role', 'listitem');
      btn.innerHTML = `
        <div class="history-item-preview">${escapeHtml(s.first_message)}</div>
        <div class="history-item-bottom">
          <div class="history-item-time">${formatDate(s.created_at)}</div>
          <span class="history-item-delete" role="button" tabindex="0" title="删除" aria-label="删除会话">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
          </span>
        </div>
      `;
      btn.addEventListener('click', (e) => {
        if (e.target.closest('.history-item-delete')) {
          e.stopPropagation();
          this.deleteSession(s.session_id);
          return;
        }
        this.selectSession(s.session_id);
      });
      this.listEl.appendChild(btn);
    });
  },

  async deleteSession(sessionId) {
    if (!await showConfirm('确定删除该会话？')) return;
    try {
      await API.deleteSession(sessionId);
      const currentSessionId = sessionStorage.getItem('ragmate_session_id');
      if (this.activeId === sessionId || currentSessionId === sessionId) {
        this.activeId = null;
        ChatPanel.clear();
      }
      this.load();
    } catch (err) {
      ChatPanel.showError('删除失败: ' + (err.message || '未知错误'));
    }
  },

  async selectSession(sessionId) {
    this.activeId = sessionId;
    sessionStorage.setItem('ragmate_session_id', sessionId);
    this.render(await this._fetchSessions());

    try {
      const data = await API.getHistory(sessionId);
      ChatPanel.loadMessages(data.messages || []);
    } catch (err) {
      ChatPanel.showError('加载历史记录失败');
    }

    // Switch to chat view
    document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
    document.querySelector('.sidebar-tab[data-tab="chat"]').classList.add('active');
    document.getElementById('view-chat').classList.remove('hidden');
    document.getElementById('view-documents').classList.add('hidden');
  },

  async _fetchSessions() {
    try {
      return (await API.getSessions()).sessions || [];
    } catch (err) {
      return [];
    }
  }
};

// ═══════════════════════ Documents Panel ═══════════════════════

const DocumentsPanel = {
  listEl: document.getElementById('doc-card-list'),
  emptyEl: document.getElementById('doc-empty'),
  uploadZone: document.getElementById('upload-zone'),
  fileInput: document.getElementById('file-input'),
  uploadError: document.getElementById('upload-error'),
  ingestBtn: document.getElementById('btn-ingest'),
  ingestBtnText: document.getElementById('btn-ingest-text'),
  ingestSpinner: document.getElementById('btn-ingest-spinner'),
  _ingestStateTimer: null,
  _ingestHandled: false,
  selectAllEl: document.getElementById('doc-select-all'),
  batchDeleteBtn: document.getElementById('btn-batch-delete'),
  batchDeleteText: document.getElementById('btn-batch-delete-text'),
  pollTimer: null,

  init() {
    this.fileInput.addEventListener('change', (e) => {
      const files = Array.from(e.target.files);
      e.target.value = '';
      files.forEach(file => this.upload(file));
    });

    this.uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); this.uploadZone.classList.add('drag-over'); });
    this.uploadZone.addEventListener('dragleave', () => this.uploadZone.classList.remove('drag-over'));
    this.uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadZone.classList.remove('drag-over');
      Array.from(e.dataTransfer.files).forEach(file => this.upload(file));
    });

    this.ingestBtn.addEventListener('click', () => this.startIngest());

    this.selectAllEl.addEventListener('change', () => {
      const checked = this.selectAllEl.checked;
      const listEl = document.getElementById('doc-card-list');
      if (!listEl) return;
      this._suppressCheckboxEvent = true;
      listEl.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.checked = checked; });
      this._suppressCheckboxEvent = false;
      this._updateBatchBar();
    });

    this.batchDeleteBtn.addEventListener('click', () => this.batchDelete());

    this._updateBatchBar();
    this.load();
  },

  _getSelectedFiles() {
    const listEl = document.getElementById('doc-card-list');
    if (!listEl) return [];
    return Array.from(listEl.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.dataset.filename);
  },

  _updateBatchBar() {
    const listEl = document.getElementById('doc-card-list');
    if (!listEl) return;
    const selectedCbs = Array.from(listEl.querySelectorAll('input[type="checkbox"]:checked'));
    const count = selectedCbs.length;
    const hasSelection = count > 0;
    const hasUnIngested = selectedCbs.some(cb => cb.dataset.status !== 'ingested');
    this.ingestBtn.removeAttribute('data-state');
    this.ingestBtn.disabled = !hasUnIngested;
    this.batchDeleteBtn.disabled = !hasSelection;
    this.batchDeleteText.textContent = hasSelection ? `删除 (${count})` : '删除';
    this.ingestBtnText.textContent = hasUnIngested ? `入库 (${selectedCbs.filter(cb => cb.dataset.status !== 'ingested').length})` : '入库';
    // 更新全选 checkbox 状态
    const allCbs = listEl.querySelectorAll('input[type="checkbox"]');
    this.selectAllEl.checked = allCbs.length > 0 && count === allCbs.length;
    this.selectAllEl.indeterminate = count > 0 && count < allCbs.length;
  },

  async batchDelete() {
    const files = this._getSelectedFiles();
    if (files.length === 0) return;
    if (!await showConfirm(`确定删除 ${files.length} 个文档？`)) return;

    const modal = showProgressModal('正在删除文档…', files.length);
    const items = { ok: [], fail: [] };

    for (let i = 0; i < files.length; i++) {
      if (modal.cancelled) break;
      modal.update(i, files[i]);
      try {
        await API.deleteDocument(files[i]);
        items.ok.push(files[i]);
      } catch (err) {
        items.fail.push({ name: files[i], error: err.message || '未知错误' });
      }
    }

    modal.done('删除完成', {
      ok: items.ok.map(f => ({ name: f })),
      fail: items.fail,
    });
    this.selectAllEl.checked = false;
    this._updateBatchBar();
    await this.load();
  },

  async load() {
    this._renderSkeleton();
    try {
      const data = await API.getDocuments();
      this.render(data.documents || []);
    } catch (err) {
      this.render([]);
    }
    this.checkIngestStatus();
  },

  async refreshList() {
    try {
      const data = await API.getDocuments();
      this.render(data.documents || []);
    } catch (err) {
      this.render([]);
    }
  },

  _renderSkeleton() {
    const skeleton = document.getElementById('doc-skeleton');
    if (!skeleton) return;
    skeleton.innerHTML = Array.from({length: 4}, () =>
      `<div class="skeleton-row">
        <div class="skeleton skeleton-check"></div>
        <div class="skeleton skeleton-bar w-40"></div>
        <div class="skeleton skeleton-bar w-10"></div>
        <div class="skeleton skeleton-bar w-16"></div>
        <div class="skeleton skeleton-bar w-20"></div>
      </div>`
    ).join('');
  },

  render(docs) {
    const listEl = document.getElementById('doc-card-list');
    if (!listEl) return;
    const skeleton = document.getElementById('doc-skeleton');
    if (skeleton) skeleton.innerHTML = '';

    // Clear old cards (keep skeleton div)
    listEl.querySelectorAll('.doc-card').forEach(el => el.remove());

    this.selectAllEl.checked = false;
    this.selectAllEl.indeterminate = false;
    this._updateBatchBar();

    if (docs.length === 0) {
      this.emptyEl.classList.remove('hidden');
      return;
    }
    this.emptyEl.classList.add('hidden');

    docs.sort((a, b) => (b.uploaded_at || '').localeCompare(a.uploaded_at || ''));
    docs.forEach(doc => {
      const ext = doc.filename.split('.').pop().toLowerCase();
      const iconClass = ['pdf','docx','xlsx','txt','md'].includes(ext) ? ext : 'txt';
      const card = document.createElement('div');
      card.className = 'doc-card';
      card.innerHTML = `
        <div class="doc-card-check"><input type="checkbox" data-filename="${escapeHtml(doc.filename)}" data-status="${doc.status}"></div>
        <div class="doc-card-icon ${iconClass}">${ext}</div>
        <div class="doc-card-info">
          <div class="doc-card-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
          <div class="doc-card-meta">
            <span>${formatFileSize(doc.size_bytes)}</span>
            <span class="badge badge-${doc.status}">${statusLabel(doc.status)}${doc.chunk_count ? ' · ' + doc.chunk_count + ' 片段' : ''}</span>
            <span>${formatDate(doc.uploaded_at)}</span>
          </div>
        </div>
        <div class="doc-card-actions">
          <button class="btn-delete" data-filename="${escapeHtml(doc.filename)}">删除</button>
        </div>
      `;
      card.querySelector('input[type="checkbox"]').addEventListener('change', () => {
        if (!this._suppressCheckboxEvent) this._updateBatchBar();
      });
      card.querySelector('.btn-delete').addEventListener('click', () => this.deleteDoc(doc.filename));
      listEl.appendChild(card);
    });
  },

  async upload(file) {
    const ext = file.name.toLowerCase().split('.').pop();
    const supported = ['pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt', 'md'];
    if (!supported.includes(ext)) {
      this.showUploadError('不支持的文件格式');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      this.showUploadError('文件不能超过 50MB');
      return;
    }

    this.uploadZone.classList.add('uploading');
    this.hideUploadError();

    try {
      await API.uploadDocument(file);
      await this.load();
    } catch (err) {
      // 409 文件已存在 → 确认替换
      if (err.message && err.message.includes('already exists')) {
        if (await showConfirm(`"${file.name}" 已存在，是否替换？`)) {
          try {
            await API.deleteDocument(file.name);
            await API.uploadDocument(file);
            await this.load();
          } catch (retryErr) {
            this.showUploadError(retryErr.message || '替换失败');
          }
        }
      } else {
        this.showUploadError(err.message || '上传失败');
      }
    } finally {
      this.uploadZone.classList.remove('uploading');
    }
  },

  async deleteDoc(filename) {
    if (!await showConfirm(`确定删除 "${filename}"？`)) return;
    try {
      await API.deleteDocument(filename);
      await this.load();
    } catch (err) {
      this.showUploadError('删除失败: ' + (err.message || '未知错误'));
    }
  },

  async startIngest() {
    const files = this._getSelectedFiles();
    this._ingestHandled = false;
    this.setIngestBtnState('running');
    this.batchDeleteBtn.disabled = true;

    try {
      const result = await API.startIngest(files.length > 0 ? files : null);
      if (result.status === 'already_running') {
        this.pollIngest();
      } else {
        // 打开进度弹窗，取消只是关闭弹窗，入库继续后台运行
        this._ingestModal = showProgressModal('正在入库…', files.length || 1);
        document.getElementById('prog-cancel').onclick = () => {
          this._ingestModal.close();
          this._ingestModal = null;
        };
        this.pollIngest();
      }
    } catch (err) {
      this.setIngestBtnState('failed');
    }
  },

  pollIngest() {
    if (this.pollTimer) clearInterval(this.pollTimer);
    this.pollTimer = setInterval(async () => {
      await this.checkIngestStatus();
    }, 2000);
  },

  async checkIngestStatus() {
    try {
      const status = await API.getIngestStatus();
      if (!status || status.status === 'idle') {
        this._ingestHandled = false;
        this.stopPolling();
        if (this._ingestModal) { this._ingestModal.close(); this._ingestModal = null; }
        this.setIngestBtnState('default');
        return;
      }
      // 已处理过的终态，不再重复触发按钮状态
      if (this._ingestHandled && status.status !== 'running') return;
      if (status.status === 'running') {
        const stage = status.stage || '';
        const stageLabels = {loading: '加载中', splitting: '切分中', encoding: '编码中', storing: '写入中'};
        const stageLabel = stageLabels[stage] || '处理中';
        const file = status.current_file || '';
        const progress = status.total || 0;
        const current = status.progress || 0;
        // 更新弹窗
        if (this._ingestModal && progress > 0) {
          this._ingestModal.update(current, `${stageLabel} — ${file}`);
        }
        this.setIngestBtnState('running');
      } else if (status.status === 'success') {
        this.stopPolling();
        if (this._ingestModal) {
          const docCount = status.document_count || 0;
          const chunkCount = status.chunk_count || 0;
          const skipped = (status.skipped || []).map(s => ({ name: s.filename, reason: s.reason }));
          const okItems = docCount > 0 ? Array.from({length: docCount}, (_, i) => ({ name: (status.filenames || [])[i] || `文档${i+1}` })) : [];
          this._ingestModal.done('入库完成', { ok: okItems, skipped, fail: [] },
            skipped.length === 0 ? `${docCount} 个文档，${chunkCount} 个片段` : '');
          this._ingestModal = null;
        }
        this._ingestHandled = true;
        this.setIngestBtnState('done');
        await this.refreshList();
      } else if (status.status === 'failed') {
        this.stopPolling();
        if (this._ingestModal) {
          this._ingestModal.done('入库失败', {
            ok: [],
            fail: [{ name: '入库任务', error: status.error || '未知错误' }],
          });
          this._ingestModal = null;
        }
        this._ingestHandled = true;
        this.setIngestBtnState('failed');
      }
    } catch (err) {
      this.stopPolling();
      if (this._ingestModal) { this._ingestModal.close(); this._ingestModal = null; }
      this.setIngestBtnState('default');
    }
  },

  stopPolling() {
    if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
  },

  setIngestBtnState(state) {
    if (this._ingestStateTimer) { clearTimeout(this._ingestStateTimer); this._ingestStateTimer = null; }
    this.ingestBtn.removeAttribute('data-state');
    if (state === 'running') {
      this.ingestBtn.disabled = true;
      this.ingestBtn.setAttribute('data-state', 'running');
      this.ingestSpinner.classList.remove('hidden');
      this.ingestBtnText.textContent = '入库中';
    } else if (state === 'done') {
      this.ingestBtn.disabled = true;
      this.ingestBtn.setAttribute('data-state', 'done');
      this.ingestSpinner.classList.add('hidden');
      this.ingestBtnText.textContent = '完成';
      this._ingestStateTimer = setTimeout(() => this.setIngestBtnState('default'), 2000);
    } else if (state === 'failed') {
      this.ingestBtn.disabled = true;
      this.ingestBtn.setAttribute('data-state', 'failed');
      this.ingestSpinner.classList.add('hidden');
      this.ingestBtnText.textContent = '失败';
      this._ingestStateTimer = setTimeout(() => this.setIngestBtnState('default'), 2000);
    } else {
      this.ingestSpinner.classList.add('hidden');
      this._updateBatchBar();
    }
  },

  showUploadError(msg) {
    this.uploadError.textContent = msg;
    this.uploadError.classList.remove('hidden');
  },

  hideUploadError() {
    this.uploadError.classList.add('hidden');
  }
};

// ── Utilities ──

function statusLabel(s) {
  const map = { uploaded: '未入库', ingesting: '入库中', ingested: '已入库', failed: '失败' };
  return map[s] || s;
}

// ═══════════════════════ App Init ═══════════════════════

document.addEventListener('DOMContentLoaded', () => {
  try { ChatPanel.init(); } catch (e) { console.error('ChatPanel init failed:', e); }
  try { HistoryPanel.init(); } catch (e) { console.error('HistoryPanel init failed:', e); }
  try { DocumentsPanel.init(); } catch (e) { console.error('DocumentsPanel init failed:', e); }

  document.querySelectorAll('.sidebar-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const tabId = tab.dataset.tab;
      document.getElementById('view-chat').classList.toggle('hidden', tabId !== 'chat');
      document.getElementById('view-documents').classList.toggle('hidden', tabId !== 'documents');

      if (tabId === 'documents') {
        DocumentsPanel.load();
        DocumentsPanel.checkIngestStatus();
      }
      if (tabId === 'chat') {
        ChatPanel.textareaEl.focus();
      }
    });
  });

  window.addEventListener('beforeunload', () => DocumentsPanel.stopPolling());

  // Hero empty state
  initHeroCanvas();
  document.querySelectorAll('.hero-suggestion').forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.dataset.q;
      if (q) {
        ChatPanel.textareaEl.value = q;
        ChatPanel.textareaEl.dispatchEvent(new Event('input'));
        ChatPanel.formEl.requestSubmit();
      }
    });
  });
});

// ═══════════════════════ Hero Canvas: 知识节点网络 ═══════════════════════

let _heroCleanup = null;  // 追踪当前 hero 动画的清理函数

function initHeroCanvas() {
  // 清理上一次 hero 动画的所有资源
  if (_heroCleanup) { _heroCleanup(); _heroCleanup = null; }

  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const hero = canvas.parentElement;
  let w, h, nodes = [], animId;

  function resize() {
    w = canvas.width = hero.offsetWidth;
    h = canvas.height = hero.offsetHeight;
  }

  function createNodes() {
    nodes = [];
    const count = Math.floor((w * h) / 12000);
    for (let i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.8,
        pulse: Math.random() * Math.PI * 2,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    const t = Date.now() * 0.001;

    // Draw connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          const alpha = (1 - dist / 120) * 0.12;
          ctx.strokeStyle = `rgba(240,180,41,${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (const node of nodes) {
      node.x += node.vx;
      node.y += node.vy;
      if (node.x < 0 || node.x > w) node.vx *= -1;
      if (node.y < 0 || node.y > h) node.vy *= -1;

      const glow = Math.sin(t * 2 + node.pulse) * 0.3 + 0.7;
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(240,180,41,${glow * 0.5})`;
      ctx.fill();

      // Glow halo
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r * 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(240,180,41,${glow * 0.06})`;
      ctx.fill();
    }

    animId = requestAnimationFrame(draw);
  }

  resize();
  createNodes();
  draw();

  const onResize = () => { resize(); createNodes(); };
  window.addEventListener('resize', onResize);

  // Hide hero when first message is sent
  const observer = new MutationObserver(() => {
    const heroEl = document.getElementById('hero-empty');
    const msgs = document.querySelectorAll('.msg');
    if (heroEl && msgs.length > 0) {
      heroEl.style.display = 'none';
      cancelAnimationFrame(animId);
    }
  });
  const msgContainer = document.getElementById('chat-messages');
  if (msgContainer) observer.observe(msgContainer, { childList: true });

  // 注册清理函数：下次 initHeroCanvas 调用时会执行
  _heroCleanup = () => {
    cancelAnimationFrame(animId);
    window.removeEventListener('resize', onResize);
    observer.disconnect();
  };
}
