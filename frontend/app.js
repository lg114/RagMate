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

  async chatStream(message, sessionId, onToken, onDone, onError) {
    try {
      const res = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
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
            if (data.done) { onDone(data.session_id); return; }
            if (data.error) { onError(data.error); return; }
            if (data.token) onToken(data.token);
          } catch (e) {
            // 忽略无法解析的行
          }
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

  startIngest() {
    return this.request('POST', '/ingest');
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
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function truncate(str, len) {
  return str.length > len ? str.slice(0, len - 3) + '...' : str;
}

function normalizeCitations(text) {
  const citationRe = /【([^】]+\.(?:pdf|docx?|xlsx?|xls|txt|md))】/gi;
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
    cleaned = cleaned.split(`【${source}】`).join('');
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
  return DOMPurify.sanitize(marked.parse(withSourceChips));
}

// ── Session ID ──
function getSessionId() {
  let sid = sessionStorage.getItem('ragmate_session_id');
  if (!sid) { sid = crypto.randomUUID(); sessionStorage.setItem('ragmate_session_id', sid); }
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
      this.textareaEl.style.height = Math.min(this.textareaEl.scrollHeight, 1.5 * 16 * 5) + 'px';
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
        this.send();
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

  async send() {
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
      (sessionId) => {
        this.finalizeStreamMessage(streamDiv, fullText);
        sessionStorage.setItem('ragmate_session_id', sessionId);
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
    // 节流：用 requestAnimationFrame 合并多次 token 更新，避免每 token 都重新 parse 全文
    if (!div._rafPending) {
      div._rafPending = true;
      requestAnimationFrame(() => {
        content.innerHTML = DOMPurify.sanitize(marked.parse(fullText));
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
        div._rafPending = false;
      });
    }
  },

  finalizeStreamMessage(div, fullText) {
    const content = div.querySelector('.msg-content');
    content.innerHTML = renderAssistantMarkdown(fullText || '没有收到回复');

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
          this.send();
        }
      });
    }
  },

  clear() {
    newSession();
    this.messagesEl.innerHTML = '<div class="empty-state"><div class="empty-label">提出问题，从知识库中检索答案</div></div>';
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
      const div = document.createElement('div');
      div.className = 'history-item' + (s.session_id === this.activeId ? ' active' : '');
      div.innerHTML = `
        <div class="history-item-content">
          <div class="history-item-preview">${escapeHtml(s.first_message)}</div>
          <div class="history-item-bottom">
            <div class="history-item-time">${formatDate(s.created_at)}</div>
            <button class="history-item-delete" title="删除">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
            </button>
          </div>
        </div>
      `;
      div.querySelector('.history-item-content').addEventListener('click', () => this.selectSession(s.session_id));
      div.querySelector('.history-item-delete').addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteSession(s.session_id);
      });
      this.listEl.appendChild(div);
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
  tbodyEl: document.getElementById('doc-table-body'),
  emptyEl: document.getElementById('doc-empty'),
  uploadZone: document.getElementById('upload-zone'),
  fileInput: document.getElementById('file-input'),
  uploadError: document.getElementById('upload-error'),
  ingestBtn: document.getElementById('btn-ingest'),
  ingestBtnText: document.getElementById('btn-ingest-text'),
  ingestSpinner: document.getElementById('btn-ingest-spinner'),
  statusBar: document.getElementById('ingest-status-bar'),
  statusText: document.getElementById('ingest-status-text'),
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
    this.resetIngestBtn();
    this.load();
  },

  async load() {
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

  render(docs) {
    this.tbodyEl.innerHTML = '';
    if (docs.length === 0) {
      this.emptyEl.classList.remove('hidden');
      return;
    }
    this.emptyEl.classList.add('hidden');

    docs.sort((a, b) => (b.uploaded_at || '').localeCompare(a.uploaded_at || ''));
    docs.forEach(doc => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td title="${escapeHtml(doc.filename)}">${escapeHtml(truncate(doc.filename, 40))}</td>
        <td>${formatFileSize(doc.size_bytes)}</td>
        <td><span class="badge badge-${doc.status}">${statusLabel(doc.status)}${doc.chunk_count ? ' · ' + doc.chunk_count + ' 个片段' : ''}</span></td>
        <td>${formatDate(doc.uploaded_at)}</td>
        <td><button class="btn-delete" data-filename="${escapeHtml(doc.filename)}">删除</button></td>
      `;
      tr.querySelector('.btn-delete').addEventListener('click', () => this.deleteDoc(doc.filename));
      this.tbodyEl.appendChild(tr);
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
      this.showUploadError(err.message || '上传失败');
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
    this.ingestBtn.disabled = true;
    this.ingestSpinner.classList.remove('hidden');
    this.ingestBtnText.textContent = '入库中…';

    try {
      const result = await API.startIngest();
      if (result.status === 'already_running') {
        this.showStatus('入库已在运行中…', 'running');
      } else {
        this.showStatus('入库已启动…', 'running');
      }
      this.pollIngest();
    } catch (err) {
      this.showStatus('入库启动失败: ' + (err.message || ''), 'failed');
      this.resetIngestBtn();
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
        this.stopPolling();
        this.hideStatus();
        this.resetIngestBtn();
        return;
      }
      if (status.status === 'running') {
        this.showStatus('入库中…', 'running');
        this.ingestBtn.disabled = true;
      } else if (status.status === 'success') {
        this.stopPolling();
        this.showStatus(
          `入库完成 — ${status.document_count} 个文档，${status.chunk_count} 个片段`,
          'success'
        );
        this.resetIngestBtn();
        await this.refreshList();
      } else if (status.status === 'failed') {
        this.stopPolling();
        this.showStatus('入库失败: ' + (status.error || '未知错误'), 'failed');
        this.resetIngestBtn();
      }
    } catch (err) {
      this.stopPolling();
      this.resetIngestBtn();
    }
  },

  stopPolling() {
    if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
  },

  showStatus(msg, type) {
    this.statusBar.classList.remove('hidden', 'running', 'failed');
    if (type) this.statusBar.classList.add(type);
    this.statusText.textContent = msg;
  },

  hideStatus() {
    this.statusBar.classList.add('hidden');
  },

  resetIngestBtn() {
    this.ingestBtn.disabled = false;
    this.ingestSpinner.classList.add('hidden');
    this.ingestBtnText.textContent = '入库';
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
  ChatPanel.init();
  HistoryPanel.init();
  DocumentsPanel.init();

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
});
