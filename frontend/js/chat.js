import { API } from './api.js';
import { escapeHtml, renderAssistantMarkdown } from './utils.js';
import { getSessionId, newSession } from './session.js';
import { on, emit } from './event-bus.js';

const AI_ICON = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>`;
const USER_ICON = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M6 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/></svg>`;

const HERO_HTML = `
  <div id="hero-empty" class="hero-empty empty-state">
    <div class="hero-content">
      <h1 class="hero-title">你的<span class="accent">知识库</span> AI 助手</h1>
      <p class="hero-subtitle">上传文档，开启智能问答</p>
      <div class="hero-steps" id="hero-steps">
        <div class="hero-step" data-step="1"><span class="hero-step-num">1</span>上传文档</div>
        <div class="hero-step-line"></div>
        <div class="hero-step" data-step="2"><span class="hero-step-num">2</span>智能提问</div>
        <div class="hero-step-line"></div>
        <div class="hero-step" data-step="3"><span class="hero-step-num">3</span>精准溯源</div>
      </div>
      <div class="hero-cards" id="hero-cards"></div>
    </div>
  </div>`;

export const ChatPanel = {
  messagesEl:  document.getElementById('chat-messages'),
  formEl:      document.getElementById('chat-form'),
  textareaEl:  document.getElementById('chat-textarea'),
  sendBtn:     document.getElementById('btn-send'),
  errorEl:     document.getElementById('chat-error'),
  loading:     false,
  _lastUserText: '',

  init() {
    this.formEl.addEventListener('submit', (e) => { e.preventDefault(); this.send(); });
    document.getElementById('btn-new-chat').addEventListener('click', () => {
      this.clear();
      emit('history:refresh');
    });

    this.textareaEl.addEventListener('input', () => {
      this.textareaEl.style.height = 'auto';
      const fontSize = parseFloat(getComputedStyle(this.textareaEl).fontSize);
      const lineHeight = parseFloat(getComputedStyle(this.textareaEl).lineHeight) || fontSize * 1.5;
      this.textareaEl.style.height = Math.min(this.textareaEl.scrollHeight, lineHeight * 5) + 'px';
    });
    this.textareaEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.send(); }
    });

    on('chat:show-error',    (msg) => this.showError(msg));
    on('chat:load-messages', (msgs) => this.loadMessages(msgs));
    on('view:switch',        (tab) => { if (tab === 'chat') this.textareaEl.focus(); });
  },

  addMessage(role, text) {
    const empty = this.messagesEl.querySelector('.empty-state');
    if (empty) empty.remove();

    if (this.messagesEl.querySelector('.msg')) {
      const divider = document.createElement('div');
      divider.className = 'msg-divider';
      this.messagesEl.appendChild(divider);
    }

    const div = document.createElement('div');
    div.className = 'msg ' + (role === 'user' ? 'msg-user' : 'msg-assistant');

    if (role === 'assistant') {
      div.innerHTML = `<div class="msg-body"><div class="msg-role"><span class="msg-role-icon ai-icon">${AI_ICON}</span>RagMate</div><div class="msg-content">${renderAssistantMarkdown(text)}</div></div>`;
    } else {
      div.innerHTML = `<div class="msg-body"><div class="msg-role"><span class="msg-role-icon user-icon">${USER_ICON}</span>You</div><div class="msg-content">${DOMPurify.sanitize(text)}</div></div>`;
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
        const lastMsg = this.messagesEl.querySelector('.msg-assistant:last-of-type');
        if (lastMsg) {
          const prev = lastMsg.previousElementSibling;
          if (prev && prev.classList.contains('msg-divider')) prev.remove();
          lastMsg.remove();
        }
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

  hideError() { this.errorEl.classList.add('hidden'); },

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
      text, sid,
      (token) => { fullText += token; this.appendStreamToken(streamDiv, fullText); },
      (doneData) => {
        this.finalizeStreamMessage(streamDiv, fullText, doneData);
        sessionStorage.setItem('ragmate_session_id', doneData.session_id);
        this.setDisabled(false);
        this.textareaEl.focus();
        emit('history:refresh');
      },
      (errMsg) => {
        this.finalizeStreamMessage(streamDiv, fullText || '');
        this.showError(errMsg);
        this.setDisabled(false);
        this.textareaEl.focus();
        emit('history:refresh');
      },
      replaceLast,
    );
  },

  startStreamMessage() {
    const empty = this.messagesEl.querySelector('.empty-state');
    if (empty) empty.remove();

    if (this.messagesEl.querySelector('.msg')) {
      const divider = document.createElement('div');
      divider.className = 'msg-divider';
      this.messagesEl.appendChild(divider);
    }

    const div = document.createElement('div');
    div.className = 'msg msg-assistant';
    div.innerHTML = `<div class="msg-body"><div class="msg-role"><span class="msg-role-icon ai-icon">${AI_ICON}</span>RagMate</div><div class="msg-content"><div class="msg-loading typing-dots">检索与生成中<span>.</span><span>.</span><span>.</span></div></div></div>`;
    this.messagesEl.appendChild(div);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    return div;
  },

  appendStreamToken(div, fullText) {
    const content = div.querySelector('.msg-content');
    if (div._streamDone) return;
    div._latestText = fullText;
    if (!div._streamTimer) {
      div._streamTimer = setTimeout(() => {
        div._streamTimer = null;
        if (div._streamDone) return;
        content.innerHTML = DOMPurify.sanitize(marked.parse(div._latestText));
        const el = this.messagesEl;
        const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
        if (isNearBottom) el.scrollTop = el.scrollHeight;
      }, 80);
    }
  },

  finalizeStreamMessage(div, fullText, doneData = {}) {
    div._streamDone = true;
    const content = div.querySelector('.msg-content');
    content.innerHTML = renderAssistantMarkdown(fullText || '没有收到回复');

    if (doneData.confidence) {
      const level = doneData.confidence.level;
      const label = { high: '高置信', medium: '中置信', low: '低置信' }[level] || level;
      const badge = document.createElement('div');
      badge.className = `confidence-badge confidence-${level}`;
      badge.textContent = label;
      badge.title = `最高相关度: ${doneData.confidence.score}, 引用片段: ${doneData.confidence.chunks}`;
      content.appendChild(badge);
    }

    if (doneData.unsupported_claims && doneData.unsupported_claims.length > 0) {
      const warn = document.createElement('div');
      warn.className = 'faithfulness-warning';
      warn.innerHTML = `⚠ 以下声明可能缺乏文献支撑：<ul>${doneData.unsupported_claims.map(c => `<li>${escapeHtml(c.claim)}</li>`).join('')}</ul>`;
      content.appendChild(warn);
    }

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

    const msgBody = div.querySelector('.msg-body') || content.parentElement;
    const copyBtn = msgBody.querySelector('[data-action="copy"]');
    const regenBtn = msgBody.querySelector('[data-action="regenerate"]');

    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(fullText || '').then(() => {
          copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> 已复制`;
          setTimeout(() => {
            copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> 复制`;
          }, 1500);
        });
      });
    }

    if (regenBtn) {
      regenBtn.addEventListener('click', () => {
        if (this.loading) return;
        let userMsg = null;
        let el = div.previousElementSibling;
        if (el && el.classList.contains('msg-divider')) el = el.previousElementSibling;
        if (el && el.classList.contains('msg-user')) userMsg = el;

        const userText = userMsg
          ? userMsg.querySelector('.msg-content')?.textContent || ''
          : (this._lastUserText || '');
        if (!userText) return;

        const prevSibling = div.previousElementSibling;
        if (prevSibling && prevSibling.classList.contains('msg-divider')) prevSibling.remove();
        div.remove();
        if (userMsg) {
          const userDivider = userMsg.previousElementSibling;
          if (userDivider && userDivider.classList.contains('msg-divider')) userDivider.remove();
          userMsg.remove();
        }
        this.textareaEl.value = userText;
        this.send(true);
      });
    }

    content.querySelectorAll('.btn-copy-code').forEach(btn => {
      btn.addEventListener('click', () => {
        const code = decodeURIComponent(escape(atob(btn.dataset.codeB64)));
        navigator.clipboard.writeText(code).then(() => {
          btn.textContent = '已复制';
          btn.classList.add('copied');
          setTimeout(() => { btn.textContent = '复制'; btn.classList.remove('copied'); }, 1500);
        });
      });
    });
  },

  clear() {
    newSession();
    this.messagesEl.innerHTML = HERO_HTML;
    this.updateHeroState();
    this.hideError();
  },

  _bindHeroCards() {
    document.querySelectorAll('.hero-card[data-q]').forEach(btn => {
      btn.addEventListener('click', () => {
        const q = btn.dataset.q;
        if (q) {
          this.textareaEl.value = q;
          this.textareaEl.dispatchEvent(new Event('input'));
          this.formEl.requestSubmit();
        }
      });
    });
    document.querySelectorAll('.hero-card[data-action="upload"]').forEach(btn => {
      btn.addEventListener('click', () => emit('view:switch', 'documents'));
    });
  },

  async updateHeroState() {
    const cardsEl = document.getElementById('hero-cards');
    const stepsEl = document.getElementById('hero-steps');
    if (!cardsEl || !stepsEl) return;

    const checkIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
    const uploadIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>`;
    const summaryIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`;
    const linkIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>`;
    const conflictIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;

    let hasDocs = false;
    try {
      const data = await API.getDocuments();
      hasDocs = (data.documents || []).some(d => d.status === 'ingested');
    } catch (_) {}

    const steps = stepsEl.querySelectorAll('.hero-step');
    const lines = stepsEl.querySelectorAll('.hero-step-line');

    if (hasDocs) {
      steps[0].className = 'hero-step done';
      steps[0].querySelector('.hero-step-num').innerHTML = checkIcon;
      lines[0].className = 'hero-step-line done';
      cardsEl.innerHTML = `
        <button class="hero-card" data-q="帮我总结这个知识库的核心内容">
          <div class="hero-card-icon">${summaryIcon}</div>
          <div class="hero-card-title">总结核心内容</div>
        </button>
        <button class="hero-card" data-q="这些文档之间有什么关联？">
          <div class="hero-card-icon">${linkIcon}</div>
          <div class="hero-card-title">分析文档关联</div>
        </button>
        <button class="hero-card" data-q="有没有互相矛盾的说法？">
          <div class="hero-card-icon">${conflictIcon}</div>
          <div class="hero-card-title">检测矛盾说法</div>
        </button>`;
    } else {
      steps[0].className = 'hero-step';
      steps[0].querySelector('.hero-step-num').textContent = '1';
      lines[0].className = 'hero-step-line';
      cardsEl.innerHTML = `
        <button class="hero-card" data-action="upload">
          <div class="hero-card-icon">${uploadIcon}</div>
          <div class="hero-card-title">上传文档开始</div>
        </button>`;
    }
    this._bindHeroCards();
  },

  loadMessages(messages) {
    this.messagesEl.innerHTML = '';
    messages.forEach(m => this.addMessage(m.role, m.content));
  }
};
