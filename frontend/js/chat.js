import { API } from './api.js';
import { getSessionId, newSession } from './session.js';
import { on, emit } from './event-bus.js';
import { debugEnabled } from './prefs.js';
import { SourcePanel } from './sources.js';
import {
  GROUNDING_COPY,
  confidenceLabel,
  dropReasonLabel,
  extractSourceLine,
  mergeCitations,
  normalizeCitations,
  renderMarkdown,
  bindCodeCopy,
  linkifyCitations,
  splitAnswerFrame,
} from './utils.js';

const PROMPTS = [
  { q: '帮我总结这个知识库的核心内容', title: '总结核心内容', detail: '先给一个可以核对的概览' },
  { q: '这些文档之间有什么关联？', title: '找出文档关联', detail: '看不同来源是否在讲同一件事' },
  { q: '有没有互相矛盾的说法？', title: '检查矛盾说法', detail: '把冲突的段落分开列出来' },
];

export const ChatPanel = {
  messagesEl: null,
  scrollEl: null,
  formEl: null,
  textareaEl: null,
  sendBtn: null,
  stopBtn: null,
  errorEl: null,
  statusEl: null,
  jumpBtn: null,
  loading: false,
  abortController: null,
  _lastUserText: '',

  init() {
    this.messagesEl = document.getElementById('chat-messages');
    this.scrollEl = document.getElementById('chat-scroll');
    this.formEl = document.getElementById('chat-form');
    this.textareaEl = document.getElementById('chat-textarea');
    this.sendBtn = document.getElementById('btn-send');
    this.stopBtn = document.getElementById('btn-stop');
    this.errorEl = document.getElementById('chat-error');
    this.statusEl = document.getElementById('chat-status');
    this.jumpBtn = document.getElementById('btn-jump-latest');

    this.formEl.addEventListener('submit', (event) => {
      event.preventDefault();
      this.send();
    });
    document.getElementById('btn-new-chat').addEventListener('click', () => this.startNew());
    this.stopBtn.addEventListener('click', () => this.stop());
    this.jumpBtn.addEventListener('click', () => this.scrollToBottom(true));
    this.textareaEl.addEventListener('input', () => this.autosize());
    this.textareaEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        this.send();
      }
    });
    this.scrollEl.addEventListener('scroll', () => this.updateJump());

    on('chat:load-messages', (payload) => this.loadMessages(payload.messages || [], payload.title));
    on('library:changed', () => this.updateHeroState());
    on('view:switch', (tab) => { if (tab === 'chat') this.textareaEl.focus(); });

    this.showHero();
    this.setStatus('');
  },

  onShow() {
    this.textareaEl?.focus();
  },

  startNew() {
    this.stop(true);
    newSession();
    emit('history:active', null);
    emit('shell:context', '新对话');
    this.clear();
    emit('history:refresh');
  },

  setStatus(text) {
    if (this.statusEl) this.statusEl.textContent = text || '';
  },

  autosize() {
    const el = this.textareaEl;
    el.style.height = 'auto';
    const line = parseFloat(getComputedStyle(el).lineHeight) || 24;
    el.style.height = `${Math.min(el.scrollHeight, line * 6)}px`;
    const count = document.getElementById('composer-count');
    if (count) {
      const len = el.value.length;
      count.hidden = len < 8000;
      count.textContent = `${len} / 10000`;
    }
  },

  nearBottom() {
    const el = this.scrollEl;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  },

  scrollToBottom(force) {
    if (force || this.nearBottom()) this.scrollEl.scrollTop = this.scrollEl.scrollHeight;
    this.updateJump();
  },

  updateJump() {
    const hasMessages = !!this.messagesEl.querySelector('.msg');
    this.jumpBtn.hidden = !hasMessages || this.nearBottom();
  },

  setBusy(busy) {
    this.loading = busy;
    this.sendBtn.hidden = busy;
    this.stopBtn.hidden = !busy;
    this.textareaEl.disabled = busy;
    this.formEl.setAttribute('aria-busy', busy ? 'true' : 'false');
  },

  showError(msg) {
    this.errorEl.replaceChildren();
    const text = document.createElement('span');
    text.textContent = msg;
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'btn btn-ghost';
    retry.textContent = '重试';
    retry.addEventListener('click', () => {
      if (!this._lastUserText || this.loading) return;
      this.hideError();
      this.removeLastTurn();
      this.textareaEl.value = this._lastUserText;
      this.send(true);
    });
    this.errorEl.append(text, retry);
    this.errorEl.hidden = false;
  },

  hideError() {
    this.errorEl.hidden = true;
    this.errorEl.replaceChildren();
  },

  removeLastTurn() {
    const assistant = this.messagesEl.querySelector('.msg-assistant:last-of-type');
    const user = assistant ? previousUser(assistant) : this.messagesEl.querySelector('.msg-user:last-of-type');
    assistant?.remove();
    user?.remove();
    if (!this.messagesEl.querySelector('.msg')) this.showHero();
  },

  showHero() {
    this.messagesEl.replaceChildren(buildHero());
    this.updateHeroState();
    this.updateJump();
  },

  clear() {
    this.hideError();
    SourcePanel.close();
    this.showHero();
  },

  async updateHeroState() {
    const actions = document.getElementById('hero-actions');
    const lead = document.getElementById('hero-lead');
    if (!actions || !lead) return;
    let ingested = 0;
    let total = 0;
    try {
      const data = await API.getDocuments();
      const docs = data.documents || [];
      total = docs.length;
      ingested = docs.filter((doc) => doc.status === 'ingested').length;
    } catch { /* leave the upload prompt in place */ }

    actions.replaceChildren();
    if (ingested > 0) {
      lead.textContent = `知识库里有 ${ingested} 篇已入库文档。提问后，回答会带上置信度和可核对的来源。`;
      PROMPTS.forEach((item) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'prompt-card';
        const title = document.createElement('span');
        title.className = 'prompt-title';
        title.textContent = item.title;
        const detail = document.createElement('span');
        detail.className = 'prompt-detail';
        detail.textContent = item.detail;
        btn.append(title, detail);
        btn.addEventListener('click', () => {
          this.textareaEl.value = item.q;
          this.autosize();
          this.formEl.requestSubmit();
        });
        actions.append(btn);
      });
    } else {
      lead.textContent = total
        ? '文档已经上传，入库完成之后就可以按来源提问。'
        : '先把资料放进文档库。入库完成后，回答会标明出处。';
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'prompt-card prompt-card-primary';
      const title = document.createElement('span');
      title.className = 'prompt-title';
      title.textContent = total ? '去文档库入库' : '上传文档';
      const detail = document.createElement('span');
      detail.className = 'prompt-detail';
      detail.textContent = '支持 PDF、Word、Excel、TXT、Markdown';
      btn.append(title, detail);
      btn.addEventListener('click', () => emit('view:switch', 'documents'));
      actions.append(btn);
    }
  },

  addUserMessage(text) {
    this.messagesEl.querySelector('.hero')?.remove();
    const article = document.createElement('article');
    article.className = 'msg msg-user';
    const content = document.createElement('div');
    content.className = 'msg-content';
    content.textContent = text;
    article.append(content);
    this.messagesEl.append(article);
    this.scrollToBottom(true);
    return article;
  },

  startStreamMessage() {
    this.messagesEl.querySelector('.hero')?.remove();
    const article = document.createElement('article');
    article.className = 'msg msg-assistant is-streaming';
    article.setAttribute('aria-busy', 'true');
    article.append(this._kicker('正在检索资料'));
    const content = document.createElement('div');
    content.className = 'msg-content prose';
    article.append(content);
    this.messagesEl.append(article);
    this.scrollToBottom(true);
    return article;
  },

  _kicker(state) {
    const row = document.createElement('div');
    row.className = 'msg-kicker';
    const mark = document.createElement('span');
    mark.className = 'avatar';
    mark.setAttribute('aria-hidden', 'true');
    mark.textContent = 'R';
    const name = document.createElement('span');
    name.className = 'msg-author';
    name.textContent = 'RagMate';
    const live = document.createElement('span');
    live.className = 'stream-state';
    live.textContent = state || '';
    row.append(mark, name, live);
    return row;
  },

  appendStreamToken(article, fullText) {
    if (article._streamDone) return;
    article._latestText = fullText;
    const state = article.querySelector('.stream-state');
    if (state) state.textContent = '正在生成';
    if (article._streamTimer) return;
    article._streamTimer = setTimeout(() => {
      article._streamTimer = null;
      if (article._streamDone) return;
      const content = article.querySelector('.msg-content');
      content.innerHTML = renderMarkdown(article._latestText || '');
      content.classList.add('is-streaming');
      this.scrollToBottom(false);
    }, 50);
  },

  finalizeStreamMessage(article, fullText, doneData = {}, stopped = false) {
    article._streamDone = true;
    if (article._streamTimer) {
      clearTimeout(article._streamTimer);
      article._streamTimer = null;
    }
    article.classList.remove('is-streaming');
    article.removeAttribute('aria-busy');
    const text = (!stopped && doneData.response) ? doneData.response : (fullText || '');
    article._finalText = text;
    article._retrievalDebug = doneData.retrieval_debug || [];
    const frame = splitAnswerFrame(text, doneData.grounding);
    if (stopped && !text) frame.stopped = true;

    const state = article.querySelector('.stream-state');
    if (state) state.remove();
    if (doneData.confidence && !frame.refused) {
      article.querySelector('.msg-kicker')?.append(confidencePill(doneData.confidence));
    }

    const content = article.querySelector('.msg-content');
    content.classList.remove('is-streaming');
    article.querySelectorAll('.msg-banner, .cite-list, .retrieval-debug, .msg-actions').forEach((el) => el.remove());

    if (frame.refused) {
      article.classList.add('is-refused');
      content.replaceChildren();
      article.insertBefore(banner('refused', '无法可靠回答', GROUNDING_COPY.refusal), content);
    } else if (frame.stopped && !frame.body) {
      content.replaceChildren();
      const note = document.createElement('p');
      note.className = 'stopped-note';
      note.textContent = '已停止生成';
      content.append(note);
    } else {
      if (frame.hedge) article.insertBefore(banner('hedged', '低置信', GROUNDING_COPY.hedge), content);
      const hasSourceLine = /^\s*数据来源[:：]/m.test(frame.body);
      const normalized = hasSourceLine ? frame.body : normalizeCitations(frame.body);
      const extracted = extractSourceLine(normalized);
      content.innerHTML = extracted.body ? renderMarkdown(extracted.body) : '';
      bindCodeCopy(content);
      linkifyCitations(content, (citation, trigger) => this.openSource(citation, trigger, article));
      if (!extracted.body && !frame.hedge) {
        const note = document.createElement('p');
        note.className = 'stopped-note';
        note.textContent = '没有收到回复';
        content.append(note);
      }
      const cites = doneData.citations?.length
        ? mergeCitations(doneData.citations, [])
        : extracted.citations;
      if (cites.length && !frame.refused) article.append(this._citationList(cites, article));
    }

    if (doneData.unsupported_claims?.length && !frame.refused) {
      article.append(claimsBanner(doneData.unsupported_claims));
    }
    if (doneData.retrieval_debug?.length) {
      article.append(renderRetrievalDebug(doneData.retrieval_debug, (citation, trigger) => {
        this.openSource(citation, trigger, article);
      }));
    }
    article.append(this._actions(article, doneData.allowRegen !== false && !frame.refused && !frame.stopped));
    this.scrollToBottom(false);

    if (doneData.quiet) return;
    if (frame.refused) this.setStatus('没有足够资料，已拒答');
    else if (frame.hedge) this.setStatus('低置信回答已完成');
    else if (stopped) this.setStatus('已停止生成');
    else this.setStatus('回答已完成');
  },

  _citationList(citations, article) {
    const list = document.createElement('div');
    list.className = 'cite-list';
    const label = document.createElement('span');
    label.className = 'cite-label';
    label.textContent = '来源';
    list.append(label);
    citations.forEach((citation, index) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'cite-btn';
      const indexEl = document.createElement('span');
      indexEl.className = 'cite-index';
      indexEl.textContent = String(index + 1);
      const name = document.createElement('span');
      name.className = 'cite-name';
      name.textContent = citation.source;
      btn.append(indexEl, name);
      if (citation.page) {
        const page = document.createElement('span');
        page.className = 'cite-page';
        page.textContent = `第 ${citation.page} 页`;
        btn.append(page);
      }
      btn.addEventListener('click', () => this.openSource(citation, btn, article));
      list.append(btn);
    });
    return list;
  },

  _actions(article, allowRegen) {
    const row = document.createElement('div');
    row.className = 'msg-actions';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'btn btn-ghost';
    copy.textContent = '复制';
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(article._finalText || '');
        copy.textContent = '已复制';
        setTimeout(() => { copy.textContent = '复制'; }, 1400);
      } catch {
        copy.textContent = '复制失败';
      }
    });
    row.append(copy);
    if (allowRegen) {
      const regen = document.createElement('button');
      regen.type = 'button';
      regen.className = 'btn btn-ghost';
      regen.textContent = '重新生成';
      regen.addEventListener('click', () => {
        if (this.loading) return;
        const user = previousUser(article);
        const userText = user?.querySelector('.msg-content')?.textContent || this._lastUserText;
        if (!userText) return;
        article.remove();
        user?.remove();
        this.textareaEl.value = userText;
        this.send(true);
      });
      row.append(regen);
    }
    return row;
  },

  openSource(citation, trigger, article) {
    const previews = [];
    (article?._retrievalDebug || []).forEach((block) => {
      (block.chunks || []).forEach((chunk) => {
        if (chunk.source !== citation.source) return;
        if (citation.page != null && chunk.page != null && chunk.page !== citation.page) return;
        if (chunk.preview) previews.push(chunk);
      });
    });
    SourcePanel.open(citation, trigger, { previews: previews.slice(0, 3) });
  },

  async send(replaceLast) {
    const text = this.textareaEl.value.trim();
    if (!text || this.loading) return;
    this.textareaEl.value = '';
    this.autosize();
    this.hideError();
    this._lastUserText = text;
    this.addUserMessage(text);
    this.setBusy(true);
    this.setStatus('正在检索资料');
    emit('shell:context', text.length > 42 ? `${text.slice(0, 42)}…` : text);

    const article = this.startStreamMessage();
    let fullText = '';
    const controller = new AbortController();
    this.abortController = controller;
    const sid = getSessionId();
    emit('history:active', sid);

    await API.chatStream(text, sid, {
      onToken: (token) => {
        fullText += token;
        this.appendStreamToken(article, fullText);
        this.setStatus('正在生成回答');
      },
      onDone: (doneData) => {
        this.finalizeStreamMessage(article, fullText, doneData);
        if (doneData.session_id) {
          sessionStorage.setItem('ragmate_session_id', doneData.session_id);
          emit('history:active', doneData.session_id);
        }
        this.setBusy(false);
        this.textareaEl.focus();
        emit('history:refresh');
      },
      onError: (message) => {
        this.finalizeStreamMessage(article, fullText, {}, true);
        if (!fullText) {
          const content = article.querySelector('.msg-content');
          content.replaceChildren();
          const note = document.createElement('p');
          note.className = 'stopped-note';
          note.textContent = '这次没有生成完';
          content.append(note);
        }
        this.showError(message);
        this.setStatus('回答失败');
        this.setBusy(false);
        this.textareaEl.focus();
        emit('history:refresh');
      },
      onAbort: () => {
        this.finalizeStreamMessage(article, fullText, {}, true);
        this.setBusy(false);
        this.textareaEl.focus();
      },
    }, {
      replaceLast: !!replaceLast,
      debug: debugEnabled(),
      signal: controller.signal,
    });
    if (this.abortController === controller) this.abortController = null;
  },

  stop(silent) {
    if (this.abortController) this.abortController.abort();
    if (silent) this.setBusy(false);
  },

  loadMessages(messages, title) {
    this.hideError();
    this.messagesEl.replaceChildren();
    if (!messages.length) {
      this.showHero();
      return;
    }
    let lastAssistant = -1;
    messages.forEach((message, index) => {
      if (message.role !== 'user') lastAssistant = index;
    });
    messages.forEach((message, index) => {
      if (message.role === 'user') this.addUserMessage(message.content || '');
      else {
        const article = this.startStreamMessage();
        this.finalizeStreamMessage(article, message.content || '', {
          quiet: true,
          allowRegen: index === lastAssistant,
        });
      }
    });
    emit('shell:context', title || '历史对话');
    this.scrollToBottom(true);
  },
};

function previousUser(article) {
  let el = article.previousElementSibling;
  while (el && !el.classList.contains('msg-user')) el = el.previousElementSibling;
  return el;
}

function banner(kind, title, body) {
  const el = document.createElement('div');
  el.className = `msg-banner banner-${kind}`;
  el.setAttribute('role', kind === 'refused' ? 'alert' : 'status');
  const strong = document.createElement('strong');
  strong.textContent = title;
  const p = document.createElement('p');
  p.textContent = body;
  el.append(strong, p);
  return el;
}

function claimsBanner(claims) {
  const el = document.createElement('div');
  el.className = 'msg-banner banner-warn';
  el.setAttribute('role', 'status');
  const strong = document.createElement('strong');
  strong.textContent = '部分说法缺少依据';
  const list = document.createElement('ul');
  claims.forEach((claim) => {
    const li = document.createElement('li');
    li.textContent = claim.claim || '';
    list.append(li);
  });
  el.append(strong, list);
  return el;
}

function confidencePill(confidence) {
  const el = document.createElement('span');
  const level = confidence.level || 'low';
  el.className = `confidence confidence-${level}`;
  const score = typeof confidence.score === 'number' ? confidence.score.toFixed(2) : '—';
  const chunks = confidence.chunks ?? 0;
  el.textContent = confidenceLabel(level);
  el.title = `最高相关度 ${score}，引用片段 ${chunks}`;
  el.setAttribute('aria-label', `${confidenceLabel(level)}，最高相关度 ${score}，${chunks} 个片段`);
  return el;
}

function renderRetrievalDebug(blocks, onOpen) {
  const details = document.createElement('details');
  details.className = 'retrieval-debug';
  details.open = true;
  const summary = document.createElement('summary');
  const count = blocks.reduce((sum, block) => sum + (block.chunks || []).length, 0);
  summary.textContent = `检索调试 · ${blocks.length} 次检索 · ${count} 个候选`;
  details.append(summary);
  blocks.forEach((block) => {
    const query = document.createElement('p');
    query.className = 'debug-query';
    query.textContent = block.query || '（空查询）';
    const meta = document.createElement('p');
    meta.className = 'debug-meta';
    const top = typeof block.top_score === 'number' ? block.top_score.toFixed(3) : '—';
    const threshold = typeof block.threshold === 'number' ? block.threshold.toFixed(3) : '—';
    meta.textContent = `最高分 ${top} · 阈值 ${threshold} · 留下 ${block.returned ?? 0}/${block.candidates ?? (block.chunks || []).length}`;
    details.append(query, meta);
    (block.chunks || []).forEach((chunk) => {
      const row = document.createElement('article');
      row.className = `debug-row${chunk.kept ? ' is-kept' : ' is-dropped'}`;
      const head = document.createElement('div');
      head.className = 'debug-row-head';
      const score = document.createElement('span');
      score.className = 'debug-score';
      score.textContent = typeof chunk.score === 'number' ? chunk.score.toFixed(3) : '—';
      const where = document.createElement('button');
      where.type = 'button';
      where.className = 'debug-where';
      const page = chunk.page ? ` · 第 ${chunk.page} 页` : '';
      where.textContent = `${chunk.source || '未知来源'}${page}`;
      where.addEventListener('click', () => onOpen({ source: chunk.source, page: chunk.page }, where));
      const reason = document.createElement('span');
      reason.className = 'debug-reason';
      reason.textContent = chunk.kept ? '保留' : dropReasonLabel(chunk.dropped_by);
      head.append(score, where, reason);
      const preview = document.createElement('p');
      preview.className = 'debug-preview';
      preview.textContent = chunk.preview || '';
      row.append(head, preview);
      details.append(row);
    });
  });
  return details;
}

function buildHero() {
  const hero = document.createElement('div');
  hero.className = 'hero';
  hero.id = 'hero-empty';
  const kicker = document.createElement('p');
  kicker.className = 'hero-kicker';
  kicker.textContent = '知识库问答';
  const title = document.createElement('h2');
  title.className = 'hero-title';
  title.textContent = '问文档里写过的事';
  const lead = document.createElement('p');
  lead.className = 'hero-lead';
  lead.id = 'hero-lead';
  lead.textContent = '上传并入库之后，回答会带上置信度和来源。';
  const actions = document.createElement('div');
  actions.className = 'hero-actions';
  actions.id = 'hero-actions';
  hero.append(kicker, title, lead, actions);
  return hero;
}
