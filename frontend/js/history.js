import { API } from './api.js';
import { formatTime, getDateKey, formatDateSeparator } from './utils.js';
import { showConfirm } from './modal.js';
import { on, emit } from './event-bus.js';
import { ChatPanel } from './chat.js';

export const HistoryPanel = {
  listEl: null,
  searchEl: null,
  sessions: [],
  activeId: null,
  query: '',

  init() {
    this.listEl = document.getElementById('history-list');
    this.searchEl = document.getElementById('session-search');
    document.getElementById('btn-refresh-sessions').addEventListener('click', () => this.load());
    this.searchEl.addEventListener('input', () => {
      this.query = this.searchEl.value.trim().toLowerCase();
      this.render(this.sessions);
    });
    on('history:refresh', () => this.load());
    on('history:active', (id) => {
      this.activeId = id;
      this.render(this.sessions);
    });
    const stored = sessionStorage.getItem('ragmate_session_id');
    if (stored) this.activeId = stored;
    this.load();
  },

  async load() {
    if (!this.sessions.length) {
      this.listEl.replaceChildren();
      this.listEl.append(skeletonLines(3));
    }
    try {
      const data = await API.getSessions();
      this.sessions = data.sessions || [];
      this.render(this.sessions);
    } catch (err) {
      this.listEl.replaceChildren();
      const empty = document.createElement('p');
      empty.className = 'history-empty';
      empty.textContent = err.message || '会话加载失败';
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.className = 'btn btn-ghost';
      retry.textContent = '重试';
      retry.addEventListener('click', () => this.load());
      this.listEl.append(empty, retry);
    }
  },

  render(sessions) {
    const filtered = this.query
      ? sessions.filter((session) => (session.first_message || '').toLowerCase().includes(this.query))
      : sessions;
    this.listEl.replaceChildren();
    if (!sessions.length) {
      const empty = document.createElement('p');
      empty.className = 'history-empty';
      empty.textContent = '还没有对话。提问之后会出现在这里。';
      this.listEl.append(empty);
      return;
    }
    if (!filtered.length) {
      const empty = document.createElement('p');
      empty.className = 'history-empty';
      empty.textContent = '没有匹配的会话';
      this.listEl.append(empty);
      return;
    }

    let lastDate = '';
    filtered.forEach((session) => {
      const dateKey = getDateKey(session.created_at);
      if (dateKey !== lastDate) {
        lastDate = dateKey;
        const sep = document.createElement('div');
        sep.className = 'history-date';
        sep.textContent = formatDateSeparator(session.created_at);
        this.listEl.append(sep);
      }
      const item = document.createElement('div');
      item.className = 'history-item';
      item.setAttribute('role', 'listitem');
      if (session.session_id === this.activeId) item.classList.add('is-active');

      const open = document.createElement('button');
      open.type = 'button';
      open.className = 'history-open';
      if (session.session_id === this.activeId) open.setAttribute('aria-current', 'true');
      const preview = document.createElement('span');
      preview.className = 'history-preview';
      preview.textContent = session.first_message || '空对话';
      const time = document.createElement('span');
      time.className = 'history-time';
      time.textContent = formatTime(session.created_at);
      open.append(preview, time);
      open.addEventListener('click', () => this.selectSession(session));

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'history-delete';
      del.setAttribute('aria-label', `删除会话：${session.first_message || '空对话'}`);
      del.textContent = '删除';
      del.addEventListener('click', () => this.deleteSession(session.session_id));

      item.append(open, del);
      this.listEl.append(item);
    });
  },

  async deleteSession(sessionId) {
    const ok = await showConfirm('删除这个会话？对话记录会从服务器移除。', {
      confirmLabel: '删除',
      danger: true,
    });
    if (!ok) return;
    try {
      await API.deleteSession(sessionId);
      const current = sessionStorage.getItem('ragmate_session_id');
      if (this.activeId === sessionId || current === sessionId) {
        this.activeId = null;
        ChatPanel.startNew();
        return;
      }
      this.load();
    } catch (err) {
      ChatPanel.showError(err.message || '删除失败');
    }
  },

  async selectSession(session) {
    this.activeId = session.session_id;
    sessionStorage.setItem('ragmate_session_id', session.session_id);
    this.render(this.sessions);
    emit('view:switch', 'chat');
    try {
      const data = await API.getHistory(session.session_id);
      emit('chat:load-messages', {
        messages: data.messages || [],
        title: session.first_message || '历史对话',
      });
    } catch (err) {
      ChatPanel.showError(err.message || '加载历史记录失败');
    }
  },
};

function skeletonLines(count) {
  const wrap = document.createDocumentFragment();
  for (let i = 0; i < count; i += 1) {
    const row = document.createElement('div');
    row.className = 'skeleton-line';
    wrap.append(row);
  }
  return wrap;
}
