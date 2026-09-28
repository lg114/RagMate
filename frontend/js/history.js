import { API } from './api.js';
import { escapeHtml, formatTime, getDateKey, formatDateSeparator } from './utils.js';
import { showConfirm } from './modal.js';
import { on, emit } from './event-bus.js';
import { ChatPanel } from './chat.js';

export const HistoryPanel = {
  listEl: document.getElementById('history-list'),
  activeId: null,

  init() {
    document.getElementById('btn-refresh-sessions').addEventListener('click', () => this.load());
    this.load();

    on('history:refresh', () => this.load());
  },

  async load() {
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

    let lastDateKey = '';
    sessions.forEach(s => {
      const dateKey = getDateKey(s.created_at);
      if (dateKey !== lastDateKey) {
        lastDateKey = dateKey;
        const sep = document.createElement('div');
        sep.className = 'history-date-separator';
        sep.innerHTML = `
          <span class="timeline-line"></span>
          <span class="timeline-node"></span>
          <span class="timeline-label">${escapeHtml(formatDateSeparator(s.created_at))}</span>
        `;
        this.listEl.appendChild(sep);
      }

      const btn = document.createElement('button');
      btn.className = 'history-item' + (s.session_id === this.activeId ? ' active' : '');
      btn.setAttribute('role', 'listitem');
      btn.innerHTML = `
        <div class="history-item-preview">${escapeHtml(s.first_message)}</div>
        <div class="history-item-bottom">
          <div class="history-item-time">${formatTime(s.created_at)}</div>
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

    emit('view:switch', 'chat');
  },

  async _fetchSessions() {
    try {
      return (await API.getSessions()).sessions || [];
    } catch (err) {
      return [];
    }
  }
};
