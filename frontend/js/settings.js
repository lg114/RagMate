import { API } from './api.js';
import {
  applyTheme,
  cycleTheme,
  debugEnabled,
  getApiToken,
  getThemeChoice,
  setApiToken,
  setDebugEnabled,
} from './prefs.js';

const CHECK_LABELS = {
  postgresql: 'PostgreSQL',
  redis: 'Redis',
  milvus: 'Milvus',
};

const THEME_LABELS = { light: '浅色', dark: '深色', system: '跟随系统' };

export const SettingsPanel = {
  init() {
    document.querySelectorAll('input[name="theme"]').forEach((input) => {
      input.checked = input.value === getThemeChoice();
      input.addEventListener('change', () => {
        if (input.checked) applyTheme(input.value);
      });
    });
    document.addEventListener('ragmate:theme', () => this.syncTheme());
    this.syncTheme();

    const settingsDebug = document.getElementById('settings-debug');
    settingsDebug.addEventListener('change', () => setDebugEnabled(settingsDebug.checked));
    document.getElementById('chat-debug')?.addEventListener('change', (event) => {
      setDebugEnabled(event.target.checked);
    });
    setDebugEnabled(debugEnabled());

    document.getElementById('btn-token-save').addEventListener('click', () => {
      setApiToken(document.getElementById('api-token').value);
      this.tokenNote('令牌已保存在这台浏览器，之后的请求会带上它。');
    });
    document.getElementById('btn-token-clear').addEventListener('click', () => {
      setApiToken('');
      document.getElementById('api-token').value = '';
      this.tokenNote('已清除本地令牌。');
    });
    document.getElementById('btn-token-reveal').addEventListener('click', () => {
      const input = document.getElementById('api-token');
      const show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      const btn = document.getElementById('btn-token-reveal');
      btn.textContent = show ? '隐藏' : '显示';
      btn.setAttribute('aria-pressed', show ? 'true' : 'false');
    });
    document.getElementById('btn-theme-quick').addEventListener('click', () => cycleTheme());
    document.getElementById('btn-ready-refresh').addEventListener('click', () => this.refreshReady());

    const existing = getApiToken();
    if (existing) document.getElementById('api-token').value = existing;
    this.refreshReady();
  },

  onShow() {
    this.refreshReady();
  },

  syncTheme() {
    const choice = getThemeChoice();
    document.querySelectorAll('input[name="theme"]').forEach((input) => {
      input.checked = input.value === choice;
    });
    const btn = document.getElementById('btn-theme-quick');
    if (btn) btn.textContent = `外观 · ${THEME_LABELS[choice] || '跟随系统'}`;
  },

  tokenNote(text) {
    const note = document.getElementById('token-note');
    note.textContent = text;
  },

  async refreshReady({ quiet = false } = {}) {
    const list = document.getElementById('ready-list');
    const summary = document.getElementById('ready-summary');
    const pill = document.getElementById('btn-ready');
    if (!quiet) {
      summary.textContent = '正在检查 PostgreSQL、Redis 和 Milvus…';
      if (pill) {
        pill.textContent = '检查中';
        pill.dataset.state = 'pending';
      }
    }
    try {
      const data = await API.getReady();
      const ok = data.status === 'ready';
      summary.textContent = ok ? '依赖都已就绪，可以上传、入库和提问。' : '有依赖没有就绪。对话和入库可能会失败。';
      if (pill) {
        pill.textContent = ok ? '就绪' : '降级';
        pill.dataset.state = ok ? 'ready' : 'degraded';
      }
      list.replaceChildren();
      const checks = data.checks || {};
      const names = Object.keys(checks);
      if (!names.length) {
        const li = document.createElement('li');
        li.textContent = `没有检查明细（HTTP ${data.httpStatus || '—'}）`;
        list.append(li);
        return;
      }
      names.forEach((name) => {
        const li = document.createElement('li');
        li.className = checks[name] ? 'is-ok' : 'is-bad';
        const term = document.createElement('span');
        term.textContent = CHECK_LABELS[name] || name;
        const state = document.createElement('span');
        state.textContent = checks[name] ? '正常' : '不可用';
        li.append(term, state);
        list.append(li);
      });
    } catch (err) {
      summary.textContent = err.message || '无法连接就绪检查。';
      if (pill) {
        pill.textContent = '离线';
        pill.dataset.state = 'degraded';
      }
      list.replaceChildren();
    }
  },
};
