import { on } from './event-bus.js';
import { applyTheme, getThemeChoice } from './prefs.js';
import { initModal } from './modal.js';
import { ChatPanel } from './chat.js';
import { HistoryPanel } from './history.js';
import { DocumentsPanel } from './documents.js';
import { SettingsPanel } from './settings.js';
import { SourcePanel } from './sources.js';

const TITLES = { chat: '对话', documents: '文档', settings: '设置' };

function setNavOpen(open) {
  document.getElementById('shell').classList.toggle('nav-open', open);
  document.getElementById('btn-nav-toggle').setAttribute('aria-expanded', open ? 'true' : 'false');
  document.getElementById('nav-scrim').hidden = !open;
}

function switchView(name) {
  if (!TITLES[name]) return;
  Object.keys(TITLES).forEach((view) => {
    const section = document.getElementById(`view-${view}`);
    if (section) section.hidden = view !== name;
    const tab = document.querySelector(`.nav-tab[data-view="${view}"]`);
    if (!tab) return;
    const active = view === name;
    tab.classList.toggle('is-active', active);
    if (active) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  });
  document.getElementById('nav-sessions').hidden = name !== 'chat';
  document.getElementById('view-title').textContent = TITLES[name];
  document.getElementById('chat-context').hidden = name !== 'chat';
  document.title = `RagMate · ${TITLES[name]}`;
  setNavOpen(false);
  if (name === 'documents') DocumentsPanel.onShow();
  if (name === 'settings') SettingsPanel.onShow();
  if (name === 'chat') ChatPanel.onShow();
}

document.addEventListener('DOMContentLoaded', () => {
  applyTheme(getThemeChoice());
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  media.addEventListener?.('change', () => {
    if (getThemeChoice() === 'system') applyTheme('system');
  });

  initModal();
  SourcePanel.init();
  ChatPanel.init();
  HistoryPanel.init();
  DocumentsPanel.init();
  SettingsPanel.init();

  document.querySelectorAll('.nav-tab').forEach((tab) => {
    tab.addEventListener('click', () => switchView(tab.dataset.view));
  });
  document.getElementById('btn-nav-toggle').addEventListener('click', () => {
    const open = !document.getElementById('shell').classList.contains('nav-open');
    setNavOpen(open);
  });
  document.getElementById('nav-scrim').addEventListener('click', () => setNavOpen(false));
  document.getElementById('btn-ready').addEventListener('click', () => switchView('settings'));

  on('view:switch', (name) => switchView(name));
  on('shell:context', (text) => {
    document.getElementById('chat-context').textContent = text || '新对话';
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      if (document.getElementById('confirm-dialog')?.open) return;
      const sourceOpen = !document.getElementById('source-panel').hidden;
      if (sourceOpen) return;
      if (document.getElementById('shell').classList.contains('nav-open')) setNavOpen(false);
      return;
    }
    if (event.altKey && !event.metaKey && !event.ctrlKey && !event.shiftKey) {
      const map = { 1: 'chat', 2: 'documents', 3: 'settings' };
      if (map[event.key]) {
        event.preventDefault();
        switchView(map[event.key]);
      }
    }
  });

  window.setInterval(() => SettingsPanel.refreshReady({ quiet: true }), 30000);
  window.addEventListener('beforeunload', () => clearTimeout(DocumentsPanel.pollTimer));
});
