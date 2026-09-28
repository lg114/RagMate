import { on, emit } from './event-bus.js';
import { ChatPanel } from './chat.js';
import { HistoryPanel } from './history.js';
import { DocumentsPanel } from './documents.js';

function switchView(tab) {
  document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.sidebar-tab[data-tab="${tab}"]`)?.classList.add('active');
  document.getElementById('view-chat').classList.toggle('hidden', tab !== 'chat');
  document.getElementById('view-documents').classList.toggle('hidden', tab !== 'documents');
  if (tab === 'documents') {
    DocumentsPanel.load();
    DocumentsPanel.checkIngestStatus();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  try { ChatPanel.init(); } catch (e) { console.error('ChatPanel init failed:', e); }
  try { HistoryPanel.init(); } catch (e) { console.error('HistoryPanel init failed:', e); }
  try { DocumentsPanel.init(); } catch (e) { console.error('DocumentsPanel init failed:', e); }

  document.querySelectorAll('.sidebar-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = tab.dataset.tab;
      switchView(tabId);
      emit('view:switch', tabId);
    });
  });

  on('view:switch', (tab) => switchView(tab));

  window.addEventListener('beforeunload', () => DocumentsPanel.stopPolling());
});
