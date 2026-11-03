import { API } from './api.js';
import { emit } from './event-bus.js';
import { formatFileSize, statusLabel, formatDate } from './utils.js';

export const SourcePanel = {
  panel: null,
  body: null,
  title: null,
  lastTrigger: null,
  current: null,

  init() {
    this.panel = document.getElementById('source-panel');
    this.body = document.getElementById('source-body');
    this.title = document.getElementById('source-title');
    document.getElementById('btn-close-source')?.addEventListener('click', () => this.close());
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || !this.panel || this.panel.hidden) return;
      if (document.getElementById('confirm-dialog')?.open) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      this.close();
    });
  },

  async open(citation, trigger, extras = {}) {
    if (!citation?.source || !this.panel) return;
    this.current = citation;
    this.lastTrigger = trigger || document.activeElement;
    this.title.textContent = citation.page ? `${citation.source}` : citation.source;
    this.body.replaceChildren();
    this.body.append(this._metaSkeleton());
    this.panel.hidden = false;
    document.getElementById('workspace')?.classList.add('has-source');
    this.title.tabIndex = -1;
    this.title.focus();

    const doc = await this._findDoc(citation.source);
    if (this.current !== citation) return;
    this.body.replaceChildren();
    this.body.append(this._render(citation, doc, extras));
  },

  close() {
    if (!this.panel || this.panel.hidden) return;
    this.panel.hidden = true;
    document.getElementById('workspace')?.classList.remove('has-source');
    this.current = null;
    const trigger = this.lastTrigger;
    if (trigger && typeof trigger.focus === 'function') trigger.focus();
  },

  async _findDoc(name) {
    try {
      const data = await API.getDocuments();
      return (data.documents || []).find((doc) => doc.filename === name) || null;
    } catch {
      return null;
    }
  },

  _metaSkeleton() {
    const p = document.createElement('p');
    p.className = 'source-note';
    p.textContent = '正在读取文档信息…';
    return p;
  },

  _render(citation, doc, extras) {
    const wrap = document.createDocumentFragment();

    const kicker = document.createElement('p');
    kicker.className = 'source-kicker';
    kicker.textContent = citation.page ? `第 ${citation.page} 页` : '未标注页码';
    wrap.append(kicker);

    if (doc) {
      const meta = document.createElement('dl');
      meta.className = 'source-meta';
      const rows = [
        ['状态', statusLabel(doc.status)],
        ['大小', formatFileSize(doc.size_bytes)],
        ['片段', doc.chunk_count ? String(doc.chunk_count) : '—'],
        ['上传', formatDate(doc.uploaded_at)],
      ];
      rows.forEach(([term, value]) => {
        const dt = document.createElement('dt');
        dt.textContent = term;
        const dd = document.createElement('dd');
        dd.textContent = value;
        meta.append(dt, dd);
      });
      wrap.append(meta);
    } else {
      const missing = document.createElement('p');
      missing.className = 'source-note';
      missing.textContent = '文档库里没有同名文件。它可能已删除，或引用来自更早的入库结果。';
      wrap.append(missing);
    }

    const previews = extras.previews || [];
    if (previews.length) {
      const heading = document.createElement('h3');
      heading.className = 'source-subhead';
      heading.textContent = '检索片段';
      wrap.append(heading);
      previews.forEach((chunk) => {
        const block = document.createElement('blockquote');
        block.className = 'source-preview';
        const cap = document.createElement('p');
        cap.className = 'source-preview-cap';
        const bits = [];
        if (typeof chunk.score === 'number') bits.push(`相关度 ${chunk.score.toFixed(3)}`);
        bits.push(chunk.kept ? '已采用' : '未采用');
        cap.textContent = bits.join(' · ');
        const quote = document.createElement('p');
        quote.textContent = chunk.preview || '';
        block.append(cap, quote);
        wrap.append(block);
      });
    } else {
      const note = document.createElement('p');
      note.className = 'source-note';
      note.textContent = '片段原文只在打开「检索调试」后随这次回答返回。这里可以核对文件名和页码，并跳到文档库。';
      wrap.append(note);
    }

    if (doc) {
      const jump = document.createElement('button');
      jump.type = 'button';
      jump.className = 'btn btn-secondary';
      jump.textContent = '在文档库中查看';
      jump.addEventListener('click', () => {
        emit('library:focus', citation.source);
        emit('view:switch', 'documents');
      });
      wrap.append(jump);
    }
    return wrap;
  },
};
