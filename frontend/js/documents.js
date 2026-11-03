import { API } from './api.js';
import { emit, on } from './event-bus.js';
import { showConfirm } from './modal.js';
import { fileExt, formatDate, formatFileSize, statusLabel } from './utils.js';

const ACCEPT = ['pdf', 'docx', 'xlsx', 'xls', 'txt', 'md'];
const MAX_BYTES = 50 * 1024 * 1024;
const STAGES = { loading: '加载', splitting: '切分', encoding: '编码', storing: '写入' };

export const DocumentsPanel = {
  docs: [],
  filter: 'all',
  query: '',
  pollTimer: null,
  phase: 'unknown',
  loadedOnce: false,
  loadError: '',
  pendingFocus: '',
  uploading: false,
  uploadQueue: [],

  init() {
    this.listEl = document.getElementById('doc-list');
    this.emptyEl = document.getElementById('doc-empty');
    this.summaryEl = document.getElementById('library-summary');
    this.errorEl = document.getElementById('library-error');
    this.bannerEl = document.getElementById('ingest-banner');
    this.queueEl = document.getElementById('upload-queue');
    this.fileInput = document.getElementById('file-input');
    this.uploadZone = document.getElementById('upload-zone');
    this.searchEl = document.getElementById('doc-search');
    this.selectAllEl = document.getElementById('doc-select-all');
    this.ingestBtn = document.getElementById('btn-ingest');
    this.deleteBtn = document.getElementById('btn-batch-delete');
    this.chipEl = document.getElementById('ingest-chip');

    this.fileInput.addEventListener('change', () => {
      this.enqueue(Array.from(this.fileInput.files || []));
      this.fileInput.value = '';
    });
    ['dragover', 'dragenter'].forEach((name) => {
      this.uploadZone.addEventListener(name, (event) => {
        event.preventDefault();
        this.uploadZone.classList.add('is-drag');
      });
    });
    this.uploadZone.addEventListener('dragleave', () => this.uploadZone.classList.remove('is-drag'));
    this.uploadZone.addEventListener('drop', (event) => {
      event.preventDefault();
      this.uploadZone.classList.remove('is-drag');
      this.enqueue(Array.from(event.dataTransfer?.files || []));
    });

    this.searchEl.addEventListener('input', () => {
      this.query = this.searchEl.value.trim().toLowerCase();
      this.render();
    });
    document.querySelectorAll('[data-filter]').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.filter = btn.dataset.filter;
        document.querySelectorAll('[data-filter]').forEach((el) => {
          el.classList.toggle('is-active', el === btn);
          if (el === btn) el.setAttribute('aria-pressed', 'true');
          else el.removeAttribute('aria-pressed');
        });
        this.render();
      });
    });
    this.selectAllEl.addEventListener('change', () => {
      this.visibleRows().forEach((row) => {
        const box = row.querySelector('input[type="checkbox"]');
        if (box) box.checked = this.selectAllEl.checked;
      });
      this.updateSelection();
    });
    this.ingestBtn.addEventListener('click', () => this.startIngest());
    this.deleteBtn.addEventListener('click', () => this.deleteSelected());
    this.chipEl.addEventListener('click', () => emit('view:switch', 'documents'));
    document.getElementById('btn-library-retry')?.addEventListener('click', () => this.load());
    on('library:focus', (name) => {
      this.pendingFocus = name || '';
      this.filter = 'all';
      this.query = '';
      this.searchEl.value = '';
      document.querySelectorAll('[data-filter]').forEach((el) => {
        const on = el.dataset.filter === 'all';
        el.classList.toggle('is-active', on);
        if (on) el.setAttribute('aria-pressed', 'true');
        else el.removeAttribute('aria-pressed');
      });
      this.render();
    });

    this.load();
    this.watch();
  },

  onShow() {
    this.load();
    this.checkIngest();
  },

  async load() {
    if (!this.loadedOnce) this.renderSkeleton();
    try {
      const data = await API.getDocuments();
      this.docs = data.documents || [];
      this.loadError = '';
    } catch (err) {
      this.loadError = err.message || '无法加载文档';
      if (!this.loadedOnce) this.docs = [];
    }
    this.loadedOnce = true;
    this.render();
    emit('library:changed');
  },

  enqueue(files) {
    files.forEach((file) => {
      const ext = fileExt(file.name);
      const item = { file, name: file.name, progress: 0, state: 'queued', error: '' };
      if (!ACCEPT.includes(ext)) {
        item.state = 'error';
        item.error = '不支持的格式';
      } else if (file.size > MAX_BYTES) {
        item.state = 'error';
        item.error = '超过 50 MB';
      }
      this.uploadQueue.push(item);
    });
    this.renderQueue();
    this.pumpQueue();
  },

  renderQueue() {
    this.queueEl.replaceChildren();
    this.queueEl.hidden = this.uploadQueue.length === 0;
    this.uploadQueue.forEach((item) => {
      const li = document.createElement('li');
      li.className = `upload-item is-${item.state}`;
      const name = document.createElement('span');
      name.className = 'upload-name';
      name.textContent = item.name;
      const meta = document.createElement('span');
      meta.className = 'upload-meta';
      if (item.state === 'uploading') meta.textContent = `${Math.round(item.progress * 100)}%`;
      else if (item.state === 'done') meta.textContent = '已上传';
      else if (item.state === 'error') meta.textContent = item.error || '失败';
      else meta.textContent = '等待';
      li.append(name, meta);
      if (item.state === 'uploading') {
        const bar = document.createElement('span');
        bar.className = 'upload-bar';
        const fill = document.createElement('span');
        fill.style.width = `${Math.round(item.progress * 100)}%`;
        bar.append(fill);
        li.append(bar);
      }
      this.queueEl.append(li);
    });
  },

  async pumpQueue() {
    if (this.uploading) return;
    this.uploading = true;
    while (this.uploadQueue.some((item) => item.state === 'queued')) {
      const item = this.uploadQueue.find((entry) => entry.state === 'queued');
      item.state = 'uploading';
      this.renderQueue();
      try {
        await API.uploadDocument(item.file, (ratio) => {
          item.progress = ratio;
          this.renderQueue();
        });
        item.state = 'done';
        item.progress = 1;
      } catch (err) {
        if (String(err.message || '').includes('already exists')) {
          const replace = await showConfirm(`“${item.name}” 已存在。用新文件替换它？`, { confirmLabel: '替换' });
          if (replace) {
            try {
              await API.deleteDocument(item.name);
              await API.uploadDocument(item.file, (ratio) => {
                item.progress = ratio;
                this.renderQueue();
              });
              item.state = 'done';
            } catch (retryErr) {
              item.state = 'error';
              item.error = retryErr.message || '替换失败';
            }
          } else {
            item.state = 'error';
            item.error = '已保留原文件';
          }
        } else {
          item.state = 'error';
          item.error = err.message || '上传失败';
        }
      }
      this.renderQueue();
      await this.load();
    }
    this.uploading = false;
  },

  visibleDocs() {
    return this.docs.filter((doc) => {
      if (this.query && !doc.filename.toLowerCase().includes(this.query)) return false;
      if (this.filter === 'pending') return doc.status === 'uploaded' || doc.status === 'ingesting';
      if (this.filter === 'ingested') return doc.status === 'ingested';
      if (this.filter === 'failed') return doc.status === 'failed';
      return true;
    });
  },

  renderSkeleton() {
    this.listEl.replaceChildren();
    for (let i = 0; i < 4; i += 1) {
      const row = document.createElement('div');
      row.className = 'skeleton-line';
      this.listEl.append(row);
    }
  },

  render() {
    const docs = this.visibleDocs();
    const ingested = this.docs.filter((doc) => doc.status === 'ingested').length;
    const pending = this.docs.filter((doc) => doc.status !== 'ingested').length;
    this.summaryEl.textContent = this.docs.length
      ? `${this.docs.length} 个文件 · ${ingested} 已入库 · ${pending} 待处理`
      : '还没有文档';
    this.errorEl.hidden = !this.loadError;
    this.errorEl.querySelector('span').textContent = this.loadError || '';

    this.listEl.replaceChildren();
    this.emptyEl.hidden = docs.length !== 0 || !this.loadedOnce;
    if (!this.docs.length && this.loadedOnce && !this.loadError) {
      this.emptyEl.hidden = false;
      this.emptyEl.textContent = '把文件拖到上方，或点击选择。入库之后，对话才能引用它们。';
    } else if (this.docs.length && !docs.length) {
      this.emptyEl.hidden = false;
      this.emptyEl.textContent = '没有符合筛选的文档';
    } else {
      this.emptyEl.textContent = '把文件拖到上方，或点击选择。入库之后，对话才能引用它们。';
    }

    docs.forEach((doc) => {
      const row = document.createElement('article');
      row.className = 'doc-row';
      row.dataset.filename = doc.filename;
      row.dataset.status = doc.status || '';
      if (doc.filename === this.pendingFocus) row.classList.add('is-focused');

      const checkLabel = document.createElement('label');
      checkLabel.className = 'doc-check';
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.dataset.filename = doc.filename;
      box.dataset.status = doc.status || '';
      const checkText = document.createElement('span');
      checkText.className = 'visually-hidden';
      checkText.textContent = `选择 ${doc.filename}`;
      checkLabel.append(box, checkText);
      box.addEventListener('change', () => this.updateSelection());

      const ext = document.createElement('span');
      ext.className = 'doc-ext';
      ext.textContent = fileExt(doc.filename).slice(0, 4) || 'file';

      const info = document.createElement('div');
      info.className = 'doc-info';
      const name = document.createElement('h3');
      name.className = 'doc-name';
      name.textContent = doc.filename;
      name.title = doc.filename;
      const meta = document.createElement('p');
      meta.className = 'doc-meta';
      const bits = [formatFileSize(doc.size_bytes), statusLabel(doc.status)];
      if (doc.chunk_count) bits.push(`${doc.chunk_count} 个片段`);
      bits.push(formatDate(doc.uploaded_at));
      meta.textContent = bits.join(' · ');
      info.append(name, meta);

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn btn-ghost';
      del.textContent = '删除';
      del.setAttribute('aria-label', `删除 ${doc.filename}`);
      del.addEventListener('click', () => this.deleteOne(doc.filename));

      row.append(checkLabel, ext, info, del);
      this.listEl.append(row);
    });

    if (this.pendingFocus) {
      const row = this.listEl.querySelector('.is-focused');
      row?.scrollIntoView({ block: 'center' });
    }
    this.updateSelection();
  },

  visibleRows() {
    return Array.from(this.listEl.querySelectorAll('.doc-row'));
  },

  selectedBoxes() {
    return Array.from(this.listEl.querySelectorAll('input[type="checkbox"]:checked'));
  },

  updateSelection() {
    const boxes = Array.from(this.listEl.querySelectorAll('input[type="checkbox"]'));
    const selected = boxes.filter((box) => box.checked);
    const pendingSelected = selected.filter((box) => box.dataset.status !== 'ingested');
    const pendingAll = this.docs.filter((doc) => doc.status !== 'ingested');
    this.selectAllEl.checked = boxes.length > 0 && selected.length === boxes.length;
    this.selectAllEl.indeterminate = selected.length > 0 && selected.length < boxes.length;
    this.deleteBtn.disabled = selected.length === 0;
    this.deleteBtn.textContent = selected.length ? `删除所选 (${selected.length})` : '删除所选';
    const running = this.phase === 'running' || !!this.startedAt;
    if (running) {
      this.ingestBtn.disabled = true;
      this.ingestBtn.textContent = '入库中';
      return;
    }
    if (pendingSelected.length) {
      this.ingestBtn.disabled = false;
      this.ingestBtn.textContent = `入库所选 (${pendingSelected.length})`;
    } else if (!selected.length && pendingAll.length) {
      this.ingestBtn.disabled = false;
      this.ingestBtn.textContent = `入库未处理 (${pendingAll.length})`;
    } else {
      this.ingestBtn.disabled = true;
      this.ingestBtn.textContent = this.docs.length && !pendingAll.length ? '都已入库' : '入库';
    }
  },

  async deleteOne(filename) {
    const ok = await showConfirm(`删除 “${filename}”？文件和对应向量都会移除。`, {
      confirmLabel: '删除',
      danger: true,
    });
    if (!ok) return;
    try {
      await API.deleteDocument(filename);
      await this.load();
    } catch (err) {
      this.showLibraryError(err.message || '删除失败');
    }
  },

  async deleteSelected() {
    const names = this.selectedBoxes().map((box) => box.dataset.filename);
    if (!names.length) return;
    const ok = await showConfirm(`删除 ${names.length} 个文档？文件和对应向量都会移除。`, {
      confirmLabel: '删除',
      danger: true,
    });
    if (!ok) return;
    const failures = [];
    for (let i = 0; i < names.length; i += 1) {
      this.summaryEl.textContent = `正在删除 ${i + 1}/${names.length}…`;
      try {
        await API.deleteDocument(names[i]);
      } catch (err) {
        failures.push(`${names[i]}：${err.message || '失败'}`);
      }
    }
    await this.load();
    if (failures.length) this.showLibraryError(failures.slice(0, 3).join('；'));
  },

  showLibraryError(message) {
    this.loadError = message;
    this.errorEl.hidden = false;
    this.errorEl.querySelector('span').textContent = message;
  },

  ingestTargets() {
    const selected = this.selectedBoxes();
    if (selected.length) {
      return selected
        .filter((box) => box.dataset.status !== 'ingested')
        .map((box) => box.dataset.filename);
    }
    return this.docs.filter((doc) => doc.status !== 'ingested').map((doc) => doc.filename);
  },

  async startIngest() {
    const filenames = this.ingestTargets();
    if (!filenames.length) return;
    this.ingestBtn.disabled = true;
    this.ingestBtn.textContent = '正在启动…';
    this.startedAt = Date.now();
    this.sawRunning = false;
    try {
      const result = await API.startIngest(filenames);
      if (result.status !== 'started' && result.status !== 'already_running') {
        this.startedAt = 0;
        this.renderTerminal({ status: 'failed', error: result.error || '没能启动入库' });
        this.updateSelection();
        return;
      }
      this.fillBanner({
        tone: 'running',
        title: '正在启动入库',
        detail: filenames.length ? filenames.join('、') : '全部待处理文件',
        progress: 6,
        actions: [{ id: 'cancel', label: '取消入库', kind: 'ghost' }],
      });
      this.schedule(400);
    } catch (err) {
      this.startedAt = 0;
      this.renderTerminal({ status: 'failed', error: err.message || '没能启动入库' });
      this.updateSelection();
    }
  },

  schedule(ms) {
    clearTimeout(this.pollTimer);
    this.pollTimer = setTimeout(() => this.watchTick(), ms);
  },

  watch() {
    this.watchTick();
  },

  async watchTick() {
    const status = await this.checkIngest();
    const running = status?.status === 'running' || this.startedAt;
    this.schedule(running ? 1500 : 12000);
  },

  async checkIngest() {
    let status;
    try {
      status = await API.getIngestStatus();
    } catch {
      this.chipEl.hidden = true;
      return null;
    }
    const name = status?.status || 'idle';
    const stamp = Date.parse(status?.last_ingest || '') || 0;
    const fresh = this.startedAt && stamp >= this.startedAt - 1500;
    if (name === 'running') {
      this.sawRunning = true;
      this.phase = 'running';
      this.renderRunning(status);
      this.updateSelection();
      return status;
    }
    const finished = name === 'success' || name === 'cancelled' || name === 'interrupted' || name === 'failed';
    if (finished && (this.sawRunning || fresh)) {
      this.startedAt = 0;
      this.sawRunning = false;
      this.phase = name;
      this.renderTerminal(status);
      this.chipEl.hidden = true;
      await this.load();
      return status;
    }
    if (this.startedAt && Date.now() - this.startedAt < 20000) {
      this.chipEl.hidden = false;
      this.chipEl.textContent = '入库启动中';
      return status;
    }
    if (this.startedAt) this.startedAt = 0;
    if (this.phase === 'unknown') {
      this.phase = name;
      if (name === 'failed' || name === 'interrupted') this.renderTerminal(status);
      else if (name === 'success') this.renderQuiet(status);
      this.chipEl.hidden = true;
    }
    return status;
  },

  renderRunning(status) {
    const stage = STAGES[status.stage] || '处理';
    const current = Number(status.progress) || 0;
    const total = Number(status.total) || 0;
    const ratio = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 8;
    const file = status.current_file || '';
    this.fillBanner({
      tone: 'running',
      title: `正在入库 · ${stage}`,
      detail: file ? `${file}${total ? ` · ${Math.min(current + 1, total)}/${total}` : ''}` : (total ? `${current}/${total}` : '正在准备任务'),
      progress: ratio,
      actions: [{ id: 'cancel', label: '取消入库', kind: 'ghost' }],
    });
    this.chipEl.hidden = false;
    this.chipEl.textContent = `入库中 · ${stage}`;
  },

  renderTerminal(status) {
    const name = status?.status || 'failed';
    if (name === 'success') {
      const docs = status.document_count ?? 0;
      const chunks = status.chunk_count ?? 0;
      const skipped = Array.isArray(status.skipped) ? status.skipped : [];
      let detail = status.message === 'All documents already ingested'
        ? '没有新的改动，这些文件已经入库。'
        : `完成 ${docs} 个文档，${chunks} 个片段。`;
      if (skipped.length) {
        detail += ` 跳过 ${skipped.length} 个：${skipped.slice(0, 3).map((item) => item.filename || item.name).filter(Boolean).join('、')}`;
      }
      this.fillBanner({ tone: 'ok', title: '入库完成', detail, actions: [{ id: 'dismiss', label: '关闭', kind: 'ghost' }] });
      return;
    }
    if (name === 'cancelled') {
      this.fillBanner({
        tone: 'idle',
        title: '入库已取消',
        detail: '已停止。已写入的进度以服务端为准，可以重新入库。',
        actions: [{ id: 'dismiss', label: '关闭', kind: 'ghost' }],
      });
      return;
    }
    const title = name === 'interrupted' ? '入库中断' : '入库失败';
    this.fillBanner({
      tone: 'danger',
      title,
      detail: friendlyIngestError(status?.error || status?.message),
      actions: [{ id: 'dismiss', label: '关闭', kind: 'ghost' }],
    });
  },

  renderQuiet(status) {
    if (status.status === 'cancelled') return;
    const when = status.last_ingest ? formatDate(status.last_ingest) : '';
    const docs = status.document_count ?? 0;
    const chunks = status.chunk_count ?? 0;
    this.fillBanner({
      tone: 'idle',
      title: '上次入库',
      detail: [when, `${docs} 个文档`, `${chunks} 个片段`].filter(Boolean).join(' · '),
      actions: [{ id: 'dismiss', label: '关闭', kind: 'ghost' }],
    });
  },

  fillBanner({ tone, title, detail, progress, actions }) {
    this.bannerEl.hidden = false;
    this.bannerEl.className = `ingest-banner is-${tone}`;
    this.bannerEl.replaceChildren();
    const head = document.createElement('div');
    head.className = 'ingest-copy';
    const h = document.createElement('h3');
    h.textContent = title;
    const p = document.createElement('p');
    p.textContent = detail || '';
    head.append(h, p);
    this.bannerEl.append(head);
    if (progress != null) {
      const bar = document.createElement('div');
      bar.className = 'progress';
      bar.setAttribute('role', 'progressbar');
      bar.setAttribute('aria-valuemin', '0');
      bar.setAttribute('aria-valuemax', '100');
      bar.setAttribute('aria-valuenow', String(progress));
      const fill = document.createElement('span');
      fill.style.width = `${progress}%`;
      bar.append(fill);
      this.bannerEl.append(bar);
    }
    const row = document.createElement('div');
    row.className = 'ingest-actions';
    (actions || []).forEach((action) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = action.kind === 'danger' ? 'btn btn-danger' : 'btn btn-ghost';
      btn.textContent = action.label;
      btn.addEventListener('click', () => {
        if (action.id === 'cancel') this.cancelIngest();
        if (action.id === 'dismiss') this.bannerEl.hidden = true;
      });
      row.append(btn);
    });
    this.bannerEl.append(row);
  },

  async cancelIngest() {
    try {
      await API.cancelIngest();
      this.fillBanner({
        tone: 'idle',
        title: '正在取消',
        detail: '已请求停止，等待当前步骤结束。',
        actions: [],
      });
    } catch (err) {
      this.showLibraryError(err.message || '取消失败');
    }
  },
};

function friendlyIngestError(message) {
  const text = String(message || '未知错误');
  if (text === 'Cancelled') return '入库已取消。';
  return text;
}
