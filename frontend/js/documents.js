import { API } from './api.js';
import { escapeHtml, formatFileSize, formatDate, statusLabel } from './utils.js';
import { showConfirm, showProgressModal } from './modal.js';
import { emit } from './event-bus.js';
import { ChatPanel } from './chat.js';

export const DocumentsPanel = {
  listEl:          document.getElementById('doc-card-list'),
  emptyEl:         document.getElementById('doc-empty'),
  uploadZone:      document.getElementById('upload-zone'),
  fileInput:       document.getElementById('file-input'),
  uploadError:     document.getElementById('upload-error'),
  ingestBtn:       document.getElementById('btn-ingest'),
  ingestBtnText:   document.getElementById('btn-ingest-text'),
  ingestSpinner:   document.getElementById('btn-ingest-spinner'),
  selectAllEl:     document.getElementById('doc-select-all'),
  batchDeleteBtn:  document.getElementById('btn-batch-delete'),
  batchDeleteText: document.getElementById('btn-batch-delete-text'),
  _ingestStateTimer: null,
  _ingestHandled:    false,
  _ingestModal:      null,
  pollTimer:         null,
  _uploadQueue:      [],
  _uploading:        false,
  _suppressCheckboxEvent: false,

  init() {
    this.fileInput.addEventListener('change', (e) => {
      const files = Array.from(e.target.files);
      e.target.value = '';
      this._enqueueUploads(files);
    });

    this.uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); this.uploadZone.classList.add('drag-over'); });
    this.uploadZone.addEventListener('dragleave', () => this.uploadZone.classList.remove('drag-over'));
    this.uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadZone.classList.remove('drag-over');
      this._enqueueUploads(Array.from(e.dataTransfer.files));
    });

    this.ingestBtn.addEventListener('click', () => this.startIngest());

    this.selectAllEl.addEventListener('change', () => {
      const checked = this.selectAllEl.checked;
      const listEl = document.getElementById('doc-card-list');
      if (!listEl) return;
      this._suppressCheckboxEvent = true;
      listEl.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.checked = checked; });
      this._suppressCheckboxEvent = false;
      this._updateBatchBar();
    });

    this.batchDeleteBtn.addEventListener('click', () => this.batchDelete());

    this._updateBatchBar();
    this.load();
  },

  _getSelectedFiles() {
    const listEl = document.getElementById('doc-card-list');
    if (!listEl) return [];
    return Array.from(listEl.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.dataset.filename);
  },

  _updateBatchBar() {
    const listEl = document.getElementById('doc-card-list');
    if (!listEl) return;
    const selectedCbs = Array.from(listEl.querySelectorAll('input[type="checkbox"]:checked'));
    const count = selectedCbs.length;
    const hasSelection = count > 0;
    const hasUnIngested = selectedCbs.some(cb => cb.dataset.status !== 'ingested');
    this.ingestBtn.removeAttribute('data-state');
    this.ingestBtn.disabled = !hasUnIngested;
    this.batchDeleteBtn.disabled = !hasSelection;
    this.batchDeleteText.textContent = hasSelection ? `删除 (${count})` : '删除';
    this.ingestBtnText.textContent = hasUnIngested ? `入库 (${selectedCbs.filter(cb => cb.dataset.status !== 'ingested').length})` : '入库';
    const allCbs = listEl.querySelectorAll('input[type="checkbox"]');
    this.selectAllEl.checked = allCbs.length > 0 && count === allCbs.length;
    this.selectAllEl.indeterminate = count > 0 && count < allCbs.length;
  },

  async batchDelete() {
    const files = this._getSelectedFiles();
    if (files.length === 0) return;
    if (!await showConfirm(`确定删除 ${files.length} 个文档？`)) return;

    const modal = showProgressModal('正在删除文档…', files.length);
    const items = { ok: [], fail: [] };

    for (let i = 0; i < files.length; i++) {
      if (modal.cancelled) break;
      modal.update(i, files[i]);
      try {
        await API.deleteDocument(files[i]);
        items.ok.push(files[i]);
      } catch (err) {
        items.fail.push({ name: files[i], error: err.message || '未知错误' });
      }
    }

    modal.done('删除完成', {
      ok: items.ok.map(f => ({ name: f })),
      fail: items.fail,
    });
    this.selectAllEl.checked = false;
    this._updateBatchBar();
    await this.load();
  },

  async load() {
    this._renderSkeleton();
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

  _renderSkeleton() {
    const skeleton = document.getElementById('doc-skeleton');
    if (!skeleton) return;
    skeleton.innerHTML = Array.from({length: 4}, () =>
      `<div class="skeleton-row">
        <div class="skeleton skeleton-check"></div>
        <div class="skeleton skeleton-bar w-40"></div>
        <div class="skeleton skeleton-bar w-10"></div>
        <div class="skeleton skeleton-bar w-16"></div>
        <div class="skeleton skeleton-bar w-20"></div>
      </div>`
    ).join('');
  },

  render(docs) {
    const listEl = document.getElementById('doc-card-list');
    if (!listEl) return;
    const skeleton = document.getElementById('doc-skeleton');
    if (skeleton) skeleton.innerHTML = '';

    listEl.querySelectorAll('.doc-card').forEach(el => el.remove());

    this.selectAllEl.checked = false;
    this.selectAllEl.indeterminate = false;
    this._updateBatchBar();

    if (docs.length === 0) {
      this.emptyEl.classList.remove('hidden');
      return;
    }
    this.emptyEl.classList.add('hidden');

    docs.sort((a, b) => (b.uploaded_at || '').localeCompare(a.uploaded_at || ''));
    docs.forEach(doc => {
      const ext = doc.filename.split('.').pop().toLowerCase();
      const iconClass = ['pdf','docx','xlsx','txt','md'].includes(ext) ? ext : 'txt';
      const card = document.createElement('div');
      card.className = 'doc-card';
      card.innerHTML = `
        <div class="doc-card-check"><input type="checkbox" data-filename="${escapeHtml(doc.filename)}" data-status="${doc.status}"></div>
        <div class="doc-card-icon ${iconClass}">${ext}</div>
        <div class="doc-card-info">
          <div class="doc-card-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
          <div class="doc-card-meta">
            <span>${formatFileSize(doc.size_bytes)}</span>
            <span class="badge badge-${doc.status}">${statusLabel(doc.status)}${doc.chunk_count ? ' · ' + doc.chunk_count + ' 片段' : ''}</span>
            <span>${formatDate(doc.uploaded_at)}</span>
          </div>
        </div>
        <div class="doc-card-actions">
          <button class="btn-delete" data-filename="${escapeHtml(doc.filename)}">删除</button>
        </div>
      `;
      card.querySelector('input[type="checkbox"]').addEventListener('change', () => {
        if (!this._suppressCheckboxEvent) this._updateBatchBar();
      });
      card.querySelector('.btn-delete').addEventListener('click', () => this.deleteDoc(doc.filename));
      listEl.appendChild(card);
    });
  },

  _enqueueUploads(files) {
    this._uploadQueue.push(...files);
    this._processUploadQueue();
  },

  async _processUploadQueue() {
    if (this._uploading) return;
    this._uploading = true;
    while (this._uploadQueue.length > 0) {
      const file = this._uploadQueue.shift();
      await this.upload(file);
    }
    this._uploading = false;
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
      if (err.message && err.message.includes('already exists')) {
        if (await showConfirm(`"${file.name}" 已存在，是否替换？`)) {
          try {
            await API.deleteDocument(file.name);
            await API.uploadDocument(file);
            await this.load();
          } catch (retryErr) {
            this.showUploadError(retryErr.message || '替换失败');
          }
        }
      } else {
        this.showUploadError(err.message || '上传失败');
      }
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
    const files = this._getSelectedFiles();
    this._ingestHandled = false;
    this.setIngestBtnState('running');
    this.batchDeleteBtn.disabled = true;

    try {
      const result = await API.startIngest(files.length > 0 ? files : null);
      if (result.status === 'already_running') {
        this.pollIngest();
      } else {
        this._ingestModal = showProgressModal('正在入库…', files.length || 1);
        document.getElementById('prog-cancel').onclick = () => {
          this._ingestModal.close();
          this._ingestModal = null;
        };
        this.pollIngest();
      }
    } catch (err) {
      this.setIngestBtnState('failed');
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
        this._ingestHandled = false;
        this.stopPolling();
        if (this._ingestModal) { this._ingestModal.close(); this._ingestModal = null; }
        this.setIngestBtnState('default');
        return;
      }
      if (this._ingestHandled && status.status !== 'running') return;
      if (status.status === 'running') {
        const stage = status.stage || '';
        const stageLabels = {loading: '加载中', splitting: '切分中', encoding: '编码中', storing: '写入中'};
        const stageLabel = stageLabels[stage] || '处理中';
        const file = status.current_file || '';
        const progress = status.total || 0;
        const current = status.progress || 0;
        if (this._ingestModal && progress > 0) {
          this._ingestModal.update(current, `${stageLabel} — ${file}`);
        }
        this.setIngestBtnState('running');
      } else if (status.status === 'success') {
        this.stopPolling();
        if (this._ingestModal) {
          const docCount = status.document_count || 0;
          const chunkCount = status.chunk_count || 0;
          const skipped = (status.skipped || []).map(s => ({ name: s.filename, reason: s.reason }));
          const okItems = docCount > 0 ? Array.from({length: docCount}, (_, i) => ({ name: (status.filenames || [])[i] || `文档${i+1}` })) : [];
          this._ingestModal.done('入库完成', { ok: okItems, skipped, fail: [] },
            skipped.length === 0 ? `${docCount} 个文档，${chunkCount} 个片段` : '');
          this._ingestModal = null;
        }
        this._ingestHandled = true;
        this.setIngestBtnState('done');
        await this.refreshList();
        ChatPanel.updateHeroState();
      } else if (status.status === 'failed') {
        this.stopPolling();
        if (this._ingestModal) {
          this._ingestModal.done('入库失败', {
            ok: [],
            fail: [{ name: '入库任务', error: status.error || '未知错误' }],
          });
          this._ingestModal = null;
        }
        this._ingestHandled = true;
        this.setIngestBtnState('failed');
      }
    } catch (err) {
      this.stopPolling();
      if (this._ingestModal) { this._ingestModal.close(); this._ingestModal = null; }
      this.setIngestBtnState('default');
    }
  },

  stopPolling() {
    if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
  },

  setIngestBtnState(state) {
    if (this._ingestStateTimer) { clearTimeout(this._ingestStateTimer); this._ingestStateTimer = null; }
    this.ingestBtn.removeAttribute('data-state');
    if (state === 'running') {
      this.ingestBtn.disabled = true;
      this.ingestBtn.setAttribute('data-state', 'running');
      this.ingestSpinner.classList.remove('hidden');
      this.ingestBtnText.textContent = '入库中';
    } else if (state === 'done') {
      this.ingestBtn.disabled = true;
      this.ingestBtn.setAttribute('data-state', 'done');
      this.ingestSpinner.classList.add('hidden');
      this.ingestBtnText.textContent = '完成';
      this._ingestStateTimer = setTimeout(() => this.setIngestBtnState('default'), 2000);
    } else if (state === 'failed') {
      this.ingestBtn.disabled = true;
      this.ingestBtn.setAttribute('data-state', 'failed');
      this.ingestSpinner.classList.add('hidden');
      this.ingestBtnText.textContent = '失败';
      this._ingestStateTimer = setTimeout(() => this.setIngestBtnState('default'), 2000);
    } else {
      this.ingestSpinner.classList.add('hidden');
      this._updateBatchBar();
    }
  },

  showUploadError(msg) {
    this.uploadError.textContent = msg;
    this.uploadError.classList.remove('hidden');
  },

  hideUploadError() {
    this.uploadError.classList.add('hidden');
  }
};
