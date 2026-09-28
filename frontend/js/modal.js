import { escapeHtml } from './utils.js';

export function showConfirm(message) {
  return new Promise((resolve) => {
    const overlay = document.getElementById('modal-overlay');
    document.getElementById('modal-message').textContent = message;
    overlay.classList.remove('hidden');
    const cleanup = (result) => { overlay.classList.add('hidden'); resolve(result); };
    document.getElementById('modal-confirm').onclick = () => cleanup(true);
    document.getElementById('modal-cancel').onclick = () => cleanup(false);
  });
}

export function showProgressModal(title, total) {
  const overlay = document.getElementById('modal-overlay');
  const box = overlay.querySelector('.modal-box');
  const origHTML = box.innerHTML;

  box.innerHTML = `
    <p class="modal-progress-text">${escapeHtml(title)} <span id="prog-count">0/${total}</span></p>
    <div class="modal-progress-bar-wrap"><div class="modal-progress-bar" id="prog-bar"></div></div>
    <div class="modal-progress-current" id="prog-current"></div>
    <div class="modal-actions"><button class="btn-modal btn-modal-cancel" id="prog-cancel">取消</button></div>
  `;
  overlay.classList.remove('hidden');

  let cancelled = false;
  document.getElementById('prog-cancel').onclick = () => { cancelled = true; };

  return {
    get cancelled() { return cancelled; },
    update(index, filename) {
      document.getElementById('prog-count').textContent = `${index + 1}/${total}`;
      document.getElementById('prog-bar').style.width = `${((index + 1) / total) * 100}%`;
      document.getElementById('prog-current').textContent = filename;
    },
    done(title, items, detail) {
      let html = `<p style="margin-bottom:16px;font-size:14px;font-weight:600;">${escapeHtml(title)}</p>`;
      if (items.ok && items.ok.length > 0) {
        html += `<p class="modal-result-item modal-result-ok">✓ 已入库：${items.ok.length} 个</p>`;
        items.ok.forEach(f => {
          if (f.name) html += `<p class="modal-result-detail">- ${escapeHtml(f.name)}</p>`;
        });
      }
      if (items.skipped && items.skipped.length > 0) {
        html += `<p class="modal-result-item modal-result-skip">⊘ 跳过：${items.skipped.length} 个</p>`;
        items.skipped.forEach(f => {
          html += `<p class="modal-result-detail">- ${escapeHtml(f.name)}（${escapeHtml(f.reason || '内容重复')}）</p>`;
        });
      }
      if (items.fail && items.fail.length > 0) {
        html += `<p class="modal-result-item modal-result-fail">✗ 失败：${items.fail.length} 个</p>`;
        items.fail.forEach(f => {
          html += `<p class="modal-result-detail">- ${escapeHtml(f.name)}（${escapeHtml(f.error)}）</p>`;
        });
      }
      if (detail) html += `<p class="modal-result-detail" style="margin-top:8px;color:var(--text-muted);">${escapeHtml(detail)}</p>`;
      html += `<div class="modal-actions" style="margin-top:20px"><button class="btn-modal btn-modal-confirm" id="prog-close">关闭</button></div>`;
      box.innerHTML = html;
      document.getElementById('prog-close').onclick = () => {
        overlay.classList.add('hidden');
        box.innerHTML = origHTML;
      };
    },
    close() {
      overlay.classList.add('hidden');
      box.innerHTML = origHTML;
    }
  };
}
