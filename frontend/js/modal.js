export function initModal() {
  const dialog = document.getElementById('confirm-dialog');
  if (!dialog) return;
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close('cancel');
  });
}

export function showConfirm(message, options = {}) {
  const dialog = document.getElementById('confirm-dialog');
  const ok = document.getElementById('confirm-ok');
  document.getElementById('confirm-message').textContent = message;
  ok.textContent = options.confirmLabel || '确定';
  ok.classList.toggle('is-danger', !!options.danger);
  return new Promise((resolve) => {
    const finish = () => {
      dialog.removeEventListener('close', finish);
      resolve(dialog.returnValue === 'ok');
    };
    dialog.addEventListener('close', finish);
    dialog.showModal();
  });
}
