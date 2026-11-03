const THEME_KEY = 'ragmate_theme';
const DEBUG_KEY = 'ragmate_debug';
const TOKEN_KEY = 'ragmate_api_token';

function read(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function write(key, value) {
  try {
    if (value == null || value === '') localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch { /* private mode */ }
}

export function getThemeChoice() {
  return read(THEME_KEY) || 'system';
}

export function applyTheme(choice) {
  const next = choice || 'system';
  write(THEME_KEY, next);
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dark = next === 'dark' || (next !== 'light' && systemDark);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.documentElement.dataset.themeChoice = next;
  document.dispatchEvent(new CustomEvent('ragmate:theme', { detail: next }));
  return next;
}

export function cycleTheme() {
  const order = ['light', 'dark', 'system'];
  const current = getThemeChoice();
  const next = order[(order.indexOf(current) + 1) % order.length];
  return applyTheme(next);
}

export function debugEnabled() {
  return read(DEBUG_KEY) === '1';
}

export function setDebugEnabled(on) {
  write(DEBUG_KEY, on ? '1' : '0');
  document.querySelectorAll('[data-debug-toggle]').forEach((el) => {
    if (el instanceof HTMLInputElement && el.checked !== !!on) el.checked = !!on;
  });
  document.dispatchEvent(new CustomEvent('ragmate:debug', { detail: !!on }));
}

export function getApiToken() {
  return (read(TOKEN_KEY) || '').trim();
}

export function setApiToken(value) {
  write(TOKEN_KEY, (value || '').trim());
}
