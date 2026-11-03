import { getApiToken } from './prefs.js';

function authHeaders(extra) {
  const headers = { ...(extra || {}) };
  const token = getApiToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

function errorFrom(status, data) {
  if (status === 401) return new Error('需要 API 令牌。请在设置中填写后重试。');
  if (status === 413) return new Error(data.detail || '文件超过大小限制');
  if (status === 429) return new Error('请求过于频繁，请稍后再试。');
  const detail = data && (data.detail || data.message);
  return new Error(detail || `请求失败（${status}）`);
}

export const API = {
  async request(method, path, body, { signal } = {}) {
    const opts = { method, headers: authHeaders(), signal };
    if (body instanceof FormData) {
      opts.body = body;
    } else if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw errorFrom(res.status, data);
    return data;
  },

  async chatStream(message, sessionId, handlers, options = {}) {
    const { onToken, onDone, onError, onAbort } = handlers;
    try {
      const res = await fetch('/chat/stream', {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          message,
          session_id: sessionId,
          replace_last: !!options.replaceLast,
          debug: !!options.debug,
        }),
        signal: options.signal,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw errorFrom(res.status, data);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const handleBlock = (block) => {
        for (const line of block.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          let data;
          try { data = JSON.parse(line.slice(6)); } catch { continue; }
          if (data.done) { onDone(data); return true; }
          if (data.error) { onError(data.error); return true; }
          if (data.token) onToken(data.token);
        }
        return false;
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        if (handleBlock(lines.join('\n'))) return;
      }
      if (buffer.trim() && handleBlock(buffer)) return;
      onError('连接意外断开');
    } catch (err) {
      if (err && err.name === 'AbortError') { onAbort?.(); return; }
      onError(err.message || '请求失败，请重试');
    }
  },

  uploadDocument(file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/documents/upload');
      const token = getApiToken();
      if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      if (xhr.upload && onProgress) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) onProgress(event.loaded / event.total);
        };
      }
      xhr.onload = () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText || '{}'); } catch { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(errorFrom(xhr.status, data));
      };
      xhr.onerror = () => reject(new Error('网络错误，上传失败'));
      xhr.onabort = () => reject(new Error('上传已取消'));
      const body = new FormData();
      body.append('file', file);
      xhr.send(body);
    });
  },

  async getReady() {
    const res = await fetch('/ready', { headers: authHeaders() });
    const data = await res.json().catch(() => ({}));
    return {
      httpOk: res.ok,
      httpStatus: res.status,
      status: data.status || (res.ok ? 'ready' : 'degraded'),
      checks: data.checks || {},
    };
  },

  getDocuments() { return this.request('GET', '/documents'); },
  deleteDocument(name) { return this.request('DELETE', `/documents/${encodeURIComponent(name)}`); },
  startIngest(filenames) {
    return filenames ? this.request('POST', '/ingest', { filenames }) : this.request('POST', '/ingest');
  },
  cancelIngest() { return this.request('POST', '/ingest/cancel'); },
  getIngestStatus() { return this.request('GET', '/ingest/status'); },
  getSessions() { return this.request('GET', '/chat/sessions'); },
  getHistory(sessionId) { return this.request('GET', `/chat/sessions/${encodeURIComponent(sessionId)}`); },
  deleteSession(sessionId) { return this.request('DELETE', `/chat/sessions/${encodeURIComponent(sessionId)}`); },
};
