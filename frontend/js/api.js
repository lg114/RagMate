export const API = {
  async request(method, path, body) {
    const opts = { method };
    if (body instanceof FormData) {
      opts.body = body;
    } else if (body !== undefined) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
    return data;
  },

  async chatStream(message, sessionId, onToken, onDone, onError, replaceLast) {
    try {
      const res = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId, replace_last: !!replaceLast }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Error ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.done) { onDone(data); return; }
            if (data.error) { onError(data.error); return; }
            if (data.token) onToken(data.token);
          } catch (e) { /* ignore */ }
        }
      }
      if (buffer.trim()) {
        for (const line of buffer.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.done) { onDone(data); return; }
            if (data.error) { onError(data.error); return; }
            if (data.token) onToken(data.token);
          } catch (e) { /* ignore */ }
        }
      }
      onError('连接意外断开');
    } catch (err) {
      onError(err.message || '请求失败，请重试');
    }
  },

  getDocuments()          { return this.request('GET', '/documents'); },
  uploadDocument(file)    { const f = new FormData(); f.append('file', file); return this.request('POST', '/documents/upload', f); },
  deleteDocument(name)    { return this.request('DELETE', `/documents/${encodeURIComponent(name)}`); },
  startIngest(filenames)  { return filenames ? this.request('POST', '/ingest', { filenames }) : this.request('POST', '/ingest'); },
  getIngestStatus()       { return this.request('GET', '/ingest/status'); },
  getSessions()           { return this.request('GET', '/chat/sessions'); },
  getHistory(sessionId)   { return this.request('GET', `/chat/sessions/${encodeURIComponent(sessionId)}`); },
  deleteSession(sessionId){ return this.request('DELETE', `/chat/sessions/${encodeURIComponent(sessionId)}`); },
};
