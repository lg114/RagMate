export function getSessionId() {
  let sid = sessionStorage.getItem('ragmate_session_id');
  if (!sid) {
    sid = crypto.randomUUID?.() || 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
    sessionStorage.setItem('ragmate_session_id', sid);
  }
  return sid;
}

export function newSession() {
  sessionStorage.removeItem('ragmate_session_id');
}
