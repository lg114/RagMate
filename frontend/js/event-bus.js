const listeners = {};

export function on(event, fn) {
  (listeners[event] ||= []).push(fn);
}

export function off(event, fn) {
  const arr = listeners[event];
  if (arr) listeners[event] = arr.filter(f => f !== fn);
}

export function emit(event, data) {
  (listeners[event] || []).forEach(fn => fn(data));
}
