/**
 * Minimal SSE frame parser for `fetch`-based streams.
 *
 * `EventSource` only does GET, and the chat endpoint is `POST /api/chat/turn`,
 * so POST streams are read off the response body and parsed here.
 */

export interface SseFrame {
  event: string;
  data: string;
}

export function parseSseChunk(buffer: string): { frames: SseFrame[]; rest: string } {
  const frames: SseFrame[] = [];
  const parts = buffer.split(/\r?\n\r?\n/);
  const rest = parts.pop() ?? '';

  for (const block of parts) {
    let event = 'message';
    const dataLines: string[] = [];
    for (const rawLine of block.split(/\r?\n/)) {
      const line = rawLine.trimEnd();
      if (!line || line.startsWith(':')) continue;
      const sep = line.indexOf(':');
      const field = sep === -1 ? line : line.slice(0, sep);
      const value = sep === -1 ? '' : line.slice(sep + 1).replace(/^ /, '');
      if (field === 'event') event = value;
      else if (field === 'data') dataLines.push(value);
    }
    if (dataLines.length > 0 || event !== 'message') {
      frames.push({ event, data: dataLines.join('\n') });
    }
  }

  return { frames, rest };
}

export interface PostSseOptions {
  url: string;
  body: unknown;
  signal: AbortSignal;
  onFrame: (frame: SseFrame) => void;
}

export async function postSse({ url, body, signal, onFrame }: PostSseOptions): Promise<void> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`${response.status} ${response.statusText}${detail ? ` — ${detail}` : ''}`);
  }
  if (!response.body) {
    throw new Error('Response has no body; SSE stream unavailable');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { frames, rest } = parseSseChunk(buffer);
      buffer = rest;
      for (const frame of frames) onFrame(frame);
    }
  } finally {
    reader.releaseLock();
  }
}
