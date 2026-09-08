import { describe, expect, it, vi } from 'vitest';
import { parseSseChunk, postSse } from './sse';

describe('parseSseChunk', () => {
  it('parses a named event with data', () => {
    const { frames, rest } = parseSseChunk('event: token\ndata: {"text":"hi"}\n\n');
    expect(frames).toEqual([{ event: 'token', data: '{"text":"hi"}' }]);
    expect(rest).toBe('');
  });

  it('defaults to the message event when none is named', () => {
    const { frames } = parseSseChunk('data: plain\n\n');
    expect(frames).toEqual([{ event: 'message', data: 'plain' }]);
  });

  it('holds an incomplete frame back until its terminator arrives', () => {
    const first = parseSseChunk('event: token\ndata: {"text":"par');
    expect(first.frames).toEqual([]);
    expect(first.rest).toBe('event: token\ndata: {"text":"par');

    const second = parseSseChunk(`${first.rest}tial"}\n\n`);
    expect(second.frames).toEqual([{ event: 'token', data: '{"text":"partial"}' }]);
    expect(second.rest).toBe('');
  });

  it('joins multi-line data with newlines, as the SSE spec requires', () => {
    const { frames } = parseSseChunk('event: trace\ndata: line one\ndata: line two\n\n');
    expect(frames[0].data).toBe('line one\nline two');
  });

  it('reads several frames out of one chunk and keeps the tail', () => {
    const { frames, rest } = parseSseChunk(
      'event: token\ndata: a\n\nevent: token\ndata: b\n\nevent: done\ndata: {',
    );
    expect(frames.map((f) => f.data)).toEqual(['a', 'b']);
    expect(rest).toBe('event: done\ndata: {');
  });

  it('ignores comment lines but still emits the frame around them', () => {
    const { frames } = parseSseChunk(': subscribed\nevent: token\ndata: a\n\n');
    expect(frames).toEqual([{ event: 'token', data: 'a' }]);
  });

  it('accepts CRLF line endings', () => {
    const { frames } = parseSseChunk('event: token\r\ndata: a\r\n\r\n');
    expect(frames).toEqual([{ event: 'token', data: 'a' }]);
  });

  it('strips exactly one leading space after the colon', () => {
    const { frames } = parseSseChunk('data:  two spaces\n\n');
    expect(frames[0].data).toBe(' two spaces');
  });

  it('treats a field with no value as empty rather than dropping the frame', () => {
    const { frames } = parseSseChunk('event: done\ndata\n\n');
    expect(frames).toEqual([{ event: 'done', data: '' }]);
  });
});

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

function mockFetch(response: Response) {
  return vi.fn<typeof fetch>().mockResolvedValue(response);
}

describe('postSse', () => {
  it('delivers frames split across network chunks in order', async () => {
    const body = streamOf(['event: token\ndata: a\n\nev', 'ent: token\ndata: b\n\n', 'event: done\ndata: {}\n\n']);
    vi.stubGlobal('fetch', mockFetch(new Response(body, { status: 200 })));

    const seen: string[] = [];
    await postSse({
      url: '/api/chat/turn',
      body: { message: 'hi' },
      signal: new AbortController().signal,
      onFrame: (frame) => seen.push(`${frame.event}:${frame.data}`),
    });

    expect(seen).toEqual(['token:a', 'token:b', 'done:{}']);
    vi.unstubAllGlobals();
  });

  it('does not split a multi-byte character across chunk boundaries', async () => {
    // "ค" is three UTF-8 bytes; the stream cuts through the middle of it.
    const encoded = new TextEncoder().encode('event: token\ndata: ค\n\n');
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoded.slice(0, 20));
        controller.enqueue(encoded.slice(20));
        controller.close();
      },
    });
    vi.stubGlobal('fetch', mockFetch(new Response(body, { status: 200 })));

    const seen: string[] = [];
    await postSse({
      url: '/api/chat/turn',
      body: {},
      signal: new AbortController().signal,
      onFrame: (frame) => seen.push(frame.data),
    });

    expect(seen).toEqual(['ค']);
    vi.unstubAllGlobals();
  });

  it('raises with the server detail on a non-2xx response', async () => {
    vi.stubGlobal('fetch', mockFetch(new Response('session expired', { status: 409, statusText: 'Conflict' })));

    await expect(
      postSse({
        url: '/api/chat/turn',
        body: {},
        signal: new AbortController().signal,
        onFrame: () => undefined,
      }),
    ).rejects.toThrow(/409 Conflict — session expired/);

    vi.unstubAllGlobals();
  });

  it('raises when the response carries no body to stream', async () => {
    vi.stubGlobal('fetch', mockFetch(new Response(null, { status: 204 })));

    await expect(
      postSse({
        url: '/api/chat/turn',
        body: {},
        signal: new AbortController().signal,
        onFrame: () => undefined,
      }),
    ).rejects.toThrow(/no body/i);

    vi.unstubAllGlobals();
  });
});
