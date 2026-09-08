import { API_BASE, SESSION } from '../config';
import type {
  ChatTurnHandlers,
  ChatTurnRequest,
  IngestEvent,
  KlMap,
  KlMapDraft,
  TurnTrace,
} from '../types';
import { postSse } from './sse';
import type { TutorApi, UploadAccepted } from './types';

/**
 * FastAPI puts the human-readable reason in `detail`. Surfacing the raw body instead
 * would show a reviewer `{"detail":"Cannot approve …"}` — the refusal messages are
 * written to be read directly, so unwrap them.
 */
async function failure(response: Response): Promise<Error> {
  const body = await response.text().catch(() => '');
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (typeof parsed.detail === 'string' && parsed.detail) return new Error(parsed.detail);
    if (Array.isArray(parsed.detail)) return new Error(JSON.stringify(parsed.detail));
  } catch {
    // Not JSON — an HTML error page or a proxy message. Keep the status, and enough
    // of the body to identify it without pasting a whole page into the UI.
  }
  const excerpt = body.trim().slice(0, 200);
  return new Error(`${response.status} ${response.statusText}${excerpt ? ` — ${excerpt}` : ''}`);
}

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) throw await failure(response);
  return (await response.json()) as T;
}

/**
 * The backend requires this on state-changing requests. `multipart/form-data` is a
 * CORS-simple content type, so without it any site the user visits could post to the
 * ingestion endpoints; a custom header cannot be set cross-origin without a preflight.
 */
const CSRF_HEADER = { 'X-Requested-With': 'XMLHttpRequest' };

export const httpApi: TutorApi = {
  /**
   * Both ingestion requests name the course explicitly. The backend has a default
   * (`ingest_routes.DEFAULT_COURSE_ID`), but relying on it means what is ingested,
   * reviewed and taught only share an id by coincidence of two constants agreeing;
   * the client knows the course (`SESSION.course_id`) and should say so.
   */
  async uploadFiles(files) {
    const form = new FormData();
    for (const file of files) form.append('files', file, file.name);
    form.append('course_id', SESSION.course_id);
    const response = await fetch(`${API_BASE}/ingest/upload`, {
      method: 'POST',
      headers: CSRF_HEADER,
      body: form,
    });
    return expectJson<UploadAccepted[]>(response);
  },

  async submitUrl(url) {
    const response = await fetch(`${API_BASE}/ingest/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...CSRF_HEADER },
      body: JSON.stringify({ url, course_id: SESSION.course_id }),
    });
    return expectJson<UploadAccepted>(response);
  },

  subscribeIngest(onEvent, onError) {
    const source = new EventSource(`${API_BASE}/ingest/events`);
    source.onmessage = (message) => {
      try {
        onEvent(JSON.parse(message.data) as IngestEvent);
      } catch (error) {
        onError?.(error instanceof Error ? error : new Error(String(error)));
      }
    };
    source.onerror = () => {
      onError?.(new Error('Ingestion event stream disconnected'));
    };
    return () => source.close();
  },

  sendTurn(request: ChatTurnRequest, handlers: ChatTurnHandlers) {
    const controller = new AbortController();

    postSse({
      url: `${API_BASE}/chat/turn`,
      body: request,
      signal: controller.signal,
      onFrame: (frame) => {
        try {
          switch (frame.event) {
            case 'token': {
              const payload = JSON.parse(frame.data) as { text?: string } | string;
              handlers.onToken(typeof payload === 'string' ? payload : (payload.text ?? ''));
              break;
            }
            case 'generating': {
              const payload = frame.data ? (JSON.parse(frame.data) as { message?: string }) : {};
              handlers.onStatus?.({ phase: 'generating', message: payload.message });
              break;
            }
            case 'trace': {
              handlers.onTrace(JSON.parse(frame.data) as Partial<TurnTrace> & { turn_id: string });
              break;
            }
            case 'done': {
              const payload = frame.data ? (JSON.parse(frame.data) as Partial<TurnTrace>) : undefined;
              handlers.onDone(payload);
              break;
            }
            case 'error': {
              const payload = JSON.parse(frame.data) as { message?: string };
              handlers.onError(new Error(payload.message ?? 'Turn failed'));
              break;
            }
            default:
              break;
          }
        } catch (error) {
          handlers.onError(error instanceof Error ? error : new Error(String(error)));
        }
      },
    }).catch((error: unknown) => {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      handlers.onError(error instanceof Error ? error : new Error(String(error)));
    });

    return () => controller.abort();
  },

  async getKlMap(courseId) {
    const response = await fetch(`${API_BASE}/klmap?course_id=${encodeURIComponent(courseId)}`);
    return expectJson<KlMap>(response);
  },

  async listDrafts() {
    return expectJson<KlMapDraft[]>(await fetch(`${API_BASE}/klmap/drafts`));
  },

  async getDraft(courseId) {
    const response = await fetch(`${API_BASE}/klmap/drafts/${encodeURIComponent(courseId)}`);
    return expectJson<KlMapDraft>(response);
  },

  async decideDraft(courseId, decision) {
    const response = await fetch(`${API_BASE}/klmap/drafts/${encodeURIComponent(courseId)}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(decision),
    });
    return expectJson<KlMapDraft>(response);
  },

  async revalidateDraft(courseId) {
    const response = await fetch(
      `${API_BASE}/klmap/drafts/${encodeURIComponent(courseId)}/revalidate`,
      { method: 'POST' },
    );
    return expectJson<KlMapDraft>(response);
  },
};
