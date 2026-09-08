import type {
  ChatTurnHandlers,
  ChatTurnRequest,
  DraftDecision,
  IngestEvent,
  KlMap,
  KlMapDraft,
} from '../types';

export interface UploadAccepted {
  file_id: string;
  filename: string;
}

/** The contract both the mock and the real backend satisfy. */
export interface TutorApi {
  /** POST /api/ingest/upload (multipart) */
  uploadFiles(files: File[]): Promise<UploadAccepted[]>;
  /** POST /api/ingest/url */
  submitUrl(url: string): Promise<UploadAccepted>;
  /** GET /api/ingest/events (SSE) — returns an unsubscribe function. */
  subscribeIngest(onEvent: (event: IngestEvent) => void, onError?: (e: Error) => void): () => void;
  /** POST /api/chat/turn (SSE: token | trace | done) — returns an abort function. */
  sendTurn(request: ChatTurnRequest, handlers: ChatTurnHandlers): () => void;
  /** GET /api/klmap — the live course map. */
  getKlMap(courseId: string): Promise<KlMap>;
  /** GET /api/klmap/drafts — extracted maps awaiting review (R18). */
  listDrafts(): Promise<KlMapDraft[]>;
  /** GET /api/klmap/drafts/{course_id} */
  getDraft(courseId: string): Promise<KlMapDraft>;
  /** POST /api/klmap/drafts/{course_id}/approve — approve or reject; audit-logged. */
  decideDraft(courseId: string, decision: DraftDecision): Promise<KlMapDraft>;
  /** POST /api/klmap/drafts/{course_id}/revalidate — re-read the YAML after an edit on disk. */
  revalidateDraft(courseId: string): Promise<KlMapDraft>;
}
