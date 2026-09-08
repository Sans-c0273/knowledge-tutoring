/**
 * Mock backend. Satisfies the same `TutorApi` contract as `src/api/http.ts`, so the
 * whole UI is explorable before FastAPI exists. Selected by `USE_MOCKS` in src/config.ts.
 */

import type { TutorApi, UploadAccepted } from '../api/types';
import type {
  ChatTurnHandlers,
  ChatTurnRequest,
  IngestEvent,
  IngestStage,
  KlMapDraft,
  TurnTrace,
} from '../types';
import { INGEST_STAGES } from '../types';
import { MOCK_DRAFTS } from './drafts';
import { MOCK_KL_MAP } from './klmap';
import { pickScenario } from './scenarios';

/** Mutable copy: approving or rejecting a draft in the UI must stick for the session. */
const draftState: KlMapDraft[] = MOCK_DRAFTS.map((draft) => structuredClone(draft));
const clone = (draft: KlMapDraft): KlMapDraft => structuredClone(draft);

const SUPPORTED_EXTENSIONS = ['pdf', 'pptx', 'docx', 'md', 'txt', 'vtt', 'srt'];

let counter = 0;
const nextId = (prefix: string) => `${prefix}-${(++counter).toString().padStart(3, '0')}`;

// ---------------------------------------------------------------------------
// Ingestion simulator
// ---------------------------------------------------------------------------

type IngestListener = (event: IngestEvent) => void;

const listeners = new Set<IngestListener>();
const timers = new Set<ReturnType<typeof setTimeout>>();

function emit(event: IngestEvent) {
  for (const listener of listeners) listener(event);
}

function later(fn: () => void, ms: number) {
  const timer = setTimeout(() => {
    timers.delete(timer);
    fn();
  }, ms);
  timers.add(timer);
}

const STAGE_LABEL: Record<IngestStage, string> = {
  parse: 'Extracting text and structure',
  chunk: 'Splitting into retrievable chunks',
  embed: 'Embedding with local BGE-M3',
  'kl-extract': 'Extracting knowledge-map candidates',
  'review-ready': 'Waiting for human review',
};

function extensionOf(filename: string): string {
  const parts = filename.split('.');
  return parts.length > 1 ? (parts.pop() as string).toLowerCase() : '';
}

/**
 * Walks a file through parse → chunk → embed → kl-extract → review-ready,
 * emitting the same event shape the real `GET /api/ingest/events` stream will.
 * Filenames containing "fail" (or an unsupported extension) exercise the error path.
 */
function simulateIngest(file: { file_id: string; filename: string; failAt?: IngestStage; failMessage?: string }) {
  const { file_id, filename } = file;
  emit({ file_id, filename, stage: 'parse', percent: 0, status: 'queued', message: 'Queued' });

  let elapsed = 250;

  for (const [stageIndex, stage] of INGEST_STAGES.entries()) {
    const stageStart = (stageIndex / INGEST_STAGES.length) * 100;
    const stageSpan = 100 / INGEST_STAGES.length;

    if (file.failAt === stage) {
      later(() => {
        emit({
          file_id,
          filename,
          stage,
          percent: Math.round(stageStart),
          status: 'error',
          message: file.failMessage ?? `Failed during ${stage}`,
        });
      }, elapsed);
      return;
    }

    for (let tick = 1; tick <= 4; tick += 1) {
      const percent = Math.round(stageStart + (stageSpan * tick) / 4);
      const isLastTickOfLastStage = stageIndex === INGEST_STAGES.length - 1 && tick === 4;
      elapsed += 220 + Math.random() * 260;
      const at = elapsed;
      later(() => {
        emit({
          file_id,
          filename,
          stage,
          percent,
          status: isLastTickOfLastStage ? 'done' : 'running',
          message: isLastTickOfLastStage ? 'Ready for knowledge-map review' : STAGE_LABEL[stage],
        });
      }, at);
    }
  }
}

function acceptFile(filename: string, source: 'upload' | 'url'): UploadAccepted {
  const file_id = nextId(source === 'url' ? 'url' : 'file');
  const extension = extensionOf(filename);

  let failAt: IngestStage | undefined;
  let failMessage: string | undefined;

  if (source === 'upload' && extension && !SUPPORTED_EXTENSIONS.includes(extension)) {
    failAt = 'parse';
    failMessage = `Unsupported file type ".${extension}". Accepted: ${SUPPORTED_EXTENSIONS.map((e) => `.${e}`).join(', ')}.`;
  } else if (/fail|corrupt/i.test(filename)) {
    failAt = 'parse';
    failMessage = 'Could not extract text — the file appears to be a scan with no text layer. Re-export with OCR or upload a transcript.';
  } else if (source === 'url' && /youtu\.?be/i.test(filename)) {
    failAt = 'parse';
    failMessage = 'No captions available for this video. The POC is transcript-only — upload a .vtt/.srt/.txt transcript instead.';
  }

  simulateIngest({ file_id, filename, failAt, failMessage });
  return { file_id, filename };
}

// ---------------------------------------------------------------------------
// Chat simulator
// ---------------------------------------------------------------------------

function tokenize(text: string, isThai: boolean): string[] {
  if (isThai) {
    const size = 4;
    const tokens: string[] = [];
    for (let i = 0; i < text.length; i += size) tokens.push(text.slice(i, i + size));
    return tokens;
  }
  return text.split(/(\s+)/).filter((t) => t.length > 0);
}

function simulateTurn(request: ChatTurnRequest, handlers: ChatTurnHandlers): () => void {
  const scenario = pickScenario(request.message);
  const turn_id = nextId('turn');
  const localTimers = new Set<ReturnType<typeof setTimeout>>();
  let aborted = false;

  const schedule = (fn: () => void, ms: number) => {
    const timer = setTimeout(() => {
      localTimers.delete(timer);
      if (!aborted) fn();
    }, ms);
    localTimers.add(timer);
  };

  let at = 0;

  // The trace arrives incrementally, one pipeline stage at a time — R20 requires
  // the inspector to update while the turn is still executing.
  schedule(() => {
    handlers.onTrace({
      turn_id,
      session_id: request.session_id,
      created_at: new Date().toISOString(),
      student_message: request.message,
      language: scenario.language,
    });
  }, 0);

  for (const stage of scenario.stages) {
    at += stage.delay_ms;
    const patch = stage.patch;
    schedule(() => handlers.onTrace({ turn_id, ...patch } as Partial<TurnTrace> & { turn_id: string }), at);
  }

  const isThai = scenario.language === 'th';
  const tokens = tokenize(scenario.reply, isThai);
  const perToken = isThai ? 38 : 26;

  // The backend buffers the whole draft, runs the guardrail, and only then streams,
  // so a withheld answer can never reach the student even for an instant. That leaves
  // a silent gap between the last decision patch and the first token: `generating`
  // is what the UI has to fill it with.
  schedule(() => handlers.onStatus?.({ phase: 'generating' }), at);

  const { guardrail, ...afterStream } = scenario.final;
  at += isThai ? 1800 : 1250;
  if (guardrail) {
    // Guardrail events land before any token now — including an alarming one.
    schedule(() => handlers.onTrace({ turn_id, guardrail }), at);
  }

  at += 60;
  for (const token of tokens) {
    at += perToken;
    schedule(() => handlers.onToken(token), at);
  }

  at += 120;
  schedule(() => {
    const complete = { turn_id, ...scenario.final } as Partial<TurnTrace> & { turn_id: string };
    handlers.onTrace({ turn_id, ...afterStream } as Partial<TurnTrace> & { turn_id: string });
    // `done` carries the complete trace so a client that missed a patch converges.
    handlers.onDone(complete);
  }, at);

  return () => {
    aborted = true;
    for (const timer of localTimers) clearTimeout(timer);
    localTimers.clear();
  };
}

// ---------------------------------------------------------------------------

export const mockApi: TutorApi = {
  async uploadFiles(files) {
    return files.map((file) => acceptFile(file.name, 'upload'));
  },

  async submitUrl(url) {
    const trimmed = url.trim();
    if (!/^https?:\/\//i.test(trimmed)) {
      throw new Error('Enter a full URL starting with http:// or https://');
    }
    return acceptFile(trimmed, 'url');
  },

  subscribeIngest(onEvent) {
    listeners.add(onEvent);
    return () => {
      listeners.delete(onEvent);
    };
  },

  sendTurn(request, handlers) {
    return simulateTurn(request, handlers);
  },

  async getKlMap() {
    return MOCK_KL_MAP;
  },

  async listDrafts() {
    return draftState.map(clone);
  },

  async getDraft(courseId) {
    const draft = draftState.find((candidate) => candidate.course_id === courseId);
    if (!draft) throw new Error(`No draft for course ${courseId}`);
    return clone(draft);
  },

  async decideDraft(courseId, decision) {
    const draft = draftState.find((candidate) => candidate.course_id === courseId);
    if (!draft) throw new Error(`No draft for course ${courseId}`);
    if (decision.approve && draft.report.errors.length > 0) {
      // The UI disables the control, but the rule belongs on the server too: a map
      // that fails validation must never become live, whatever the client sends.
      throw new Error(
        `Cannot approve ${courseId}: ${draft.report.errors.length} blocking error(s) remain.`,
      );
    }
    draft.status = decision.approve ? 'approved' : 'rejected';
    draft.reviewed_at = new Date().toISOString();
    draft.review_note = decision.note;
    return clone(draft);
  },

  async revalidateDraft(courseId) {
    const draft = draftState.find((candidate) => candidate.course_id === courseId);
    if (!draft) throw new Error(`No draft for course ${courseId}`);
    // Nothing edits the YAML in a mock run, so re-validation reports the same
    // issues — enough to prove the round trip and the "no change" message.
    return clone(draft);
  },
};

export { MOCK_KL_MAP } from './klmap';
export { SCENARIOS } from './scenarios';
