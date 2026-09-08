import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { SESSION } from '../config';
import type { ChatMessage, TurnTrace } from '../types';
import { langOf } from '../components/ui';

let messageSeq = 0;
const messageId = () => `m${++messageSeq}`;

/**
 * Owns the conversation and the traces it produced. Trace events arrive
 * incrementally during a turn and are merged, so the inspector fills in as the
 * pipeline executes rather than only at the end (R20).
 */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [traces, setTraces] = useState<Record<string, TurnTrace>>({});
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  /**
   * Where the turn is, for the waiting indicator:
   * `deciding` — sent, but nothing has come back yet. Against the real backend the
   * first trace only lands once the intent call returns, which is seconds.
   * `composing` — the plan is fixed and the model is drafting text the student
   * cannot see until the guardrail passes it.
   */
  const [phase, setPhase] = useState<'idle' | 'deciding' | 'composing'>('idle');
  const abortRef = useRef<(() => void) | null>(null);
  const pinnedRef = useRef(false);
  const tokensSeenRef = useRef(false);

  useEffect(() => () => abortRef.current?.(), []);

  const send = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    const studentId = messageId();
    const tutorId = messageId();
    const language = langOf(trimmed);

    setMessages((current) => [
      ...current,
      { id: studentId, role: 'student', text: trimmed, language },
      { id: tutorId, role: 'tutor', text: '', language, streaming: true },
    ]);
    setStreaming(true);
    setPhase('deciding');
    pinnedRef.current = false;
    tokensSeenRef.current = false;

    abortRef.current = api.sendTurn(
      { ...SESSION, message: trimmed },
      {
        onToken: (token) => {
          // The first token means the guardrail has passed the draft.
          tokensSeenRef.current = true;
          setPhase('idle');
          setMessages((current) =>
            current.map((m) => (m.id === tutorId ? { ...m, text: m.text + token } : m)),
          );
        },
        onStatus: (status) => {
          if (status.phase === 'generating') setPhase('composing');
        },
        onTrace: (partial) => {
          setTraces((current) => {
            const previous = current[partial.turn_id];
            const merged = { ...(previous ?? emptyTrace(partial.turn_id, trimmed)), ...partial } as TurnTrace;
            return { ...current, [partial.turn_id]: merged };
          });
          setMessages((current) =>
            current.map((m) =>
              m.id === tutorId
                ? { ...m, turn_id: partial.turn_id, language: partial.language ?? m.language }
                : m,
            ),
          );
          // A finished plan with no tokens yet means generation is running. Derived
          // rather than assumed, so the composing state holds even if the backend
          // sends no `generating` event.
          if (partial.response_plan && !tokensSeenRef.current) setPhase('composing');
          // Follow the live turn unless the reviewer has pinned an older one.
          if (!pinnedRef.current) setActiveTurnId(partial.turn_id);
        },
        onDone: (final) => {
          // `done` carries the complete trace, so a client that missed a patch
          // converges here rather than displaying a half-filled inspector.
          if (final?.turn_id) {
            const turnId = final.turn_id;
            setTraces((current) => ({
              ...current,
              [turnId]: { ...(current[turnId] ?? emptyTrace(turnId, trimmed)), ...final } as TurnTrace,
            }));
          }
          setStreaming(false);
          setPhase('idle');
          setMessages((current) => current.map((m) => (m.id === tutorId ? { ...m, streaming: false } : m)));
          abortRef.current = null;
        },
        onError: (error) => {
          setStreaming(false);
          setPhase('idle');
          setMessages((current) =>
            current.map((m) =>
              m.id === tutorId
                ? { ...m, streaming: false, error: error.message, text: m.text || '' }
                : m,
            ),
          );
          abortRef.current = null;
        },
      },
    );
  }, []);

  const selectTurn = useCallback((turnId: string) => {
    pinnedRef.current = true;
    setActiveTurnId(turnId);
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setStreaming(false);
    setPhase('idle');
    setMessages((current) => current.map((m) => (m.streaming ? { ...m, streaming: false } : m)));
  }, []);

  return {
    messages,
    traces,
    activeTrace: activeTurnId ? (traces[activeTurnId] ?? null) : null,
    activeTurnId,
    streaming,
    phase,
    composing: phase === 'composing',
    send,
    selectTurn,
    cancel,
  };
}

function emptyTrace(turnId: string, message: string): TurnTrace {
  return {
    turn_id: turnId,
    session_id: SESSION.session_id,
    created_at: new Date().toISOString(),
    student_message: message,
    language: langOf(message),
  };
}
