import { useEffect, useRef, useState } from 'react';
import { OPERATOR_MODE, USE_MOCKS } from '../config';
import { SCENARIOS } from '../mocks';
import type { ChatMessage } from '../types';
import { isThai, Pill } from './ui';

interface ChatPanelProps {
  messages: ChatMessage[];
  activeTurnId: string | null;
  streaming: boolean;
  /** Where the turn is, so a long wait never reads as a stall. */
  phase: 'idle' | 'deciding' | 'composing';
  onSend: (text: string) => void;
  onSelectTurn: (turnId: string) => void;
  onCancel: () => void;
}

/** R19 — the tutoring conversation, streamed, bilingual. */
export function ChatPanel({
  messages,
  activeTurnId,
  streaming,
  phase,
  onSend,
  onSelectTurn,
  onCancel,
}: ChatPanelProps) {
  const [draft, setDraft] = useState('');
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const submit = () => {
    if (streaming || draft.trim().length === 0) return;
    onSend(draft);
    setDraft('');
  };

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Chat</h2>
        <span className="muted small">student ↔ tutor</span>
        {streaming && (
          <button type="button" className="reveal-btn" style={{ marginLeft: 'auto' }} onClick={onCancel}>
            stop
          </button>
        )}
      </div>

      <div className="panel-body" ref={listRef}>
        {messages.length === 0 && (
          <div className="inspector-empty">
            Send a message to run a turn. Every tutor reply is clickable — selecting one loads its TurnTrace into the
            inspector.
          </div>
        )}
        <div className="messages">
          {messages.map((message) => (
            <Message
              key={message.id}
              message={message}
              selected={Boolean(message.turn_id) && message.turn_id === activeTurnId}
              phase={phase}
              onSelect={onSelectTurn}
            />
          ))}
        </div>
      </div>

      {USE_MOCKS && OPERATOR_MODE && (
        <div className="quick-prompts">
          {SCENARIOS.map((scenario) => (
            <button
              key={scenario.id}
              type="button"
              title={scenario.label}
              lang={scenario.language}
              onClick={() => setDraft(scenario.prompt)}
            >
              {scenario.label}
            </button>
          ))}
        </div>
      )}

      <div className="composer">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask in Thai or English — พิมพ์ภาษาไทยหรืออังกฤษก็ได้"
            rows={2}
            lang={isThai(draft) ? 'th' : 'en'}
          />
          <button type="submit" className="btn primary" disabled={streaming || draft.trim().length === 0}>
            Send
          </button>
        </form>
        <div className="composer-hint">Enter sends · Shift+Enter adds a line. The reply language follows yours.</div>
      </div>
    </section>
  );
}

function Message({
  message,
  selected,
  phase,
  onSelect,
}: {
  message: ChatMessage;
  selected: boolean;
  phase: 'idle' | 'deciding' | 'composing';
  onSelect: (turnId: string) => void;
}) {
  const isTutor = message.role === 'tutor';
  const clickable = isTutor && Boolean(message.turn_id) && OPERATOR_MODE;

  return (
    <div
      className={`msg ${message.role}${selected ? ' selected' : ''}`}
      lang={message.language}
      onClick={() => {
        if (clickable && message.turn_id) onSelect(message.turn_id);
      }}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={(e) => {
        if (clickable && message.turn_id && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onSelect(message.turn_id);
        }
      }}
    >
      {message.text}
      {message.streaming && phase !== 'idle' && message.text.length === 0 ? (
        <span className="composing" lang="en">
          <span className="dots" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          {phase === 'deciding'
            ? 'Working out how to answer…'
            : 'Composing — the guardrail checks the draft before any of it reaches you.'}
        </span>
      ) : (
        message.streaming && <span className="cursor">&nbsp;</span>
      )}
      {message.error && (
        <div className="notice red" style={{ marginTop: 8 }} lang="en">
          Turn failed: {message.error}
        </div>
      )}
      {isTutor && message.turn_id && OPERATOR_MODE && (
        <div className="msg-meta" lang="en">
          <span className="mono-value">{message.turn_id}</span>
          {selected ? <Pill tone="blue">in inspector</Pill> : <span>click to inspect</span>}
        </div>
      )}
    </div>
  );
}
