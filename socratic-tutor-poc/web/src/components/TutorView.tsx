import { OPERATOR_MODE } from '../config';
import type { KlMap, TurnTrace } from '../types';
import { ChatPanel } from './ChatPanel';
import { InspectorPanel } from './inspector/InspectorPanel';
import { KlMapPanel } from './klmap/KlMapPanel';
import type { useChat } from '../hooks/useChat';

/** Three-panel layout: chat | glass-box inspector | knowledge map. */
export function TutorView({
  chat,
  map,
  mapError,
  onOpenFullMap,
}: {
  chat: ReturnType<typeof useChat>;
  map: KlMap | null;
  mapError: string | null;
  onOpenFullMap: () => void;
}) {
  const trace: TurnTrace | null = chat.activeTrace;

  // Student view is the conversation and nothing else: the trace describes what was
  // withheld and why, which is the one thing a student must not be shown.
  return (
    <div className={`tutor-layout${OPERATOR_MODE ? '' : ' student-only'}`}>
      <ChatPanel
        messages={chat.messages}
        activeTurnId={chat.activeTurnId}
        streaming={chat.streaming}
        phase={chat.phase}
        onSend={chat.send}
        onSelectTurn={chat.selectTurn}
        onCancel={chat.cancel}
      />
      {OPERATOR_MODE && (
        <>
          <InspectorPanel trace={trace} streaming={chat.streaming} composing={chat.composing} />
          <KlMapPanel map={map} error={mapError} trace={trace} onOpenFullMap={onOpenFullMap} />
        </>
      )}
    </div>
  );
}
