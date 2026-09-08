import { useEffect, useState } from 'react';
import type { KlMap, TurnTrace } from '../../types';
import { RELATION_LABEL } from '../../reference';
import { Pill } from '../ui';
import { KlMapGraph } from './KlMapGraph';
import { api } from '../../api';
import { SESSION } from '../../config';
import { useKlMap } from '../../hooks/useKlMap';

/**
 * Full-width course map. R18 adds approve/reject on top of this view; for now it
 * renders the map and marks what the selected turn touched.
 *
 * `map`/`error` are the *active session's* course (shared with the tutor view,
 * kept in sync with any live turn's `trace`). The course switcher below lets an
 * operator browse any other course's map — draft or approved — without moving
 * the session itself off the course it's actually teaching from; switching
 * away from the session's own course drops the per-turn "touched" highlight,
 * since a trace belongs to the turn it was recorded on, not to whatever course
 * happens to be on screen.
 */
export function KlMapView({ map: sessionMap, error: sessionError, trace }: { map: KlMap | null; error: string | null; trace: TurnTrace | null }) {
  const [courseIds, setCourseIds] = useState<string[]>([SESSION.course_id]);
  const [selectedCourseId, setSelectedCourseId] = useState(SESSION.course_id);

  useEffect(() => {
    let cancelled = false;
    api
      .listDrafts()
      .then((drafts) => {
        if (cancelled) return;
        setCourseIds(Array.from(new Set([SESSION.course_id, ...drafts.map((draft) => draft.course_id)])));
      })
      .catch(() => {
        // The switcher just falls back to the one course already known; the
        // page underneath still works without the extra options.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const isSessionCourse = selectedCourseId === SESSION.course_id;
  const { map: otherMap, error: otherError } = useKlMap(selectedCourseId, !isSessionCourse);
  const map = isSessionCourse ? sessionMap : otherMap;
  const error = isSessionCourse ? sessionError : otherError;
  const hits = isSessionCourse ? trace?.knowledge_context?.nodes_hit ?? [] : [];
  const currentNodeId = isSessionCourse ? trace?.knowledge_context?.current_node_id : undefined;

  return (
    <div className="ingest">
      <div className="ingest-inner" style={{ maxWidth: 1180 }}>
        <h1>Knowledge map</h1>
        {courseIds.length > 1 && (
          <div style={{ marginBottom: 12 }}>
            <label htmlFor="klmap-course-select" style={{ marginRight: 8 }}>
              Course
            </label>
            <select
              id="klmap-course-select"
              value={selectedCourseId}
              onChange={(event) => setSelectedCourseId(event.target.value)}
            >
              {courseIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                  {id === SESSION.course_id ? ' (active session)' : ''}
                </option>
              ))}
            </select>
          </div>
        )}
        {error && <div className="notice red">{error}</div>}
        {!map && !error && <p className="muted">Loading…</p>}

        {map && (
          <>
            <p className="lead">
              {map.course_name} — {map.nodes.length} concepts, {map.edges.length} relations. Columns are prerequisite
              depth; an edge never implies the student lacks the prerequisite, only that the tutor may probe it.
            </p>
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              <Pill tone={map.approved ? 'green' : map.source === 'seed' ? 'grey' : 'amber'}>
                {map.approved ? 'approved · live' : map.source === 'seed' ? 'seed pack · not reviewed' : 'awaiting review'}
              </Pill>
              <Pill tone="grey">{map.course_id}</Pill>
              {hits.length > 0 && <Pill tone="blue">{hits.length} nodes touched by the selected turn</Pill>}
            </div>

            <div className="kl-map" style={{ border: '1px solid var(--line)', borderRadius: 8, padding: 12 }}>
              <KlMapGraph map={map} hits={hits} currentNodeId={currentNodeId} />
            </div>

            <table className="kl-table">
              <thead>
                <tr>
                  <th>From</th>
                  <th>Relation</th>
                  <th>To</th>
                </tr>
              </thead>
              <tbody>
                {map.edges.map((edge, index) => {
                  const name = (id: string) => map.nodes.find((n) => n.id === id)?.name ?? id;
                  return (
                    <tr key={`${edge.from}-${edge.to}-${index}`}>
                      <td>
                        <span className="mono">{edge.from}</span> {name(edge.from)}
                      </td>
                      <td className="mono">{RELATION_LABEL[edge.relation] ?? edge.relation}</td>
                      <td>
                        <span className="mono">{edge.to}</span> {name(edge.to)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
}
