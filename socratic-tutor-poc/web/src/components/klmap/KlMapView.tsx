import type { KlMap, TurnTrace } from '../../types';
import { RELATION_LABEL } from '../../reference';
import { Pill } from '../ui';
import { KlMapGraph } from './KlMapGraph';

/**
 * Full-width course map. R18 adds approve/reject on top of this view; for now it
 * renders the map and marks what the selected turn touched.
 */
export function KlMapView({ map, error, trace }: { map: KlMap | null; error: string | null; trace: TurnTrace | null }) {
  const hits = trace?.knowledge_context?.nodes_hit ?? [];
  const currentNodeId = trace?.knowledge_context?.current_node_id;

  return (
    <div className="ingest">
      <div className="ingest-inner" style={{ maxWidth: 1180 }}>
        <h1>Knowledge map</h1>
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
