import type { KlMap, TurnTrace } from '../../types';
import { Pill } from '../ui';
import { KnowledgeGraph } from './KnowledgeGraph';

/**
 * Third panel of the tutor view: the KL Map as it relates to the selected turn.
 * The whole-course layered map lives on its own full-width route, because at
 * panel width its labels stop being readable.
 */
export function KlMapPanel({
  map,
  error,
  trace,
  onOpenFullMap,
}: {
  map: KlMap | null;
  error: string | null;
  trace: TurnTrace | null;
  onOpenFullMap: () => void;
}) {
  const context = trace?.knowledge_context;
  const hits = new Set(context?.nodes_hit ?? []);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Knowledge map</h2>
        {map && (
          <Pill tone={map.approved ? 'green' : map.source === 'seed' ? 'grey' : 'amber'}>
            {map.approved ? 'approved' : map.source === 'seed' ? 'seed pack' : 'unapproved'}
          </Pill>
        )}
        <button type="button" className="reveal-btn" style={{ marginLeft: 'auto' }} onClick={onOpenFullMap}>
          full map
        </button>
      </div>

      <div className="panel-body" style={{ padding: '12px 14px 24px' }}>
        {error && <div className="notice red">{error}</div>}
        {!map && !error && <p className="muted small">Loading course map…</p>}

        {map && (
          <>
            <p className="small muted" style={{ marginTop: 0 }}>
              {map.course_name} · {map.nodes.length} nodes · {map.edges.length} edges
            </p>

            {context ? (
              <>
                <h3 className="section-title" style={{ marginTop: 14 }}>
                  Neighbourhood traversed this turn
                </h3>
                <KnowledgeGraph context={context} />
              </>
            ) : (
              <p className="muted small">
                Run a turn to see which nodes the lookup traversed. Nodes touched by the selected turn are highlighted
                in the table below.
              </p>
            )}

            <table className="kl-table">
              <thead>
                <tr>
                  <th>Node</th>
                  <th>Topic</th>
                  <th>ไทย</th>
                </tr>
              </thead>
              <tbody>
                {map.nodes.map((node) => (
                  <tr
                    key={node.id}
                    className={`${hits.has(node.id) ? 'hit' : ''} ${
                      node.id === context?.current_node_id ? 'current' : ''
                    }`}
                  >
                    <td className="mono">{node.id}</td>
                    <td>{node.name}</td>
                    <td lang="th">{node.name_th ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </section>
  );
}
