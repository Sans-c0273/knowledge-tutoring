import { useMemo, useState } from 'react';
import type { KlEdge, KlMap, KlNode } from '../../types';
import type { Severity } from '../review/issues';
import { edgeKey, edgePath, RELATION_STROKE, wrapLabel } from './graph';

const NODE_W = 122;
const NODE_H = 38;
const COL_GAP = 168;
const ROW_GAP = 54;
const PAD = 16;

/** Ordering relations decide the column; the rest are drawn as annotations. */
const ORDERING = new Set(['prerequisite_of', 'next_topic']);

const RELATION_ORDER = ['prerequisite_of', 'next_topic', 'related_to', 'part_of', 'uses'] as const;

/**
 * Edges that close a cycle, found by DFS colouring.
 *
 * A valid prerequisite graph is a DAG, but a *draft under review* is exactly where
 * a cycle shows up — and layering over a cyclic graph pushes the offending nodes
 * into ever-deeper columns until the diagram is unreadable. Back edges are left out
 * of the layering and then drawn right-to-left, which is what a cycle should look like.
 */
function backEdges(nodeIds: string[], edges: KlEdge[]): Set<string> {
  const adjacency = new Map<string, KlEdge[]>(nodeIds.map((id) => [id, []]));
  for (const edge of edges) {
    adjacency.get(edge.from)?.push(edge);
  }

  const state = new Map<string, 'open' | 'closed'>();
  const back = new Set<string>();

  for (const start of nodeIds) {
    if (state.has(start)) continue;
    // Iterative DFS: (node, index of the next outgoing edge to follow).
    const stack: { id: string; next: number }[] = [{ id: start, next: 0 }];
    state.set(start, 'open');

    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const outgoing = adjacency.get(frame.id) ?? [];
      if (frame.next >= outgoing.length) {
        state.set(frame.id, 'closed');
        stack.pop();
        continue;
      }
      const edge = outgoing[frame.next];
      frame.next += 1;
      const target = state.get(edge.to);
      if (target === 'open') {
        back.add(edgeKey(edge));
      } else if (target === undefined) {
        state.set(edge.to, 'open');
        stack.push({ id: edge.to, next: 0 });
      }
    }
  }
  return back;
}

function layerNodes(nodes: KlNode[], edges: KlEdge[]): Map<string, number> {
  const depth = new Map<string, number>(nodes.map((n) => [n.id, 0]));
  const known = new Set(depth.keys());
  const ordering = edges.filter(
    (edge) => ORDERING.has(edge.relation) && known.has(edge.from) && known.has(edge.to),
  );
  const back = backEdges([...known], ordering);
  const forward = ordering.filter((edge) => !back.has(edgeKey(edge)));

  // Longest-path layering over the acyclic remainder; the pass cap is a belt-and-braces
  // stop in case two parallel edges between the same pair slip past the back-edge scan.
  for (let pass = 0; pass < nodes.length; pass += 1) {
    let changed = false;
    for (const edge of forward) {
      const from = depth.get(edge.from) ?? 0;
      const to = depth.get(edge.to) ?? 0;
      if (to < from + 1) {
        depth.set(edge.to, from + 1);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return depth;
}

export function KlMapGraph({
  map,
  hits = [],
  currentNodeId,
  issueNodes,
  issueEdges,
  focusNodeIds = [],
  focusEdgeKeys = [],
  onSelectNode,
  defaultStructuralOnly = true,
  legend = 'turn',
}: {
  map: KlMap;
  hits?: string[];
  currentNodeId?: string;
  /** Highest severity per node id (R18 review mode). */
  issueNodes?: Map<string, Severity>;
  /** Highest severity per edge key (R18 review mode). */
  issueEdges?: Map<string, Severity>;
  /** Nodes of the selected issue — drawn emphasised. */
  focusNodeIds?: string[];
  /** Edges of the selected issue, e.g. every hop of a prerequisite cycle. */
  focusEdgeKeys?: string[];
  onSelectNode?: (nodeId: string) => void;
  defaultStructuralOnly?: boolean;
  legend?: 'turn' | 'review';
}) {
  const [structuralOnly, setStructuralOnly] = useState(defaultStructuralOnly);
  const hitSet = useMemo(() => new Set(hits), [hits]);
  const focusNodeSet = useMemo(() => new Set(focusNodeIds), [focusNodeIds]);
  const focusEdgeSet = useMemo(() => new Set(focusEdgeKeys), [focusEdgeKeys]);

  const { positions, width, height } = useMemo(() => {
    const depth = layerNodes(map.nodes, map.edges);
    const columns = new Map<number, string[]>();
    for (const node of map.nodes) {
      const column = depth.get(node.id) ?? 0;
      const bucket = columns.get(column) ?? [];
      bucket.push(node.id);
      columns.set(column, bucket);
    }

    const maxRows = Math.max(1, ...[...columns.values()].map((c) => c.length));
    const columnCount = Math.max(0, ...columns.keys()) + 1;
    const positionMap = new Map<string, { x: number; y: number }>();

    for (const [column, ids] of columns) {
      const columnHeight = ids.length * NODE_H + (ids.length - 1) * (ROW_GAP - NODE_H);
      const full = maxRows * NODE_H + (maxRows - 1) * (ROW_GAP - NODE_H);
      let y = PAD + (full - columnHeight) / 2;
      for (const id of [...ids].sort()) {
        positionMap.set(id, { x: PAD + column * COL_GAP, y });
        y += ROW_GAP;
      }
    }

    return {
      positions: positionMap,
      width: PAD * 2 + (columnCount - 1) * COL_GAP + NODE_W,
      height: PAD * 2 + maxRows * NODE_H + (maxRows - 1) * (ROW_GAP - NODE_H),
    };
  }, [map]);

  const edges = map.edges.filter((edge) => !structuralOnly || ORDERING.has(edge.relation));
  const reviewing = legend === 'review';

  return (
    <div>
      <label className="small muted" style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
        <input
          type="checkbox"
          checked={structuralOnly}
          onChange={(e) => setStructuralOnly(e.target.checked)}
        />
        Structural edges only (prerequisite_of, next_topic)
      </label>

      <div className="kl-scroll">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ minWidth: width }}
          role="img"
          aria-label={`Knowledge map for ${map.course_name}`}
        >
        <defs>
          <marker
            id="arrow-unknown"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M0 0 L10 5 L0 10 z" fill="#a83232" />
          </marker>
          {RELATION_ORDER.map((relation) => (
            <marker
              key={relation}
              id={`arrow-${relation}`}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M0 0 L10 5 L0 10 z" fill={RELATION_STROKE[relation].colour} />
            </marker>
          ))}
        </defs>

        {edges.map((edge, index) => {
          const from = positions.get(edge.from);
          const to = positions.get(edge.to);
          if (!from || !to) return null; // dangling edge; the caller lists it separately
          // A relation outside the closed vocabulary is drawn as its own thing, not as a
          // default grey line: the whole point of this screen is catching exactly that.
          const known = edge.relation in RELATION_STROKE;
          const stroke = known ? RELATION_STROKE[edge.relation] : { colour: '#a83232', dash: '2 4' };
          const key = edgeKey(edge);
          const touched = hitSet.has(edge.from) && hitSet.has(edge.to);
          const severity = issueEdges?.get(key);
          const focused = focusEdgeSet.has(key);
          // Edges spanning more than one column dip between rows instead of
          // running behind the boxes in between.
          const spansColumns = Math.abs(to.x - from.x) > COL_GAP * 1.4;
          const d = edgePath(
            from.x + NODE_W,
            from.y + NODE_H / 2,
            to.x,
            to.y + NODE_H / 2,
            NODE_H + 16,
            spansColumns ? 34 : 0,
          );

          return (
            <g key={`${key}-${index}`}>
              <title>
                {edge.from} —{edge.relation}→ {edge.to}
                {known ? '' : ' · not in the closed vocabulary'}
              </title>
              {severity && (
                <path
                  d={d}
                  fill="none"
                  stroke={severity === 'error' ? '#a83232' : '#b26a00'}
                  strokeWidth={focused ? 8 : 5}
                  strokeOpacity={focused ? 0.4 : 0.22}
                />
              )}
              <path
                d={d}
                fill="none"
                stroke={stroke.colour}
                strokeWidth={focused ? 2.4 : touched || reviewing ? 1.6 : 1}
                strokeOpacity={touched || reviewing || focused ? 1 : 0.45}
                strokeDasharray={stroke.dash}
                markerEnd={known ? `url(#arrow-${edge.relation})` : 'url(#arrow-unknown)'}
              />
            </g>
          );
        })}

        {map.nodes.map((node) => {
          const pos = positions.get(node.id);
          if (!pos) return null;
          const isCurrent = node.id === currentNodeId;
          const isHit = hitSet.has(node.id);
          const severity = issueNodes?.get(node.id);
          const focused = focusNodeSet.has(node.id);

          const fill = severity
            ? severity === 'error'
              ? '#fbe9e9'
              : '#fdf1dc'
            : isCurrent
              ? '#e8f0fa'
              : isHit
                ? '#e6f3ec'
                : '#ffffff';
          const stroke = severity
            ? severity === 'error'
              ? '#a83232'
              : '#b26a00'
            : isCurrent
              ? '#1f5fa8'
              : isHit
                ? '#20714a'
                : '#d8dde5';
          const strokeWidth = focused ? 2.6 : severity || isCurrent ? 2 : isHit ? 1.5 : 1;
          const labelLines = wrapLabel(node.name, 20);

          return (
            <g
              key={node.id}
              onClick={onSelectNode ? () => onSelectNode(node.id) : undefined}
              style={onSelectNode ? { cursor: 'pointer' } : undefined}
            >
              <title>
                {node.id} · {node.name}
                {node.name_th ? ` · ${node.name_th}` : ''}
              </title>
              {focused && (
                <rect
                  x={pos.x - 3}
                  y={pos.y - 3}
                  width={NODE_W + 6}
                  height={NODE_H + 6}
                  rx={8}
                  fill="none"
                  stroke="#a83232"
                  strokeWidth={1}
                  strokeDasharray="3 2"
                />
              )}
              <rect
                x={pos.x}
                y={pos.y}
                width={NODE_W}
                height={NODE_H}
                rx={6}
                fill={fill}
                stroke={stroke}
                strokeWidth={strokeWidth}
              />
              <text className="kl-node-label" x={pos.x + 7} y={pos.y + 12} fill="#5b6472" fontSize={8.5}>
                {node.id}
              </text>
              {labelLines.map((line, index) => (
                <text
                  key={index}
                  className="kl-node-label"
                  x={pos.x + 7}
                  y={pos.y + 24 + index * 10}
                  fontWeight={isCurrent || focused ? 700 : 400}
                >
                  {line}
                </text>
              ))}
            </g>
          );
        })}
        </svg>
      </div>

      <div className="kl-legend">
        {reviewing ? (
          <>
            <span>
              <span className="swatch" style={{ background: '#fbe9e9', border: '1px solid #a83232' }} />
              blocking error
            </span>
            <span>
              <span className="swatch" style={{ background: '#fdf1dc', border: '1px solid #b26a00' }} />
              warning
            </span>
            <span>
              <span className="swatch" style={{ background: '#a83232' }} />
              relation outside the vocabulary
            </span>
          </>
        ) : (
          <>
            <span>
              <span className="swatch" style={{ background: '#e8f0fa', border: '1px solid #1f5fa8' }} />
              current topic
            </span>
            <span>
              <span className="swatch" style={{ background: '#e6f3ec', border: '1px solid #20714a' }} />
              touched this turn
            </span>
          </>
        )}
        {RELATION_ORDER.map((relation) => (
          <span key={relation}>
            <span className="swatch" style={{ background: RELATION_STROKE[relation].colour }} />
            {relation}
          </span>
        ))}
      </div>
    </div>
  );
}
