import type { KnowledgeContext, KnowledgeNodeRef } from '../../types';
import { edgePath, wrapLabel } from './graph';

const NODE_W = 86;
const NODE_H = 32;
const V_GAP = 10;
const COL_X = [4, 104, 204, 304];
const VIEW_W = COL_X[3] + NODE_W + 4;
const RELATED_GAP = 28;

type Column = { x: number; nodes: KnowledgeNodeRef[] };

/**
 * The turn's KL Map neighbourhood: prerequisites to the left, current topic in the
 * middle, next topics to the right, related concepts below. Only nodes the BFS
 * actually returned are drawn, so everything on screen was touched this turn.
 */
export function KnowledgeGraph({ context }: { context: KnowledgeContext }) {
  const farPrereqs = context.prerequisites.filter((p) => p.distance > 1);
  const nearPrereqs = context.prerequisites.filter((p) => p.distance <= 1);
  const current: KnowledgeNodeRef = {
    topic: context.current_topic,
    node_id: context.current_node_id,
    distance: 0,
  };

  const columns: Column[] = [
    { x: COL_X[0], nodes: farPrereqs },
    { x: COL_X[1], nodes: nearPrereqs },
    { x: COL_X[2], nodes: [current] },
    { x: COL_X[3], nodes: context.next_topics },
  ];

  const related = context.related_topics;
  const stackHeight = (count: number) => (count === 0 ? 0 : count * NODE_H + (count - 1) * V_GAP);
  const centreColumnHeight =
    NODE_H + (related.length > 0 ? RELATED_GAP + stackHeight(related.length) : 0);
  const height =
    Math.max(
      stackHeight(farPrereqs.length),
      stackHeight(nearPrereqs.length),
      centreColumnHeight,
      stackHeight(context.next_topics.length),
    ) + 22; // room for the distance badge above the top row

  const positions = new Map<string, { x: number; y: number }>();
  const key = (ref: KnowledgeNodeRef, fallback: string) => ref.node_id ?? `${fallback}:${ref.topic}`;

  columns.forEach((column, columnIndex) => {
    const columnHeight = columnIndex === 2 ? centreColumnHeight : stackHeight(column.nodes.length);
    let y = (height - columnHeight) / 2;
    column.nodes.forEach((node) => {
      positions.set(key(node, `c${columnIndex}`), { x: column.x, y });
      y += NODE_H + V_GAP;
    });
  });

  const currentPos = positions.get(key(current, 'c2'))!;
  related.forEach((node, index) => {
    positions.set(key(node, 'rel'), {
      x: COL_X[2],
      y: currentPos.y + NODE_H + RELATED_GAP + index * (NODE_H + V_GAP),
    });
  });

  const lines: { d: string; colour: string; dash?: string }[] = [];

  // Near prerequisites → current topic.
  for (const node of nearPrereqs) {
    const from = positions.get(key(node, 'c1'))!;
    lines.push({
      d: edgePath(from.x + NODE_W, from.y + NODE_H / 2, currentPos.x, currentPos.y + NODE_H / 2),
      colour: '#20714a',
    });
  }

  // Distance-2 prerequisites attach to the single distance-1 node when there is
  // exactly one; otherwise they are drawn against the current topic as depth-2 hits.
  for (const node of farPrereqs) {
    const from = positions.get(key(node, 'c0'))!;
    const target =
      nearPrereqs.length === 1 ? positions.get(key(nearPrereqs[0], 'c1'))! : currentPos;
    lines.push({
      d: edgePath(from.x + NODE_W, from.y + NODE_H / 2, target.x, target.y + NODE_H / 2),
      colour: '#20714a',
      dash: '4 3',
    });
  }

  for (const node of context.next_topics) {
    const to = positions.get(key(node, 'c3'))!;
    lines.push({
      d: edgePath(currentPos.x + NODE_W, currentPos.y + NODE_H / 2, to.x, to.y + NODE_H / 2),
      colour: '#1f5fa8',
    });
  }

  for (const node of related) {
    const to = positions.get(key(node, 'rel'))!;
    lines.push({
      d: `M ${currentPos.x + NODE_W / 2} ${currentPos.y + NODE_H} L ${to.x + NODE_W / 2} ${to.y}`,
      colour: '#8a93a2',
      dash: '4 3',
    });
  }

  const renderNode = (node: KnowledgeNodeRef, fallback: string, kind: 'current' | 'prereq' | 'next' | 'related') => {
    const pos = positions.get(key(node, fallback));
    if (!pos) return null;
    const style = {
      current: { fill: '#e8f0fa', stroke: '#1f5fa8', width: 1.8 },
      prereq: { fill: '#e6f3ec', stroke: '#20714a', width: 1.2 },
      next: { fill: '#f4f6f9', stroke: '#5b6472', width: 1.1 },
      related: { fill: '#f4f6f9', stroke: '#8a93a2', width: 1.1 },
    }[kind];
    const lines2 = wrapLabel(node.topic, 16);

    return (
      <g key={`${kind}-${key(node, fallback)}`}>
        <title>
          {node.node_id ? `${node.node_id} · ` : ''}
          {node.topic}
          {node.topic_th ? ` · ${node.topic_th}` : ''}
          {node.distance > 0 ? ` (distance ${node.distance})` : ''}
        </title>
        <rect
          x={pos.x}
          y={pos.y}
          width={NODE_W}
          height={NODE_H}
          rx={5}
          fill={style.fill}
          stroke={style.stroke}
          strokeWidth={style.width}
        />
        {lines2.map((line, index) => (
          <text
            key={index}
            className="kl-node-label"
            x={pos.x + NODE_W / 2}
            y={pos.y + NODE_H / 2 + (lines2.length === 1 ? 3 : index === 0 ? -2 : 9)}
            textAnchor="middle"
          >
            {line}
          </text>
        ))}
        {node.distance > 0 && (
          <text
            className="kl-node-label"
            x={pos.x + NODE_W - 2}
            y={pos.y - 3}
            textAnchor="end"
            fill="#5b6472"
            fontSize={8}
          >
            d{node.distance}
          </text>
        )}
      </g>
    );
  };

  return (
    <div className="kl-inline">
      <svg viewBox={`0 0 ${VIEW_W} ${height}`} role="img" aria-label="Knowledge map neighbourhood for this turn">
        {lines.map((line, index) => (
          <path
            key={index}
            d={line.d}
            fill="none"
            stroke={line.colour}
            strokeWidth={1.2}
            strokeDasharray={line.dash}
          />
        ))}
        {farPrereqs.map((node) => renderNode(node, 'c0', 'prereq'))}
        {nearPrereqs.map((node) => renderNode(node, 'c1', 'prereq'))}
        {renderNode(current, 'c2', 'current')}
        {context.next_topics.map((node) => renderNode(node, 'c3', 'next'))}
        {related.map((node) => renderNode(node, 'rel', 'related'))}
      </svg>
      <div className="kl-legend">
        <span>
          <span className="swatch" style={{ background: '#e8f0fa', border: '1px solid #1f5fa8' }} />
          current topic
        </span>
        <span>
          <span className="swatch" style={{ background: '#e6f3ec', border: '1px solid #20714a' }} />
          prerequisite
        </span>
        <span>
          <span className="swatch" style={{ background: '#f4f6f9', border: '1px solid #5b6472' }} />
          next
        </span>
        <span>
          <span className="swatch" style={{ background: '#f4f6f9', border: '1px solid #8a93a2' }} />
          related
        </span>
        <span>BFS depth {context.max_depth}</span>
      </div>
    </div>
  );
}
