/** Shared geometry helpers for the two SVG graph renderers. No layout library. */

export function wrapLabel(text: string, maxChars: number, maxLines = 2): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = '';

  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (candidate.length <= maxChars) {
      line = candidate;
    } else {
      if (line) lines.push(line);
      line = word;
    }
    if (lines.length === maxLines) break;
  }
  if (line && lines.length < maxLines) lines.push(line);

  if (lines.length === maxLines) {
    const consumed = lines.join(' ');
    if (consumed.length < text.length) {
      const last = lines[maxLines - 1];
      lines[maxLines - 1] = `${last.slice(0, Math.max(0, maxChars - 1))}…`;
    }
  }
  return lines.length > 0 ? lines : [text];
}

/**
 * Cubic path from the right edge of one box to the left edge of another.
 *
 * `bow` is how far a backward edge dips below the boxes. It has to clear the node
 * height, or a mutual pair (A→B and B→A, the shape of a prerequisite cycle) draws
 * as one thick band instead of two arrows a reviewer can tell apart.
 */
export function edgePath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  bow = 26,
  arc = 0,
): string {
  const dx = Math.max(24, Math.abs(x2 - x1) * 0.5);
  if (x2 >= x1) {
    // `arc` dips a long forward edge into the gap between rows; without it an edge
    // spanning several columns runs straight through the boxes in between.
    return `M ${x1} ${y1} C ${x1 + dx} ${y1 + arc}, ${x2 - dx} ${y2 + arc}, ${x2} ${y2}`;
  }
  return `M ${x1} ${y1} C ${x1 + 30} ${y1 + bow}, ${x2 - 30} ${y2 + bow}, ${x2} ${y2}`;
}

/**
 * One stroke per relation, all five distinguishable. Downstream the lookup collapses
 * `related_to` / `part_of` / `uses` into a single bucket, but telling them apart is
 * exactly what a reviewer is checking, so the graph never collapses them.
 */
export const RELATION_STROKE: Record<string, { colour: string; dash?: string }> = {
  prerequisite_of: { colour: '#20714a' },
  next_topic: { colour: '#1f5fa8' },
  related_to: { colour: '#8a93a2', dash: '5 3' },
  part_of: { colour: '#6b4fa8', dash: '1 3' },
  uses: { colour: '#b26a00', dash: '2 3' },
};

/** Stable identity for an edge; the review UI addresses issues by it. */
export function edgeKey(edge: { from: string; to: string; relation: string }): string {
  return `${edge.from}|${edge.relation}|${edge.to}`;
}
