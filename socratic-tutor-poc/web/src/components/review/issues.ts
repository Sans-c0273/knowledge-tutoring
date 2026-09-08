/**
 * Anchors validation issues to the draft items they are about.
 *
 * The loader gives every issue a `location` (`edges[7]`, `nodes[2]`, or a whole
 * section like `edges`), so the review UI can point at the offending item in both
 * the graph and the table instead of leaving the reviewer to find it.
 */

import type { KlEdge, KlMapDraft, KlNode, ValidationIssue } from '../../types';
import { edgeKey } from '../klmap/graph';

export type Severity = 'error' | 'warning';

export interface AnchoredIssue extends ValidationIssue {
  severity: Severity;
  /** Index into the draft's node list, when the location names one. */
  nodeIndex?: number;
  /** Index into the draft's edge list, when the location names one. */
  edgeIndex?: number;
  /** Node ids this issue implicates — the anchored item, or a cycle's whole path. */
  nodeIds: string[];
  /** Edge keys this issue implicates, including every hop of a cycle. */
  edgeKeys: string[];
}

const LOCATION = /^(nodes|edges)\[(\d+)\]$/;

/**
 * Node ids in the cycle a KL009 issue reports.
 *
 * The backend may send `path` directly. When it does not, the ids are read out of
 * the message, which the loader formats as
 * `… cycle: C009 (Two-Step Linear Equations) -> C008 (One-Step …) -> C009 (…). Prerequisites …`.
 * Parsing prose is a fallback, not the contract: if the format changes the path is
 * simply not highlighted, and the issue still renders in full.
 */
export function cyclePath(issue: ValidationIssue): string[] {
  if (issue.path && issue.path.length > 0) return issue.path;

  const marker = issue.message.indexOf('cycle:');
  if (marker === -1) return [];

  const tail = issue.message.slice(marker + 'cycle:'.length);
  const arrowSeparated = tail.split('->');
  const ids: string[] = [];

  for (const segment of arrowSeparated) {
    const match = segment.trim().match(/^([A-Za-z][A-Za-z0-9_-]*)/);
    if (!match) break;
    ids.push(match[1]);
  }
  return ids.length >= 2 ? ids : [];
}

function edgeKeysForPath(path: string[], edges: KlEdge[]): string[] {
  const keys: string[] = [];
  for (let i = 0; i < path.length - 1; i += 1) {
    const from = path[i];
    const to = path[i + 1];
    const edge = edges.find(
      (candidate) =>
        candidate.from === from && candidate.to === to && candidate.relation === 'prerequisite_of',
    );
    if (edge) keys.push(edgeKey(edge));
  }
  return keys;
}

export function anchorIssues(draft: KlMapDraft): AnchoredIssue[] {
  const anchored: AnchoredIssue[] = [];

  const add = (issue: ValidationIssue, severity: Severity) => {
    const match = issue.location?.match(LOCATION);
    const nodeIds: string[] = [];
    const edgeKeys: string[] = [];
    let nodeIndex: number | undefined;
    let edgeIndex: number | undefined;

    if (match) {
      const index = Number(match[2]);
      if (match[1] === 'nodes') {
        nodeIndex = index;
        const node = draft.nodes[index];
        if (node) nodeIds.push(node.id);
      } else {
        edgeIndex = index;
        const edge = draft.edges[index];
        if (edge) {
          edgeKeys.push(edgeKey(edge));
          nodeIds.push(edge.from, edge.to);
        }
      }
    }

    const path = cyclePath(issue);
    if (path.length > 0) {
      nodeIds.push(...path);
      edgeKeys.push(...edgeKeysForPath(path, draft.edges));
    }

    anchored.push({
      ...issue,
      severity,
      nodeIndex,
      edgeIndex,
      nodeIds: [...new Set(nodeIds)],
      edgeKeys: [...new Set(edgeKeys)],
    });
  };

  for (const issue of draft.report.errors) add(issue, 'error');
  for (const issue of draft.report.warnings) add(issue, 'warning');
  return anchored;
}

/** Highest severity per node id, so the graph can colour a node once. */
export function severityByNode(issues: AnchoredIssue[]): Map<string, Severity> {
  const map = new Map<string, Severity>();
  for (const issue of issues) {
    for (const id of issue.nodeIds) {
      if (issue.severity === 'error' || !map.has(id)) map.set(id, issue.severity);
    }
  }
  return map;
}

/** Highest severity per edge key. */
export function severityByEdge(issues: AnchoredIssue[]): Map<string, Severity> {
  const map = new Map<string, Severity>();
  for (const issue of issues) {
    for (const key of issue.edgeKeys) {
      if (issue.severity === 'error' || !map.has(key)) map.set(key, issue.severity);
    }
  }
  return map;
}

/**
 * Edges whose endpoints are not in the node list. The graph cannot draw them, so
 * they would silently vanish from review — the one failure mode a review UI must
 * not have. The caller surfaces them explicitly instead.
 */
export function danglingEdges(nodes: KlNode[], edges: KlEdge[]): KlEdge[] {
  const known = new Set(nodes.map((node) => node.id));
  return edges.filter((edge) => !known.has(edge.from) || !known.has(edge.to));
}
