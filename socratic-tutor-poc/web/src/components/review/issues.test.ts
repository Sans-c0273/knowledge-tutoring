import { describe, expect, it } from 'vitest';
import type { KlMapDraft } from '../../types';
import { anchorIssues, cyclePath, danglingEdges, severityByEdge, severityByNode } from './issues';

/**
 * Anchoring is what turns a list of messages into a reviewable graph, and the cycle
 * path is read out of prose when the backend does not send it — both fail silently
 * if wrong (the issue simply stops pointing anywhere), so both are tested.
 */

const draft: KlMapDraft = {
  course_id: 'PHYS-101',
  course_name: 'Newtonian Motion',
  path: '/tmp/klmap-PHYS-101.yaml',
  extracted_at: '2026-09-01T00:00:00Z',
  status: 'draft',
  sources: [],
  nodes: [
    { id: 'C001', name: 'Displacement' },
    { id: 'C004', name: 'Acceleration' },
    { id: 'C006', name: "Newton's Second Law" },
    { id: 'C010', name: 'Momentum' },
  ],
  edges: [
    { from: 'C001', to: 'C004', relation: 'prerequisite_of' },
    { from: 'C004', to: 'C006', relation: 'prerequisite_of' },
    { from: 'C006', to: 'C004', relation: 'prerequisite_of' },
    { from: 'C011', to: 'C001', relation: 'prerequisite_of' },
  ],
  report: {
    source: '/tmp/klmap-PHYS-101.yaml',
    errors: [
      {
        code: 'KL009',
        location: 'edges',
        message:
          "'prerequisite_of' edges form a cycle: C004 (Acceleration) -> C006 (Newton's Second Law) -> C004 (Acceleration). Prerequisites must form a DAG; reverse or remove one edge in that path.",
      },
      {
        code: 'KL006',
        location: 'edges[3]',
        message: "Edge 'C011' -> 'C001' (prerequisite_of) references unknown node id 'C011'.",
      },
    ],
    warnings: [
      {
        code: 'KL101',
        location: 'nodes[3]',
        message: 'Concept C010 (Momentum) has no edges.',
      },
      { code: 'KX102', location: null, message: "Dropped proposed edge 'C012' -related_to-> 'C001'." },
    ],
  },
};

describe('cyclePath', () => {
  it('reads the path out of the loader message when the backend sends no path field', () => {
    expect(cyclePath(draft.report.errors[0])).toEqual(['C004', 'C006', 'C004']);
  });

  it('prefers a structured path when the backend does send one', () => {
    expect(
      cyclePath({ code: 'KL009', message: 'cycle: C004 (A) -> C006 (B)', path: ['C1', 'C2', 'C1'] }),
    ).toEqual(['C1', 'C2', 'C1']);
  });

  it('returns nothing rather than guessing when the message has no cycle', () => {
    expect(cyclePath({ code: 'KL006', message: 'references unknown node id' })).toEqual([]);
  });

  it('returns nothing when only one node can be read', () => {
    expect(cyclePath({ code: 'KL009', message: 'edges form a cycle: C004 (Acceleration).' })).toEqual([]);
  });
});

describe('anchorIssues', () => {
  it('anchors an indexed edge location to that edge and its endpoints', () => {
    const unknownId = anchorIssues(draft).find((issue) => issue.code === 'KL006');
    expect(unknownId?.edgeIndex).toBe(3);
    expect(unknownId?.edgeKeys).toEqual(['C011|prerequisite_of|C001']);
    expect(unknownId?.nodeIds).toEqual(['C011', 'C001']);
  });

  it('anchors an indexed node location to that node', () => {
    const isolated = anchorIssues(draft).find((issue) => issue.code === 'KL101');
    expect(isolated?.nodeIndex).toBe(3);
    expect(isolated?.nodeIds).toEqual(['C010']);
  });

  it('expands a cycle into every hop, so the whole path can be highlighted', () => {
    const cycle = anchorIssues(draft).find((issue) => issue.code === 'KL009');
    expect(cycle?.nodeIds).toEqual(['C004', 'C006']);
    expect(cycle?.edgeKeys).toEqual(['C004|prerequisite_of|C006', 'C006|prerequisite_of|C004']);
  });

  it('keeps an issue with no location, anchored to nothing', () => {
    const dropped = anchorIssues(draft).find((issue) => issue.code === 'KX102');
    expect(dropped).toBeDefined();
    expect(dropped?.nodeIds).toEqual([]);
    expect(dropped?.edgeKeys).toEqual([]);
  });

  it('marks errors and warnings distinctly and keeps errors first', () => {
    const issues = anchorIssues(draft);
    expect(issues.map((issue) => issue.severity)).toEqual(['error', 'error', 'warning', 'warning']);
  });
});

describe('severity maps', () => {
  it('lets an error win over a warning on the same node', () => {
    const issues = anchorIssues({
      ...draft,
      report: {
        source: '',
        errors: [{ code: 'KL006', location: 'nodes[1]', message: 'bad' }],
        warnings: [{ code: 'KL101', location: 'nodes[1]', message: 'isolated' }],
      },
    });
    expect(severityByNode(issues).get('C004')).toBe('error');
  });

  it('maps edge keys to their severity', () => {
    expect(severityByEdge(anchorIssues(draft)).get('C011|prerequisite_of|C001')).toBe('error');
  });
});

describe('danglingEdges', () => {
  it('finds edges the graph cannot draw, so they are not silently invisible', () => {
    expect(danglingEdges(draft.nodes, draft.edges)).toEqual([
      { from: 'C011', to: 'C001', relation: 'prerequisite_of' },
    ]);
  });

  it('finds nothing when every endpoint exists', () => {
    expect(danglingEdges(draft.nodes, draft.edges.slice(0, 3))).toEqual([]);
  });
});
