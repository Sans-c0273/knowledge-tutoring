import { useMemo, useState } from 'react';
import { useDrafts } from '../../hooks/useDrafts';
import { RELATION_LABEL } from '../../reference';
import type { KlMapDraft } from '../../types';
import { KlMapGraph } from '../klmap/KlMapGraph';
import { edgeKey } from '../klmap/graph';
import { Pill } from '../ui';
import { anchorIssues, danglingEdges, severityByEdge, severityByNode, type AnchoredIssue } from './issues';

/**
 * R18 — the human gate. An LLM-extracted map cannot serve traffic until a reviewer
 * approves it here, and a draft with blocking errors cannot be approved at all.
 * Editing happens in the YAML on disk (POC decision: no in-browser graph editor),
 * so the file path and a re-validate action are first-class controls.
 */
export function ReviewView() {
  const { drafts, selected, selectedId, select, loading, error, busy, notice, decide, revalidate, dismissNotice } =
    useDrafts();

  return (
    <div className="ingest">
      <div className="ingest-inner" style={{ maxWidth: 1180 }}>
        <h1>Knowledge map review</h1>
        <p className="lead">
          Extracted maps wait here. Nothing serves traffic until it is approved, and a draft with blocking errors
          cannot be approved — fix it in the YAML, then re-validate.
        </p>

        {loading && <p className="muted">Loading drafts…</p>}
        {error && <div className="notice red">{error}</div>}
        {notice && (
          <div className="notice green">
            {notice}{' '}
            <button type="button" className="reveal-btn" onClick={dismissNotice}>
              dismiss
            </button>
          </div>
        )}
        {!loading && drafts.length === 0 && (
          <p className="muted">
            No drafts. Ingest course content — extraction runs at the <code>kl-extract</code> stage and lands here.
          </p>
        )}

        {drafts.length > 0 && (
          <div className="draft-tabs">
            {drafts.map((draft) => (
              <button
                key={draft.course_id}
                type="button"
                className={`draft-tab${draft.course_id === selectedId ? ' active' : ''}`}
                onClick={() => select(draft.course_id)}
              >
                <span className="draft-tab-name">{draft.course_name}</span>
                <span className="draft-tab-meta">
                  <span className="mono">{draft.course_id}</span>
                  {draft.report.errors.length > 0 && <Pill tone="red">{draft.report.errors.length} errors</Pill>}
                  {draft.report.warnings.length > 0 && (
                    <Pill tone="amber">{draft.report.warnings.length} warnings</Pill>
                  )}
                  {draft.report.errors.length === 0 && draft.report.warnings.length === 0 && (
                    <Pill tone="green">clean</Pill>
                  )}
                  <StatusPill status={draft.status} />
                </span>
              </button>
            ))}
          </div>
        )}

        {selected && (
          <DraftReview
            key={selected.course_id}
            draft={selected}
            busy={busy}
            onDecide={(decision) => void decide(selected.course_id, decision)}
            onRevalidate={() => void revalidate(selected.course_id)}
          />
        )}
      </div>
    </div>
  );
}

/** A draft placed on disk by hand has no extraction sidecar, so these can be empty. */
function formatWhen(value: string | undefined): string | null {
  if (!value) return null;
  const when = new Date(value);
  return Number.isNaN(when.getTime()) ? null : when.toLocaleString();
}

function StatusPill({ status }: { status: KlMapDraft['status'] }) {
  if (status === 'approved') return <Pill tone="green">approved · live</Pill>;
  if (status === 'rejected') return <Pill tone="red">rejected</Pill>;
  return <Pill tone="grey">awaiting review</Pill>;
}

function DraftReview({
  draft,
  busy,
  onDecide,
  onRevalidate,
}: {
  draft: KlMapDraft;
  busy: boolean;
  onDecide: (decision: { approve: boolean; note?: string }) => void;
  onRevalidate: () => void;
}) {
  const [selectedIssue, setSelectedIssue] = useState<number | null>(null);
  const [note, setNote] = useState('');
  const [copied, setCopied] = useState(false);

  const issues = useMemo(() => anchorIssues(draft), [draft]);
  const nodeSeverity = useMemo(() => severityByNode(issues), [issues]);
  const edgeSeverity = useMemo(() => severityByEdge(issues), [issues]);
  const dangling = useMemo(() => danglingEdges(draft.nodes, draft.edges), [draft]);

  const focus = selectedIssue === null ? null : issues[selectedIssue];
  const blocking = draft.report.errors.length;
  // Only a recorded decision closes a draft. The backend's unreviewed status is
  // `pending`, so testing against `draft` alone would disable both controls.
  const decided = draft.status === 'approved' || draft.status === 'rejected';

  const map = {
    course_id: draft.course_id,
    course_name: draft.course_name,
    approved: false,
    source: 'draft' as const,
    nodes: draft.nodes,
    edges: draft.edges,
  };

  return (
    <>
      <div className="draft-meta">
        <dl className="kv" style={{ gridTemplateColumns: '110px 1fr' }}>
          <dt>Draft file</dt>
          <dd>
            <code>{draft.path}</code>{' '}
            <button
              type="button"
              className="reveal-btn"
              onClick={() => {
                void navigator.clipboard?.writeText(draft.path).then(
                  () => {
                    setCopied(true);
                    window.setTimeout(() => setCopied(false), 2000);
                  },
                  () => setCopied(false),
                );
              }}
            >
              {copied ? 'copied' : 'copy path'}
            </button>
          </dd>
          <dt>Extracted</dt>
          <dd>{formatWhen(draft.extracted_at) ?? 'not recorded'}</dd>
          <dt>From</dt>
          <dd className="small muted">
            {draft.sources.length > 0 ? draft.sources.join(' · ') : 'no extraction record on file'}
          </dd>
          <dt>Size</dt>
          <dd>
            {draft.nodes.length} concepts · {draft.edges.length} relations
          </dd>
          {draft.reviewed_at && (
            <>
              <dt>Reviewed</dt>
              <dd>
                {formatWhen(draft.reviewed_at) ?? draft.reviewed_at}
                {draft.review_note ? ` — ${draft.review_note}` : ''}
              </dd>
            </>
          )}
        </dl>
      </div>

      <IssueList
        issues={issues}
        selected={selectedIssue}
        onSelect={(index) => setSelectedIssue(index === selectedIssue ? null : index)}
      />

      {dangling.length > 0 && (
        <div className="notice red" style={{ marginTop: 12 }}>
          <strong>{dangling.length} edge(s) cannot be drawn</strong> because they reference a concept that is not in
          the node list ({dangling.map((edge) => `${edge.from} → ${edge.to}`).join(', ')}). They are in the edge table
          below, marked. Fix the id or add the concept.
        </div>
      )}

      <div className="kl-map" style={{ border: '1px solid var(--line)', borderRadius: 8, padding: 12, marginTop: 14 }}>
        <KlMapGraph
          map={map}
          issueNodes={nodeSeverity}
          issueEdges={edgeSeverity}
          focusNodeIds={focus?.nodeIds ?? []}
          focusEdgeKeys={focus?.edgeKeys ?? []}
          defaultStructuralOnly={false}
          legend="review"
        />
      </div>

      <h3 className="section-title" style={{ marginTop: 20 }}>
        Concepts
      </h3>
      <table className="kl-table">
        <thead>
          <tr>
            <th style={{ width: 30 }} />
            <th>Node</th>
            <th>Concept</th>
            <th>ไทย</th>
          </tr>
        </thead>
        <tbody>
          {draft.nodes.map((node, index) => {
            const severity = nodeSeverity.get(node.id);
            const focused = focus?.nodeIndex === index || focus?.nodeIds.includes(node.id);
            return (
              <tr key={node.id} className={`${severity ?? ''}${focused ? ' focused' : ''}`}>
                <td className="mono muted">{index}</td>
                <td className="mono">{node.id}</td>
                <td>{node.name}</td>
                <td lang="th">{node.name_th ?? '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <h3 className="section-title" style={{ marginTop: 18 }}>
        Relations
      </h3>
      <table className="kl-table">
        <thead>
          <tr>
            <th style={{ width: 30 }} />
            <th>From</th>
            <th>Relation</th>
            <th>To</th>
          </tr>
        </thead>
        <tbody>
          {draft.edges.map((edge, index) => {
            const key = edgeKey(edge);
            const severity = edgeSeverity.get(key);
            const focused = focus?.edgeIndex === index || focus?.edgeKeys.includes(key);
            const isDangling = dangling.includes(edge);
            const name = (id: string) => draft.nodes.find((node) => node.id === id)?.name;
            return (
              <tr key={`${key}-${index}`} className={`${severity ?? ''}${focused ? ' focused' : ''}`}>
                <td className="mono muted">{index}</td>
                <td>
                  <span className="mono">{edge.from}</span> {name(edge.from) ?? <em className="danger-text">unknown concept</em>}
                </td>
                <td className="mono">{RELATION_LABEL[edge.relation] ?? edge.relation}</td>
                <td>
                  <span className="mono">{edge.to}</span> {name(edge.to) ?? <em className="danger-text">unknown concept</em>}
                  {isDangling && <> <Pill tone="red">not drawn</Pill></>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="review-actions">
        <div>
          <button type="button" className="btn" disabled={busy} onClick={onRevalidate}>
            Re-validate from disk
          </button>
          <span className="small muted" style={{ marginLeft: 10 }}>
            Edit the YAML at the path above, then re-read it.
          </span>
        </div>

        <div className="review-decide">
          <input
            type="text"
            placeholder="Review note (optional)"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            aria-label="Review note"
          />
          <button
            type="button"
            className="btn"
            disabled={busy || decided}
            onClick={() => onDecide({ approve: false, note: note.trim() || undefined })}
          >
            Reject
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={busy || decided || blocking > 0}
            title={
              blocking > 0
                ? `Cannot approve: ${blocking} blocking error(s) must be fixed in the YAML first.`
                : undefined
            }
            onClick={() => onDecide({ approve: true, note: note.trim() || undefined })}
          >
            Approve — make live
          </button>
        </div>

        {blocking > 0 && (
          <div className="notice red" style={{ marginTop: 10 }}>
            Approval is blocked: {blocking} error{blocking === 1 ? '' : 's'} above must be fixed in{' '}
            <code>{draft.path.split('/').pop()}</code> first. Warnings do not block.
          </div>
        )}
        {blocking === 0 && draft.report.warnings.length > 0 && !decided && (
          <div className="notice amber" style={{ marginTop: 10 }}>
            {draft.report.warnings.length} warning{draft.report.warnings.length === 1 ? '' : 's'} — approvable, but
            read them first: an isolated concept is invisible to every lookup.
          </div>
        )}
        {decided && (
          <div className={`notice ${draft.status === 'approved' ? 'green' : 'red'}`} style={{ marginTop: 10 }}>
            This draft was {draft.status}
            {draft.reviewed_at ? ` on ${formatWhen(draft.reviewed_at) ?? draft.reviewed_at}` : ''}. Re-extract or edit the
            YAML to review it again.
          </div>
        )}
      </div>
    </>
  );
}

function IssueList({
  issues,
  selected,
  onSelect,
}: {
  issues: AnchoredIssue[];
  selected: number | null;
  onSelect: (index: number) => void;
}) {
  if (issues.length === 0) {
    return (
      <div className="notice green" style={{ marginTop: 14 }}>
        No validation issues. The map satisfies the closed vocabulary, single-direction and prerequisite-DAG rules.
      </div>
    );
  }

  return (
    <div className="issue-list">
      {issues.map((issue, index) => (
        <button
          key={`${issue.code}-${issue.location ?? 'none'}-${index}`}
          type="button"
          className={`issue ${issue.severity}${selected === index ? ' selected' : ''}`}
          onClick={() => onSelect(index)}
          aria-pressed={selected === index}
        >
          <span className="issue-head">
            <span className="issue-code mono">{issue.code}</span>
            <Pill tone={issue.severity === 'error' ? 'red' : 'amber'}>
              {issue.severity === 'error' ? 'blocking' : 'warning'}
            </Pill>
            {issue.location && <span className="issue-where mono">{issue.location}</span>}
            {issue.edgeKeys.length > 1 && <span className="small muted">{issue.edgeKeys.length}-edge path</span>}
          </span>
          <span className="issue-message">{issue.message}</span>
        </button>
      ))}
    </div>
  );
}
