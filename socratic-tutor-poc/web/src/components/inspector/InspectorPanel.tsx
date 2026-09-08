import { Fragment } from 'react';
import {
  CONFIDENCE_THRESHOLD,
  GUARDRAIL_MEANING,
  INTENT_LABEL,
  KM_RULES,
  KNOWLEDGE_ACTION_LABEL,
  PRE_STREAM_BUDGET_MS,
  strategyName,
  STRATEGY_LIBRARY,
} from '../../reference';
import type {
  GuardrailEvent,
  PipelineStep,
  StrategyId,
  StrategyRef,
  TriggerMatch,
  TurnTrace,
} from '../../types';
import type { ReactNode } from 'react';
import { Bar, Mono, Pill, type Tone } from '../ui';
import { KnowledgeGraph } from '../klmap/KnowledgeGraph';
import { Section } from './Section';
import { summarise } from './summary';

/**
 * Did this pipeline step run? The backend omits a field it has no value for, so an
 * absent `retrieval` could mean "no chunks matched" or "never got that far" — very
 * different facts. `timings` records every step that executed, so it settles it.
 */
function stepRan(trace: TurnTrace, step: PipelineStep): boolean {
  return Boolean(trace.timings?.some((timing) => timing.step === step));
}

/** Real turns run to tens of seconds, where raw milliseconds stop being readable. */
function formatMs(ms: number): string {
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

function EmptyStep({ children }: { children: ReactNode }) {
  return <div className="notice small">{children}</div>;
}

/** R20 — the glass box. Every teaching decision in the turn, in pipeline order. */
export function InspectorPanel({
  trace,
  streaming,
  composing = false,
}: {
  trace: TurnTrace | null;
  streaming: boolean;
  composing?: boolean;
}) {
  if (!trace) {
    return (
      <section className="panel">
        <div className="panel-head">
          <h2>Glass-box inspector</h2>
        </div>
        <div className="panel-body">
          <div className="inspector-empty">
            No turn selected. Send a message, or click any tutor reply, to see the TurnTrace that produced it:
            detected intent, matched rule rows, policy constraints, knowledge-map traversal, response plan, guardrail
            events and per-step latency.
          </div>
        </div>
      </section>
    );
  }

  const summary = summarise(trace);
  const showEvaluation = Boolean(trace.answer_evaluation) || trace.intent?.intent === 'check_answer';

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Glass-box inspector</h2>
        <span className="mono-value">{trace.turn_id}</span>
        {streaming && <Pill tone="amber">{composing ? 'generating' : 'live'}</Pill>}
      </div>

      <div className="panel-body">
        {composing && (
          <div className="composing-banner">
            <span className="dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            Generation is running. Every teaching decision is already made and shown below; the
            guardrail scans the draft before a word of it reaches the student.
          </div>
        )}
        <div className="summary-strip">
          <h3>What I understood</h3>
          <p className="summary-line" lang="en">
            {summary.understood}
          </p>
          <h3>What I intend to do</h3>
          <p className="summary-line" lang="en">
            {summary.intend}
          </p>
          {summary.flags.length > 0 && (
            <div className="summary-flags">
              {summary.flags.map((flag) => (
                <Pill key={flag.label} tone={flag.tone} title={flag.title}>
                  {flag.label}
                </Pill>
              ))}
            </div>
          )}
        </div>

        {trace.error && (
          <div className="notice red" style={{ margin: '12px 14px 0' }}>
            <strong>This turn aborted.</strong> {trace.error} Everything the pipeline reached before
            the failure is below.
          </div>
        )}

        <IntentSection trace={trace} />
        <TopicSection trace={trace} />
        <RetrievalSection trace={trace} />
        <KnowledgeSection trace={trace} />
        {showEvaluation && <EvaluationSection trace={trace} />}
        <StrategySection trace={trace} />
        <PlanSection trace={trace} />
        <GuardrailSection trace={trace} />
        <TimingSection trace={trace} />
        <VersionsSection trace={trace} />
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------

function StrategyPill({ strategy, tone = 'blue' }: { strategy: StrategyRef; tone?: Tone }) {
  const entry = STRATEGY_LIBRARY[strategy.strategy_id];
  return (
    <Pill tone={tone} title={entry?.behavior}>
      {strategy.strategy_id} · {entry?.name ?? strategy.strategy_name}
    </Pill>
  );
}

function IntentSection({ trace }: { trace: TurnTrace }) {
  const intent = trace.intent;
  return (
    <Section
      title="Intent"
      engine="llm"
      ready={Boolean(intent)}
      pendingLabel="Call A running…"
      hint={intent ? `${intent.intent} · ${(intent.confidence * 100).toFixed(0)}%` : undefined}
    >
      {intent && (
        <>
          <dl className="kv">
            <dt>Intent</dt>
            <dd>
              <Pill tone="amber">{intent.intent}</Pill>{' '}
              <span className="muted small">{INTENT_LABEL[intent.intent]}</span>
            </dd>
            <dt>Learner state</dt>
            <dd>
              <Pill tone={intent.learner_state === 'confused' ? 'amber' : 'grey'}>{intent.learner_state}</Pill>
            </dd>
            <dt>Special handling</dt>
            <dd>
              <Pill tone={intent.special_handling.type === 'none' ? 'grey' : 'red'}>
                {intent.special_handling.type}
              </Pill>{' '}
              <span className="muted small">detected: {String(intent.special_handling.detected)}</span>
            </dd>
          </dl>

          <div>
            <div className="confidence-row">
              <span className="small muted">Confidence</span>
              <Bar
                value={intent.confidence * 100}
                tone={intent.confidence < CONFIDENCE_THRESHOLD ? 'red' : 'green'}
              />
              <Mono>{intent.confidence.toFixed(2)}</Mono>
            </div>
            <div className="small muted" style={{ marginTop: 3 }}>
              Fallback threshold {CONFIDENCE_THRESHOLD.toFixed(2)}
            </div>
          </div>

          {intent.fallback_applied && (
            <div className="notice amber">
              <strong>Low-confidence fallback.</strong> {intent.fallback_reason}
            </div>
          )}
        </>
      )}
    </Section>
  );
}

function TopicSection({ trace }: { trace: TurnTrace }) {
  const topic = trace.topic;
  const ran = stepRan(trace, 'topic_level');
  return (
    <Section
      title="Topic & level"
      engine="deterministic"
      ready={Boolean(topic) || ran}
      hint={topic ? topic.student_level : ran ? 'no match' : undefined}
    >
      {!topic && ran && (
        <EmptyStep>
          Topic resolution ran and matched nothing in the syllabus, so the turn carried no topic
          and no student level. Everything downstream decided without them.
        </EmptyStep>
      )}
      {topic && (
        <dl className="kv">
          <dt>Syllabus topic</dt>
          <dd>
            {topic.topic}
            {topic.node_id && (
              <>
                {' '}
                <Mono>{topic.node_id}</Mono>
              </>
            )}
          </dd>
          {topic.topic_th && (
            <>
              <dt>ไทย</dt>
              <dd lang="th">{topic.topic_th}</dd>
            </>
          )}
          <dt>Student level</dt>
          <dd>
            <Pill tone="blue">{topic.student_level}</Pill>{' '}
            <span className="muted small">{topic.level_source.replace(/_/g, ' ')}</span>
          </dd>
          <dt>Matched by</dt>
          <dd className="muted small">
            {topic.matched_by}
            {topic.confidence !== undefined && ` · ${topic.confidence.toFixed(2)}`}
            {topic.resolved_by_context && ' · resolved from the previous turn'}
          </dd>
          {topic.ambiguous && (
            <>
              <dt>Ambiguous</dt>
              <dd>
                <Pill tone="amber">more than one topic matched</Pill>
                {topic.candidates && topic.candidates.length > 0 && (
                  <div className="small muted">also matched: {topic.candidates.join(', ')}</div>
                )}
              </dd>
            </>
          )}
        </dl>
      )}
    </Section>
  );
}

function RetrievalSection({ trace }: { trace: TurnTrace }) {
  const chunks = trace.retrieval;
  const ran = stepRan(trace, 'retrieval');
  return (
    <Section
      title="Retrieval"
      engine="data"
      defaultOpen={false}
      ready={Boolean(chunks) || ran}
      hint={chunks ? `${chunks.length} chunk${chunks.length === 1 ? '' : 's'}` : ran ? '0 chunks' : undefined}
    >
      {!chunks && ran && (
        <div className="notice red">
          Retrieval ran and returned nothing. Teaching Policy row 4 prohibits factual claims on a
          turn with no evidence — check that the course content has actually been ingested.
        </div>
      )}
      {chunks && chunks.length === 0 && (
        <div className="notice red">
          No evidence retrieved. Teaching Policy row 4 prohibits factual claims for this turn; the planner must route
          to a check or scaffold and the response must state the limitation.
        </div>
      )}
      {chunks?.map((chunk) => (
        <div className="chunk" key={chunk.chunk_id}>
          <div className="chunk-head">
            <Mono>{chunk.chunk_id}</Mono>
            <span>{chunk.source_ref}</span>
            <span className="chunk-score">{chunk.score.toFixed(2)}</span>
          </div>
          <div className="chunk-text" lang={chunk.language}>
            {chunk.text}
          </div>
        </div>
      ))}
    </Section>
  );
}

function KnowledgeSection({ trace }: { trace: TurnTrace }) {
  const context = trace.knowledge_context;
  const ran = stepRan(trace, 'kl_lookup');
  return (
    <Section
      title="Knowledge map"
      engine="data"
      ready={Boolean(context) || ran}
      hint={
        context ? `${context.nodes_hit.length} nodes · depth ${context.max_depth}` : ran ? 'no nodes' : undefined
      }
    >
      {!context && ran && (
        <EmptyStep>
          The lookup ran and returned no knowledge context — with no resolved topic there is no
          start node, so no prerequisite or related concept informed this turn.
        </EmptyStep>
      )}
      {context && (
        <>
          <KnowledgeGraph context={context} />

          <dl className="kv">
            <dt>Prerequisites</dt>
            <dd>
              {context.prerequisites.length === 0
                ? '—'
                : context.prerequisites.map((p) => `${p.topic} (d${p.distance})`).join(', ')}
            </dd>
            <dt>Related</dt>
            <dd>
              {context.related_topics.length === 0
                ? '—'
                : context.related_topics.map((p) => `${p.topic} (d${p.distance})`).join(', ')}
            </dd>
            <dt>Next</dt>
            <dd>
              {context.next_topics.length === 0
                ? '—'
                : context.next_topics.map((p) => `${p.topic} (d${p.distance})`).join(', ')}
            </dd>
            {context.relationships.length > 0 && (
              <>
                <dt>Relationships</dt>
                <dd className="small">{context.relationships.join(' · ')}</dd>
              </>
            )}
          </dl>

          {trace.km_rules && trace.km_rules.length > 0 && (
            <div>
              <h4 className="section-title" style={{ marginTop: 4 }}>
                KM rules
              </h4>
              {trace.km_rules.map((rule) => {
                const reference = KM_RULES[rule.rule];
                const condition = rule.condition ?? reference?.condition;
                const action = rule.action ?? reference?.action;
                return (
                  <div key={rule.rule} className="trigger-row" style={{ marginBottom: 6 }}>
                    <div className="trigger-head">
                      <span className="rowid">{rule.rule}</span>
                      <Pill tone={rule.applied ? 'green' : 'grey'}>
                        {rule.applied ? 'applied' : 'not applied'}
                      </Pill>
                    </div>
                    {condition && <div className="small muted">{condition}</div>}
                    {action && <div className="small">{action}</div>}
                    {rule.note && <div className="small muted">{rule.note}</div>}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </Section>
  );
}

function EvaluationSection({ trace }: { trace: TurnTrace }) {
  const evaluation = trace.answer_evaluation;
  const tone: Tone = !evaluation
    ? 'grey'
    : evaluation.evaluation === 'correct'
      ? 'green'
      : evaluation.evaluation === 'cannot_evaluate'
        ? 'grey'
        : 'amber';

  return (
    <Section
      title="Answer evaluation"
      engine="llm"
      ready={Boolean(evaluation)}
      pendingLabel="Call B running (solve silently, then compare)…"
      hint={evaluation ? evaluation.evaluation.replace(/_/g, ' ') : undefined}
    >
      {evaluation && (
        <>
          <dl className="kv">
            <dt>Verdict</dt>
            <dd>
              <Pill tone={tone}>{evaluation.evaluation}</Pill>
            </dd>
            <dt>Decided by</dt>
            <dd>
              {evaluation.deterministic_checker_used ? (
                <Pill tone="green" title="A deterministic check cannot be argued out of its answer.">
                  deterministic checker
                </Pill>
              ) : (
                <Pill tone="amber">LLM comparison only</Pill>
              )}
              {evaluation.source && <span className="muted small"> · {evaluation.source}</span>}
            </dd>
            {evaluation.error_locus && (
              <>
                <dt>Error locus</dt>
                <dd>{evaluation.error_locus}</dd>
              </>
            )}
          </dl>
          {evaluation.checker_note && <div className="small muted">{evaluation.checker_note}</div>}
          {evaluation.deterministic_checker_used ? (
            <div className="notice green small">
              A deterministic check produced this verdict, so no amount of student push-back can move it. Where the
              domain allows one, this is the strongest guarantee in the pipeline.
            </div>
          ) : (
            <div className="notice blue small">
              Isolation rule: this call saw only the problem, the evidence and the attempt text — never the student's
              framing and never the running dialogue.
            </div>
          )}
        </>
      )}
    </Section>
  );
}

function StrategySection({ trace }: { trace: TurnTrace }) {
  const strategy = trace.strategy;
  const policy = trace.teaching_policy;

  return (
    <Section
      title="Strategy"
      engine="deterministic"
      ready={Boolean(strategy)}
      hint={
        strategy ? (
          <>
            {strategy.primary_strategy.strategy_id}
            {policy?.override_applied ? ' · overridden' : ''}
          </>
        ) : undefined
      }
    >
      {strategy && (
        <>
          {trace.trigger_matches && trace.trigger_matches.length > 0 && (
            <div>
              <h4 className="section-title" style={{ marginTop: 0 }}>
                Trigger matrix
              </h4>
              {trace.trigger_matches.map((row) => (
                <div
                  key={row.row_id}
                  className={`trigger-row ${row.selected ? 'selected' : row.matched ? '' : 'excluded'}`}
                  style={{ marginBottom: 6 }}
                >
                  <div className="trigger-head">
                    <span className="rowid">{row.row_id}</span>
                    {row.condition && <span>{row.condition}</span>}
                    {row.key &&
                      Object.entries(row.key)
                        .filter(([, value]) => value !== null && value !== undefined && value !== 'any')
                        .map(([field, value]) => (
                          <Pill key={field} tone={field === 'special_handling' ? 'red' : 'grey'} title={field}>
                            {value}
                          </Pill>
                        ))}
                    {row.special_handling && row.special_handling !== 'none' && (
                      <Pill tone="red">{row.special_handling}</Pill>
                    )}
                    <span className="priority">
                      {row.priority !== undefined && `priority ${row.priority}`}
                      {row.specificity !== undefined && ` · specificity ${row.specificity}`}
                    </span>
                  </div>
                  {(row.primary || (row.supporting?.length ?? 0) > 0) && (
                    <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', margin: '4px 0' }}>
                      {row.primary && (
                        <StrategyPill strategy={row.primary} tone={row.selected ? 'blue' : 'grey'} />
                      )}
                      {row.supporting?.map((support) => (
                        <StrategyPill key={support.strategy_id} strategy={support} tone="grey" />
                      ))}
                    </div>
                  )}
                  {row.selected && <div className="small" style={{ color: 'var(--blue)' }}>selected</div>}
                  {row.lost_to && <LostOn row={row} />}
                  {row.note && !row.excluded_reason && <div className="small muted">{row.note}</div>}
                  {row.excluded_reason && <div className="small muted">{row.excluded_reason}</div>}
                </div>
              ))}
            </div>
          )}

          {trace.combination && (
            <div>
              <h4 className="section-title">Combination rules</h4>
              <div className="small">
                <Pill tone={trace.combination.allowed ? 'green' : 'red'}>
                  {trace.combination.allowed ? 'allowed' : 'rejected'}
                </Pill>{' '}
                {trace.combination.rule}
              </div>
              {trace.combination.sequence && trace.combination.sequence.length > 0 && (
                <div className="structure-seq" style={{ marginTop: 6 }}>
                  {trace.combination.sequence.map((step, index) => (
                    <span key={`${step}-${index}`} style={{ display: 'inline-flex', gap: 5, alignItems: 'center' }}>
                      {index > 0 && <span className="arrow">→</span>}
                      <Pill tone="grey">{step}</Pill>
                    </span>
                  ))}
                </div>
              )}
              {trace.combination.dropped.length > 0 && (
                <div className="small muted" style={{ marginTop: 4 }}>
                  Dropped:{' '}
                  {trace.combination.dropped
                    .map((d) => (typeof d === 'string' ? `${d} ${strategyName(d as StrategyId)}` : `${d.strategy_id} ${d.strategy_name}`))
                    .join(', ')}
                </div>
              )}
              {trace.combination.note && (
                <div className="small muted" style={{ marginTop: 4 }}>
                  {trace.combination.note}
                </div>
              )}
            </div>
          )}

          {policy && (
            // The card is prominent whenever the policy is *constraining* the turn, not
            // only when it overrode the Trigger Matrix: a row that forbids S01 shaped this
            // turn even if the matrix never proposed S01.
            <div
              className={`policy-card${
                policy.override_applied || (policy.prohibited_strategies?.length ?? 0) > 0 ? ' override' : ''
              }`}
            >
              <h4>
                Teaching policy · {policy.row_label ?? policy.context ?? policy.row}
                <span className="mono-value">{policy.row}</span>
                {policy.override_applied ? (
                  <Pill tone="red">override applied</Pill>
                ) : (policy.prohibited_strategies?.length ?? 0) > 0 ? (
                  <Pill tone="red">constraints applied</Pill>
                ) : null}
              </h4>
              <dl className="kv" style={{ gridTemplateColumns: '108px 1fr' }}>
                <dt>Direct answer</dt>
                <dd>{policy.direct_answer.replace(/_/g, ' ')}</dd>
                <dt>Attempt</dt>
                <dd>{policy.attempt.replace(/_/g, ' ')}</dd>
                <dt>Check</dt>
                <dd>{policy.check.replace(/_/g, ' ')}</dd>
              </dl>
              {policy.prohibited_strategies && policy.prohibited_strategies.length > 0 && (
                <p className="small" style={{ margin: '8px 0 0' }}>
                  Forbidden on this turn:{' '}
                  {policy.prohibited_strategies
                    .map((id) => `${id} ${strategyName(id as StrategyId)}`)
                    .join(', ')}
                </p>
              )}
              {[policy.override_note, ...(policy.override_notes ?? [])]
                .filter((note): note is string => Boolean(note))
                .map((note) => (
                  <p key={note} className="small" style={{ margin: '8px 0 0' }}>
                    {note}
                  </p>
                ))}
              {policy.also_applied && policy.also_applied.length > 0 && (
                <p className="small muted" style={{ margin: '6px 0 0' }}>
                  Rows that also matched: {policy.also_applied.join(', ')}
                </p>
              )}
              {policy.precedence !== undefined && (
                <p className="small muted" style={{ margin: '6px 0 0' }}>
                  {typeof policy.precedence === 'number'
                    ? `Policy priority ${policy.precedence} · Teaching Policy → Special Handling → Learner State → Intent → Student Level`
                    : `Precedence: ${policy.precedence}`}
                </p>
              )}
            </div>
          )}

          <div>
            <h4 className="section-title">Selected strategies</h4>
            <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
              <StrategyPill strategy={strategy.primary_strategy} tone="blue" />
              {strategy.supporting_strategies.map((support) => (
                <StrategyPill key={support.strategy_id} strategy={support} tone="grey" />
              ))}
            </div>
          </div>

          <div className="constraint-grid">
            <Constraint label="direct_answer_allowed" value={strategy.constraints.direct_answer_allowed} />
            <Constraint label="full_solution_allowed_now" value={strategy.constraints.full_solution_allowed_now} />
            <Constraint label="student_attempt_required" value={strategy.constraints.student_attempt_required} invert />
          </div>

          {strategy.knowledge_guidance && (
            <dl className="kv">
              <dt>Knowledge guidance</dt>
              <dd>
                {KNOWLEDGE_ACTION_LABEL[strategy.knowledge_guidance.action] ?? strategy.knowledge_guidance.action}
                <div className="small muted">
                  <Mono>{strategy.knowledge_guidance.action}</Mono> → {strategy.knowledge_guidance.target_concept} (
                  {strategy.knowledge_guidance.relationship}
                  {strategy.knowledge_guidance.direction === 'inverse' && ', read inverse'}, d
                  {strategy.knowledge_guidance.distance})
                </div>
              </dd>
            </dl>
          )}
        </>
      )}
    </Section>
  );
}

/**
 * How a row lost. A tie broken on the row id means no deliberate rule decided this
 * turn's strategy — alphabetical order did — which is the case a reviewer should see
 * rather than have to infer from two equal priorities.
 */
function LostOn({ row }: { row: TriggerMatch }) {
  const basis = row.lost_on ?? '';
  const undecided = /alphabet|row_?id|tie/i.test(basis);
  return (
    <div className={`small${undecided ? ' notice amber' : ' muted'}`} style={{ marginTop: 4 }}>
      Lost to <span className="mono">{row.lost_to}</span>
      {basis && <> on {basis.replace(/_/g, ' ')}</>}
      {undecided && (
        <>
          {' '}
          — the rows tied on every deliberate criterion, so the row id broke it. Nothing chose this
          outcome; if the other row is the right one, give it a higher priority.
        </>
      )}
    </div>
  );
}

function Constraint({ label, value, invert = false }: { label: string; value: boolean; invert?: boolean }) {
  const positive = invert ? !value : value;
  return (
    <div className="constraint">
      <span className={`state ${positive ? 'on' : 'off'}`}>{value ? 'true' : 'false'}</span>
      <span>{label}</span>
    </div>
  );
}

function PlanSection({ trace }: { trace: TurnTrace }) {
  const plan = trace.response_plan;
  return (
    <Section
      title="Response plan"
      engine="deterministic"
      ready={Boolean(plan)}
      hint={plan ? `≤ ${plan.max_words} words` : undefined}
    >
      {plan && (
        <>
          <div className="structure-seq">
            {plan.structure.map((step, index) => (
              <span key={step} style={{ display: 'inline-flex', gap: 5, alignItems: 'center' }}>
                {index > 0 && <span className="arrow">→</span>}
                <Pill tone="blue">{step}</Pill>
              </span>
            ))}
          </div>

          <dl className="kv">
            <dt>Template</dt>
            <dd>{plan.template_id ?? '—'}</dd>
            <dt>Max words</dt>
            <dd>
              <Mono>{plan.max_words}</Mono>
            </dd>
            <dt>Max questions</dt>
            <dd>
              <Mono>{plan.max_questions}</Mono>
            </dd>
            <dt>Language</dt>
            <dd>
              <Pill tone="grey">{plan.language}</Pill> <span className="muted small">{plan.language_level}</span>
            </dd>
          </dl>

          <div className="constraint-grid">
            <Constraint label="full_solution_allowed" value={plan.full_solution_allowed} />
            <Constraint label="wait_for_student" value={plan.wait_for_student} invert />
          </div>

          {plan.global_rules && (
            <div className="versions-grid">
              {Object.entries(plan.global_rules).map(([key, value]) => (
                <Fragment key={key}>
                  <dt>{key}</dt>
                  <dd>{String(value)}</dd>
                </Fragment>
              ))}
            </div>
          )}
        </>
      )}
    </Section>
  );
}

function GuardrailSection({ trace }: { trace: TurnTrace }) {
  const events = trace.guardrail;
  const ran = stepRan(trace, 'guardrail');
  const blocked = events?.some(
    (event) => event.type === 'leak_blocked' || event.type === 'fallback_served',
  );

  return (
    <Section
      title="Guardrail"
      engine="policy"
      ready={Boolean(events) || ran}
      pendingLabel="scan runs after generation…"
      hint={
        events ? (
          blocked ? (
            <Pill tone="red">leak blocked</Pill>
          ) : (
            `${events.length} event${events.length === 1 ? '' : 's'}`
          )
        ) : ran ? (
          <Pill tone={trace.scan_coverage?.scanned ? 'green' : 'amber'}>
            {trace.scan_coverage ? (trace.scan_coverage.scanned ? 'scanned, clean' : 'not scanned') : 'no events'}
          </Pill>
        ) : undefined
      }
    >
      <ScanCoverage trace={trace} ran={ran} hasEvents={Boolean(events?.length)} />
      {events?.map((event, index) => (
        <GuardrailRow key={`${event.type}-${index}`} event={event} />
      ))}
    </Section>
  );
}

/**
 * Whether a scan happened at all, and what it knowingly cannot see. Without this a
 * turn with no guardrail events is ambiguous — clean, or never scanned? — and that
 * ambiguity is the shape of several defects already found in this pipeline.
 */
function ScanCoverage({
  trace,
  ran,
  hasEvents,
}: {
  trace: TurnTrace;
  ran: boolean;
  hasEvents: boolean;
}) {
  const coverage = trace.scan_coverage;

  if (!coverage) {
    if (hasEvents || !ran) return null;
    return (
      <div className="notice amber">
        The guardrail step ran but logged nothing, and reported no scan coverage. Whether it was
        asked to scan cannot be told from this trace — which is not the same as a clean scan.
      </div>
    );
  }

  return (
    <div className={`notice ${coverage.scanned ? 'green' : 'red'}`}>
      <strong>
        {coverage.scanned
          ? 'A scan ran on this turn.'
          : 'No scan ran on this turn — nothing could have been caught.'}
      </strong>
      {coverage.withheld_solution !== undefined && (
        <div className="small" style={{ marginTop: 4 }}>
          Solution withheld this turn: {String(coverage.withheld_solution)}
        </div>
      )}
      {coverage.forms && coverage.forms.length > 0 && (
        <div className="small" style={{ marginTop: 4 }}>
          Covered: {coverage.forms.join(', ')}
        </div>
      )}
      {coverage.not_detected && coverage.not_detected.length > 0 && (
        <div className="small" style={{ marginTop: 4 }}>
          <strong>Known blind spots:</strong> {coverage.not_detected.join(', ')} — a leak in one of
          these forms would pass unrecorded.
        </div>
      )}
    </div>
  );
}

function GuardrailRow({ event }: { event: GuardrailEvent }) {
  return (
    <div className={`guardrail-event ${event.severity}`}>
      <span className="etype">{event.type}</span>
      <span className="guardrail-meaning">{GUARDRAIL_MEANING[event.type] ?? 'Guardrail check.'}</span>
      {/* The detail is a non-reversible handle, never the withheld answer itself. */}
      {event.detail && <span className="guardrail-detail mono">{event.detail}</span>}
      {event.action_taken && (
        <div className="small" style={{ marginTop: 4 }}>
          Action taken: {event.action_taken}
        </div>
      )}
      <div className="small muted" style={{ marginTop: 5 }}>at {event.at_ms} ms</div>
    </div>
  );
}

function TimingSection({ trace }: { trace: TurnTrace }) {
  const timings = trace.timings;
  const max = timings ? Math.max(...timings.map((t) => t.ms)) : 1;
  const overBudget = (trace.pre_stream_ms ?? 0) > PRE_STREAM_BUDGET_MS;

  return (
    <Section
      title="Timing"
      engine="deterministic"
      defaultOpen={false}
      ready={Boolean(timings)}
      hint={trace.total_ms ? formatMs(trace.total_ms) : undefined}
    >
      {timings?.map((timing, index) => (
        // A retried generation reports under the same `step`, so the index is what
        // keeps the two rows distinct.
        <div className="timing-row" key={`${timing.step}-${index}`}>
          <span className="label" title={timing.label}>
            {timing.label}
          </span>
          <Bar
            value={(timing.ms / max) * 100}
            tone={
              timing.engine === 'llm'
                ? 'amber'
                : timing.engine === 'data'
                  ? 'green'
                  : timing.engine === 'policy'
                    ? 'red'
                    : 'blue'
            }
          />
          <span className="ms">{formatMs(timing.ms)}</span>
        </div>
      ))}
      {(trace.pre_stream_ms || trace.total_ms) && (
        <div className="timing-total">
          {trace.pre_stream_ms !== undefined && (
            <span>
              pre-stream <Mono>{formatMs(trace.pre_stream_ms)}</Mono>{' '}
              <Pill tone={overBudget ? 'red' : 'green'}>
                {overBudget ? 'over' : 'within'} {PRE_STREAM_BUDGET_MS} ms budget
              </Pill>
            </span>
          )}
          {trace.total_ms !== undefined && (
            <span>
              total <Mono>{formatMs(trace.total_ms)}</Mono>
            </span>
          )}
        </div>
      )}
    </Section>
  );
}

function VersionsSection({ trace }: { trace: TurnTrace }) {
  const versions = trace.versions;
  return (
    <Section title="Models & versions" engine="deterministic" defaultOpen={false} ready={Boolean(versions)}>
      {versions && (
        <>
          {(['models', 'tables', 'prompts'] as const)
            .filter((group) => Object.keys(versions[group] ?? {}).length > 0)
            .map((group) => (
            <div key={group}>
              <h4 className="section-title" style={{ marginTop: 0 }}>
                {group}
              </h4>
              <dl className="versions-grid">
                {Object.entries(versions[group]).map(([key, value]) => (
                  <Fragment key={key}>
                    <dt>{key}</dt>
                    <dd>{value}</dd>
                  </Fragment>
                ))}
              </dl>
            </div>
          ))}
        </>
      )}
    </Section>
  );
}
