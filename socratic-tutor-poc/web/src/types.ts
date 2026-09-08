/**
 * Wire types for the glass-box UI.
 *
 * These mirror the Technical Specification: §2 (data schemas), §3 (rule tables),
 * §4.3 (ResponsePlan), §6 (guardrail), §7 (TurnTrace + APIs). Field names follow the
 * spec's JSON exactly; fields the UI needs and the spec leaves implicit (node ids for
 * graph rendering, per-step timings) are additive and marked below.
 */

// ---------------------------------------------------------------------------
// §2.1 IntentResult
// ---------------------------------------------------------------------------

export type Intent =
  | 'explain'
  | 'clarify'
  | 'solve'
  | 'hint'
  | 'check_answer'
  | 'practice_quiz'
  | 'summarize_review';

export type LearnerState = 'normal' | 'confused';
export type SpecialHandlingType = 'none' | 'homework' | 'assessment';
export type StudentLevel = 'beginner' | 'intermediate' | 'advanced';
export type Language = 'th' | 'en';

export interface IntentResult {
  intent: Intent;
  learner_state: LearnerState;
  special_handling: { detected: boolean; type: SpecialHandlingType };
  confidence: number;
  /** True when confidence fell below threshold and the pipeline fell back to `explain` (§2.1). */
  fallback_applied?: boolean;
  fallback_reason?: string;
}

// ---------------------------------------------------------------------------
// Step 2 — topic + student level
// ---------------------------------------------------------------------------

export interface TopicResolution {
  topic: string;
  topic_th?: string;
  /** KL Map node id, so the inspector can highlight the node this turn started from. */
  node_id?: string;
  student_level: StudentLevel;
  /** How the topic was matched: syllabus lookup, embedding similarity, carried from previous turn. */
  matched_by: string;
  level_source: 'onboarding_questionnaire' | 'default_beginner' | 'carried_from_session';
  /** Match score, and what else the message nearly matched — a bad turn often starts here. */
  confidence?: number;
  ambiguous?: boolean;
  resolved_by_context?: boolean;
  candidates?: string[];
}

// ---------------------------------------------------------------------------
// Step 2a — RAG retrieval
// ---------------------------------------------------------------------------

export interface RetrievedChunk {
  chunk_id: string;
  score: number;
  text: string;
  source_ref: string;
  language: Language;
}

// ---------------------------------------------------------------------------
// §2.3 KL Map + KnowledgeContext
// ---------------------------------------------------------------------------

export type Relation = 'prerequisite_of' | 'related_to' | 'next_topic' | 'part_of' | 'uses';

export interface KlNode {
  id: string;
  name: string;
  name_th?: string;
}

export interface KlEdge {
  from: string;
  to: string;
  relation: Relation;
}

/**
 * Which file answered `GET /api/klmap`. `approved` alone cannot separate a map a
 * reviewer published from the hand-authored seed pack; `source` can. Mirrors
 * `klmap_routes.SOURCE_*`.
 */
export type KlMapSource = 'live' | 'draft' | 'seed';

export interface KlMap {
  course_id: string;
  course_name: string;
  /** Reviewed and published through the review API. The seed pack never earns this. */
  approved: boolean;
  source: KlMapSource;
  nodes: KlNode[];
  edges: KlEdge[];
}

export interface KnowledgeNodeRef {
  topic: string;
  topic_th?: string;
  node_id?: string;
  distance: number;
}

export interface KnowledgeContext {
  current_topic: string;
  current_node_id?: string;
  prerequisites: KnowledgeNodeRef[];
  related_topics: KnowledgeNodeRef[];
  next_topics: KnowledgeNodeRef[];
  relationships: string[];
  /** BFS depth actually used — widens to 2 for prerequisites when learner_state = confused. */
  max_depth: number;
  /** Node ids traversed this turn; the inspector highlights exactly these. */
  nodes_hit: string[];
}

// ---------------------------------------------------------------------------
// R18 — KL Map review: extraction drafts and their validation reports
// ---------------------------------------------------------------------------

/**
 * One problem found in a draft map, as `domain/klmap/loader.py` emits it.
 * `location` points at the offending item in the YAML (`edges[7]`, `nodes[2]`),
 * or at a whole section (`edges`) when the problem spans several entries.
 */
export interface ValidationIssue {
  code: string;
  message: string;
  location?: string | null;
  /**
   * Cycle path for KL009, newest-first node ids. The loader reports the path inside
   * `message`; when the backend does not send this field the UI parses it from there
   * (see `cyclePath` in `src/components/review/issues.ts`).
   */
  path?: string[];
}

export interface ValidationReport {
  source: string;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
}

/** `pending` is what the backend sends for an unreviewed draft; `draft` is accepted too. */
export type DraftStatus = 'pending' | 'draft' | 'approved' | 'rejected';

/**
 * An LLM-extracted map awaiting human review. Drafts that fail validation are
 * returned too — a reviewer has to see what the model actually proposed in order
 * to fix it, so nothing is silently repaired.
 */
export interface KlMapDraft {
  course_id: string;
  course_name: string;
  /** Absolute path of the YAML the reviewer edits; approval never edits in place. */
  path: string;
  extracted_at: string;
  status: DraftStatus;
  reviewed_at?: string;
  review_note?: string;
  /** Source files the extraction drew on, for provenance. */
  sources: string[];
  nodes: KlNode[];
  edges: KlEdge[];
  report: ValidationReport;
}

export interface DraftDecision {
  approve: boolean;
  note?: string;
}

/** §3.5 — KM rules that fired (POC ships KM01–KM03). */
export interface KmRuleHit {
  rule: string;
  /** Absent on the wire: §3.5 defines these statically, so the UI fills them from `reference.ts`. */
  condition?: string;
  action?: string;
  applied: boolean;
  /** Set when a rule matched but lost a tie-break (O2: KM02 beats KM03; both logged). */
  note?: string;
}

// ---------------------------------------------------------------------------
// §3 Strategy selection
// ---------------------------------------------------------------------------

export type StrategyId = 'S01' | 'S02' | 'S03' | 'S04' | 'S05' | 'S06' | 'S07' | 'S08';

export interface StrategyRef {
  strategy_id: StrategyId;
  strategy_name: string;
}

export type Evaluation = 'correct' | 'partially_correct' | 'incorrect' | 'cannot_evaluate';

export interface AnswerEvaluation {
  evaluation: Evaluation;
  /** Where the attempt went wrong. Never names the withheld answer — it arrives redacted. */
  error_locus?: string;
  /** True when a deterministic checker (not the LLM) produced the verdict (§5.2). */
  deterministic_checker_used: boolean;
  /** Which verdict source decided it, e.g. `deterministic`, `llm_confirmed`, `llm_unverified`. */
  source?: string;
  checker_note?: string;
}

/**
 * §3.1 Trigger Matrix rows evaluated this turn — the whole contest, not just the winner.
 *
 * `lost_on` is the field that matters most: knowing TM12 beat TM16 is half an answer,
 * while knowing whether a chosen priority decided it or **alphabetical row id** did is
 * the difference between a rule working as designed and a rule nobody actually decided.
 */
export interface TriggerMatch {
  row_id: string;
  matched: boolean;
  selected: boolean;
  /** The row's match conditions as a mapping, e.g. `{intent: 'check_answer'}`. */
  key?: Record<string, string | null>;
  priority?: number;
  specificity?: number;
  /** What this row would have selected, including for rows that lost. */
  primary?: StrategyRef;
  supporting?: StrategyRef[];
  /** Which row beat this one, and on what — `priority`, `specificity`, or a tie-break. */
  lost_to?: string;
  lost_on?: string;
  /** Free-text outcome from the selector, e.g. `(selected)` or `(fallback row)`. */
  note?: string;
  condition?: string;
  special_handling?: string;
  /** Why a near-miss row did not match — the reason a reviewer would otherwise have to infer. */
  excluded_reason?: string;
}

/** §3.3 Combination Rules verdict. */
export interface CombinationVerdict {
  allowed: boolean;
  /** The primary the row belongs to (`S06`), or a sentence describing the pairing. */
  rule: string;
  /** Human-readable order the strategies run in, e.g. ["Evaluate","Feedback","Next action"]. */
  sequence?: string[];
  /** Strategies the rules removed. Refs from the mock, ids from the backend. */
  dropped: (StrategyRef | string)[];
  actions?: string[];
  note?: string;
}

/**
 * §3.4 Teaching Policy — hard override, top of precedence.
 *
 * The backend sends the policy table's own row id (`TP02`) plus already-human text
 * for each column, so these are plain strings rather than closed unions.
 */
export interface TeachingPolicyApplied {
  /** Row id (`TP02`) or the semantic row name. */
  row: string;
  /** Which situation the row covers, e.g. `homework`. */
  context?: string;
  row_label?: string;
  /**
   * Permission columns, verbatim from `teaching_policy.yaml` ("Not initially - hint
   * first", "Required first"). Deliberately strings, not a union: §3.4 is written in
   * prose, and re-encoding it here would put a third vocabulary between the policy
   * table and the screen. Display these; never switch on them.
   */
  direct_answer: string;
  attempt: string;
  check: string;
  /** True when the policy changed the outcome the Trigger Matrix would have produced. */
  override_applied: boolean;
  override_note?: string;
  override_notes?: string[];
  /** Strategies this policy forbids on this turn — a constraint even without an override. */
  prohibited_strategies?: string[];
  /** Other policy rows that also matched. */
  also_applied?: string[];
  /** Priority number from the table, or the precedence chain as prose. */
  precedence?: number | string;
}

export interface StrategySelection {
  primary_strategy: StrategyRef;
  evaluation?: Evaluation;
  supporting_strategies: StrategyRef[];
  constraints: {
    direct_answer_allowed: boolean;
    full_solution_allowed_now: boolean;
    student_attempt_required: boolean;
  };
  knowledge_guidance?: {
    action: string;
    target_concept: string;
    /** Closed `Relation` vocabulary; `direction` says which way it was read. */
    relationship: string;
    direction?: 'forward' | 'inverse';
    distance: number;
  };
  context: { topic: string; student_level: StudentLevel };
}

// ---------------------------------------------------------------------------
// §4.3 ResponsePlan
// ---------------------------------------------------------------------------

export interface ResponsePlan {
  structure: string[];
  max_words: number;
  max_questions: number;
  language_level: StudentLevel;
  language: Language;
  full_solution_allowed: boolean;
  wait_for_student: boolean;
  /** §4.1 — which template row produced the structure. */
  template_id?: string;
  /** §4.2 global rules, rendered for inspection. */
  global_rules?: Record<string, string | number | boolean>;
}

// ---------------------------------------------------------------------------
// §6 Guardrail
// ---------------------------------------------------------------------------

export type GuardrailEventType =
  | 'leak_scan_clean'
  | 'leak_regenerated'
  | 'leak_blocked'
  | 'fallback_served'
  | 'plan_violation'
  | 'quiz_answer_scan_clean'
  | 'other';

export interface GuardrailEvent {
  type: GuardrailEventType;
  severity: 'info' | 'warning' | 'critical';
  detail: string;
  /** What the guardrail did about it: regenerated, served a fallback, logged only. */
  action_taken?: string;
  /**
   * No field here ever carries the withheld answer. The guardrail describes what it
   * matched with a non-reversible handle (`canonical_answer matched as exact (P03,
   * 5 chars)`); the browser is not a place the secret is allowed to reach.
   */
  at_ms: number;
}

// ---------------------------------------------------------------------------
// §7 TurnTrace
// ---------------------------------------------------------------------------

export type PipelineStep =
  | 'intent'
  | 'topic_level'
  | 'retrieval'
  | 'kl_lookup'
  | 'answer_evaluation'
  | 'strategy_selection'
  | 'response_planning'
  | 'generation'
  | 'guardrail';

/** Colour bands from the design document: amber = LLM, blue = deterministic, green = data, red = policy/guardrail. */
export type EngineKind = 'llm' | 'deterministic' | 'data' | 'policy';

export interface StepTiming {
  step: PipelineStep;
  label: string;
  engine: EngineKind;
  ms: number;
}

export interface TraceVersions {
  models: Record<string, string>;
  tables: Record<string, string>;
  prompts: Record<string, string>;
}

export interface TurnTrace {
  turn_id: string;
  session_id: string;
  created_at: string;
  student_message: string;
  language: Language;
  intent?: IntentResult;
  topic?: TopicResolution;
  retrieval?: RetrievedChunk[];
  knowledge_context?: KnowledgeContext;
  km_rules?: KmRuleHit[];
  answer_evaluation?: AnswerEvaluation;
  trigger_matches?: TriggerMatch[];
  combination?: CombinationVerdict;
  teaching_policy?: TeachingPolicyApplied;
  strategy?: StrategySelection;
  response_plan?: ResponsePlan;
  guardrail?: GuardrailEvent[];
  timings?: StepTiming[];
  versions?: TraceVersions;
  /**
   * What the guardrail actually scanned for. The decisive field for reading a turn
   * with no guardrail events: `scanned: false` means no scan happened, which is not
   * the same as a clean scan, and `not_detected` names the gaps it knows about.
   */
  scan_coverage?: {
    scanned: boolean;
    withheld_solution?: boolean;
    forms?: string[];
    not_detected?: string[];
  };
  /** The served text, after the guardrail. Present once generation has finished. */
  generated_text?: string;
  /** Set when the turn aborted; the trace still renders everything it got to. */
  error?: string;
  /** Pre-stream latency budget check (§10: p95 ≤ 800 ms). */
  pre_stream_ms?: number;
  total_ms?: number;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatMessage {
  id: string;
  role: 'student' | 'tutor';
  text: string;
  language: Language;
  /** Tutor messages carry the turn they came from, so selecting one opens its trace. */
  turn_id?: string;
  streaming?: boolean;
  error?: string;
}

export interface ChatTurnRequest {
  session_id: string;
  student_id: string;
  course_id: string;
  message: string;
}

/**
 * The generation phase. The backend buffers a withholdable draft, runs the guardrail,
 * and only then streams — so there is a silent gap between the last decision patch and
 * the first token, and `generating` is what fills it.
 */
export interface TurnStatus {
  phase: 'generating' | string;
  message?: string;
}

export interface ChatTurnHandlers {
  onToken: (token: string) => void;
  onStatus?: (status: TurnStatus) => void;
  onTrace: (partial: Partial<TurnTrace> & { turn_id: string }) => void;
  onDone: (trace?: Partial<TurnTrace>) => void;
  onError: (error: Error) => void;
}

// ---------------------------------------------------------------------------
// Ingestion (R17)
// ---------------------------------------------------------------------------

export type IngestStage = 'parse' | 'chunk' | 'embed' | 'kl-extract' | 'review-ready';

export const INGEST_STAGES: IngestStage[] = ['parse', 'chunk', 'embed', 'kl-extract', 'review-ready'];

export type IngestStatus = 'queued' | 'running' | 'done' | 'error';

export interface IngestEvent {
  file_id: string;
  filename: string;
  stage: IngestStage;
  percent: number;
  status: IngestStatus;
  message?: string;
}

export interface IngestFile {
  file_id: string;
  filename: string;
  source: 'upload' | 'url';
  stage: IngestStage;
  percent: number;
  status: IngestStatus;
  message?: string;
  /** Highest percent reached per stage, for the stage rail. */
  stage_progress: Partial<Record<IngestStage, number>>;
  updated_at: string;
}
