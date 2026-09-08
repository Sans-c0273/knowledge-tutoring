/**
 * Read-only reference data from the Technical Specification, used to render
 * machine values (S04, KM02, `use_prerequisite_as_scaffold`) as readable text.
 * The backend remains the source of truth for what fired; this only names things.
 */

import type { EngineKind, Intent, PipelineStep, Relation, StrategyId } from './types';

export const STRATEGY_LIBRARY: Record<StrategyId, { name: string; behavior: string; next: string }> = {
  S01: {
    name: 'Direct Answer / Explanation',
    behavior: 'Give a supported answer + explanation for the level',
    next: 'Optional check understanding',
  },
  S02: {
    name: 'Example / Analogy',
    behavior: 'Make the concept concrete with ONE example',
    next: 'Return to the concept',
  },
  S03: {
    name: 'Step-by-Step Guidance',
    behavior: 'Break the process into ordered steps',
    next: 'Ask student to perform next step',
  },
  S04: {
    name: 'Hint / Scaffold',
    behavior: 'Give the smallest useful clue, NOT the answer',
    next: 'Wait for student attempt',
  },
  S05: {
    name: 'Re-explain Differently',
    behavior: 'Change wording, example, or approach',
    next: 'Check understanding',
  },
  S06: {
    name: 'Feedback',
    behavior: "Evaluate the student's work and explain it",
    next: 'Correct, guide, or reinforce',
  },
  S07: {
    name: 'Check Understanding',
    behavior: 'Ask ONE focused understanding question',
    next: 'Wait for student response',
  },
  S08: {
    name: 'Practice / Quiz',
    behavior: 'Ask an appropriate practice question',
    next: 'Evaluate with S06',
  },
};

export function strategyName(id: StrategyId): string {
  return STRATEGY_LIBRARY[id]?.name ?? id;
}

export const INTENT_LABEL: Record<Intent, string> = {
  explain: 'Explain — learn a new concept',
  clarify: 'Clarify — previous explanation not understood',
  solve: 'Solve — help completing a problem',
  hint: 'Hint — partial help only',
  check_answer: 'Check answer — evaluate an attempt',
  practice_quiz: 'Practice / quiz — test knowledge',
  summarize_review: 'Summarize / review — consolidate material',
};

/** Plain-language topic phrase used by the inspector summary strip. */
export const INTENT_PHRASE: Record<Intent, string> = {
  explain: 'wants an explanation of',
  clarify: 'did not understand the previous explanation of',
  solve: 'wants help solving a problem in',
  hint: 'is asking for a hint on',
  check_answer: 'wants their attempt checked in',
  practice_quiz: 'wants practice questions on',
  summarize_review: 'wants a review of',
};

export const RELATION_LABEL: Record<Relation, string> = {
  prerequisite_of: 'prerequisite of',
  related_to: 'related to',
  next_topic: 'next topic',
  part_of: 'part of',
  uses: 'uses',
};

export const KNOWLEDGE_ACTION_LABEL: Record<string, string> = {
  use_current_topic_context: 'Use the current topic only',
  use_prerequisite_as_scaffold: 'Scaffold from the prerequisite',
  use_related_concept_for_reexplanation: 'Re-explain via a related concept',
  practice_current_topic_relationship: 'Practise the current topic relationship',
  check_or_scaffold_prerequisite: 'Check the prerequisite first, scaffold only if the probe fails',
};

/** Which engine runs each pipeline step — drives the colour band in the inspector. */
export const STEP_ENGINE: Record<PipelineStep, EngineKind> = {
  intent: 'llm',
  topic_level: 'deterministic',
  retrieval: 'data',
  kl_lookup: 'data',
  answer_evaluation: 'llm',
  strategy_selection: 'deterministic',
  response_planning: 'deterministic',
  generation: 'llm',
  guardrail: 'policy',
};

export const STEP_LABEL: Record<PipelineStep, string> = {
  intent: 'Intent detection (Call A)',
  topic_level: 'Topic + student level',
  retrieval: 'RAG retrieval',
  kl_lookup: 'KL Map lookup',
  answer_evaluation: 'Answer evaluation (Call B)',
  strategy_selection: 'Strategy selection',
  response_planning: 'Response planning',
  generation: 'Generation (Call C)',
  guardrail: 'Output guardrail',
};

export const ENGINE_LABEL: Record<EngineKind, string> = {
  llm: 'LLM call',
  deterministic: 'Deterministic rules',
  data: 'Data / knowledge',
  policy: 'Policy & guardrail',
};

/**
 * §3.5 KM rules. The trace names which rule fired but not what it says, because the
 * table is static — the inspector fills the wording in from here, and defers to the
 * backend whenever it does send `condition`/`action`.
 */
export const KM_RULES: Record<string, { condition: string; action: string }> = {
  KM01: {
    condition: 'learner_state = normal',
    action: 'Current topic only; nearby concepts usable in explanation',
  },
  KM02: {
    condition: 'learner_state = confused AND immediate prerequisite exists',
    action:
      'Return the prerequisite at distance 1 and check-or-scaffold it: probe with one question, scaffold only on a failed probe',
  },
  KM03: {
    condition: 'intent = clarify AND related concept exists',
    action: 'Return the related concept for an alternative explanation',
  },
  KM04: {
    condition: 'intent = practice_quiz (not in the POC)',
    action: 'Current topic relationship',
  },
  KM05: {
    condition: 'Student demonstrates understanding and a next topic exists (not in the POC)',
    action: 'Return the next topic as an optional extension',
  },
};

/**
 * What each guardrail event means, keyed by type.
 *
 * The `detail` string is deliberately uninformative — it carries a non-reversible
 * handle (`canonical_answer matched as exact (P03, 5 chars)`) rather than the
 * withheld answer itself. So the meaning and the severity of an event have to come
 * from its type: the *fact* a leak was blocked is the signal, not its content.
 */
export const GUARDRAIL_MEANING: Record<string, string> = {
  leak_scan_clean: 'Scan clean — the draft contained none of the withheld content.',
  quiz_answer_scan_clean: 'Scan clean — the draft contained none of the quiz key.',
  leak_regenerated:
    'Leak caught in the draft and regenerated with a warning note. The student never saw it.',
  leak_blocked:
    'Leak blocked. The redraft leaked again, so generation was discarded. The student never saw it.',
  fallback_served:
    'A safe template was served in place of the model’s draft. Counts as a guardrail save, not a hard leak.',
  plan_violation:
    'The draft exceeded a response-plan limit. Served anyway and logged for prompt tuning.',
  other: 'Guardrail check.',
};

/** §2.1 — below this the pipeline falls back to `explain`. */
export const CONFIDENCE_THRESHOLD = 0.6;

/** §10 — p95 pre-stream latency budget. */
export const PRE_STREAM_BUDGET_MS = 800;
