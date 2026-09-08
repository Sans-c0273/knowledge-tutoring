import { describe, expect, it } from 'vitest';
import type { ResponsePlan, StrategySelection, TeachingPolicyApplied, TurnTrace } from '../../types';
import { summarise } from './summary';

/**
 * The summary strip is the system's own account of what it just did, in the words
 * Tanat will read. If it drifts from the trace it is worse than nothing, so these
 * tests are about faithfulness: never claim a constraint the trace does not carry,
 * and never omit one it does.
 */

function trace(overrides: Partial<TurnTrace> = {}): TurnTrace {
  return {
    turn_id: 't1',
    session_id: 's1',
    created_at: '2026-09-01T10:00:00Z',
    student_message: 'test',
    language: 'en',
    ...overrides,
  };
}

const strategy = (overrides: Partial<StrategySelection> = {}): StrategySelection => ({
  primary_strategy: { strategy_id: 'S04', strategy_name: 'Hint / Scaffold' },
  supporting_strategies: [{ strategy_id: 'S07', strategy_name: 'Check Understanding' }],
  constraints: {
    direct_answer_allowed: false,
    full_solution_allowed_now: false,
    student_attempt_required: true,
  },
  context: { topic: 'Linear Equations', student_level: 'beginner' },
  ...overrides,
});

const plan = (overrides: Partial<ResponsePlan> = {}): ResponsePlan => ({
  structure: ['single_hint', 'student_attempt_prompt'],
  max_words: 50,
  max_questions: 1,
  language_level: 'beginner',
  language: 'en',
  full_solution_allowed: false,
  wait_for_student: true,
  ...overrides,
});

const homeworkPolicy: TeachingPolicyApplied = {
  row: 'homework',
  row_label: 'Homework',
  direct_answer: 'not_initially_hint_first',
  attempt: 'required_first',
  check: 'recommended',
  override_applied: true,
  override_note: 'Trigger Matrix chose S01; the homework row re-routed to S04.',
  precedence: 'Teaching Policy → Special Handling → Learner State → Intent → Student Level',
};

describe('summarise — what I understood', () => {
  it('names the intent, topic and level', () => {
    const { understood } = summarise(
      trace({
        intent: {
          intent: 'check_answer',
          learner_state: 'normal',
          special_handling: { detected: false, type: 'none' },
          confidence: 0.93,
        },
        topic: {
          topic: 'Two-Step Linear Equations',
          student_level: 'beginner',
          matched_by: 'syllabus',
          level_source: 'onboarding_questionnaire',
        },
      }),
    );
    expect(understood).toContain('wants their attempt checked in Two-Step Linear Equations');
    expect(understood).toContain('(beginner)');
  });

  it('says the learner is confused when the trace says so', () => {
    const { understood } = summarise(
      trace({
        intent: {
          intent: 'explain',
          learner_state: 'confused',
          special_handling: { detected: false, type: 'none' },
          confidence: 0.88,
        },
      }),
    );
    expect(understood).toMatch(/^The student is confused and /);
  });

  it('reports homework detection', () => {
    const { understood, flags } = summarise(
      trace({
        intent: {
          intent: 'check_answer',
          learner_state: 'normal',
          special_handling: { detected: true, type: 'homework' },
          confidence: 0.9,
        },
      }),
    );
    expect(understood).toContain('They said this is homework.');
    expect(flags.map((f) => f.label)).toContain('homework mode');
  });

  it('states the low-confidence fallback with the number and the threshold', () => {
    const { understood, flags } = summarise(
      trace({
        intent: {
          intent: 'explain',
          learner_state: 'normal',
          special_handling: { detected: false, type: 'none' },
          confidence: 0.58,
          fallback_applied: true,
          fallback_reason: 'split between solve and check_answer',
        },
      }),
    );
    expect(understood).toContain('0.58');
    expect(understood).toContain('below the 0.6 threshold');
    expect(understood).toContain('fell back to explain');
    expect(flags.map((f) => f.label)).toContain('low confidence 0.58');
  });

  it('does not invent an understanding before the intent call returns', () => {
    const { understood } = summarise(trace());
    expect(understood).toBe('Reading the student message…');
  });
});

describe('summarise — what I intend to do', () => {
  it('reads back the strategy, the structure and the word cap', () => {
    const { intend } = summarise(trace({ strategy: strategy(), response_plan: plan() }));
    expect(intend).toContain('Hint / Scaffold supported by Check Understanding');
    expect(intend).toContain('one hint, then ask them to try');
    expect(intend).toContain('at most 50 words');
  });

  it('says the solution is withheld exactly when the plan withholds it', () => {
    const withheld = summarise(trace({ strategy: strategy(), response_plan: plan() }));
    expect(withheld.intend).toContain('The full solution is withheld this turn.');

    const permitted = summarise(
      trace({
        strategy: strategy({
          constraints: {
            direct_answer_allowed: true,
            full_solution_allowed_now: true,
            student_attempt_required: false,
          },
        }),
        response_plan: plan({ full_solution_allowed: true, wait_for_student: false }),
      }),
    );
    expect(permitted.intend).not.toContain('withheld');
    expect(permitted.intend).not.toContain('wait for their attempt');
  });

  it('does not describe a plan that does not exist yet', () => {
    const { intend } = summarise(
      trace({
        intent: {
          intent: 'explain',
          learner_state: 'normal',
          special_handling: { detected: false, type: 'none' },
          confidence: 0.9,
        },
      }),
    );
    expect(intend).toBe('Planning the response…');
  });
});

describe('summarise — flags', () => {
  it('flags a teaching-policy override and carries its note', () => {
    const { flags } = summarise(
      trace({ strategy: strategy(), response_plan: plan(), teaching_policy: homeworkPolicy }),
    );
    const override = flags.find((flag) => flag.label === 'policy override');
    expect(override).toBeDefined();
    expect(override?.title).toBe(homeworkPolicy.override_note);
  });

  it('does not flag an override when the policy did not override anything', () => {
    const { flags } = summarise(
      trace({
        strategy: strategy(),
        response_plan: plan(),
        teaching_policy: { ...homeworkPolicy, override_applied: false, override_note: undefined },
      }),
    );
    expect(flags.map((f) => f.label)).not.toContain('policy override');
  });

  it('flags the withheld answer and the required attempt from the constraints', () => {
    const { flags } = summarise(trace({ strategy: strategy(), response_plan: plan() }));
    const labels = flags.map((f) => f.label);
    expect(labels).toContain('answer withheld');
    expect(labels).toContain('attempt required');
  });

  it('reports the evaluation verdict', () => {
    const { flags } = summarise(
      trace({
        answer_evaluation: { evaluation: 'incorrect', deterministic_checker_used: true },
      }),
    );
    expect(flags.find((f) => f.label === 'attempt incorrect')?.tone).toBe('amber');

    const correct = summarise(
      trace({ answer_evaluation: { evaluation: 'correct', deterministic_checker_used: true } }),
    );
    expect(correct.flags.find((f) => f.label === 'attempt correct')?.tone).toBe('green');
  });

  it('flags a blocked leak as critical and prefers it over the regeneration flag', () => {
    const { flags } = summarise(
      trace({
        guardrail: [
          { type: 'leak_regenerated', severity: 'warning', detail: 'first draft leaked', at_ms: 1 },
          { type: 'leak_blocked', severity: 'critical', detail: 'second draft leaked', at_ms: 2 },
        ],
      }),
    );
    const labels = flags.map((f) => f.label);
    expect(labels).toContain('leak blocked');
    expect(labels).not.toContain('regenerated once');
  });

  it('flags a served fallback as loudly as a block', () => {
    const { flags } = summarise(
      trace({
        guardrail: [
          { type: 'leak_regenerated', severity: 'warning', detail: 'first draft leaked', at_ms: 1 },
          {
            type: 'fallback_served',
            severity: 'warning',
            detail: 'safe template served instead of the draft',
            at_ms: 2,
          },
        ],
      }),
    );
    expect(flags.find((f) => f.label === 'fallback served')?.tone).toBe('red');
    expect(flags.map((f) => f.label)).not.toContain('regenerated once');
  });

  it('flags a single regeneration on its own', () => {
    const { flags } = summarise(
      trace({
        guardrail: [
          { type: 'leak_regenerated', severity: 'warning', detail: 'first draft leaked', at_ms: 1 },
        ],
      }),
    );
    expect(flags.map((f) => f.label)).toContain('regenerated once');
  });

  it('raises no alarm on a clean guardrail scan', () => {
    const { flags } = summarise(
      trace({
        guardrail: [{ type: 'leak_scan_clean', severity: 'info', detail: 'clean', at_ms: 1 }],
      }),
    );
    const labels = flags.map((f) => f.label);
    expect(labels).not.toContain('leak blocked');
    expect(labels).not.toContain('regenerated once');
  });
});
