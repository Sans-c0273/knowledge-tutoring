/**
 * Plain-language narration for the top of the inspector, composed purely from
 * TurnTrace fields — no extra LLM call (addendum, Web UI scope item 3).
 */

import { CONFIDENCE_THRESHOLD, INTENT_PHRASE, strategyName } from '../../reference';
import type { TurnTrace } from '../../types';
import type { Tone } from '../ui';

export interface TurnSummary {
  understood: string;
  intend: string;
  flags: { tone: Tone; label: string; title?: string }[];
}

const READABLE_STRUCTURE: Record<string, string> = {
  direct_answer: 'give the answer',
  explanation: 'explain it',
  source_reference: 'cite the source',
  check_understanding: 'check understanding',
  ordered_steps: 'lay out the steps',
  next_step_prompt: 'ask for the next step',
  single_hint: 'one hint',
  student_attempt_prompt: 'ask them to try',
  reframed_explanation: 're-explain differently',
  one_example_or_analogy: 'one example',
  single_check_question: 'one check question',
  state_evaluation: 'state the verdict',
  explain_feedback: 'explain the feedback',
  next_action: 'point to the next action',
  single_practice_question: 'one practice question',
};

const readable = (step: string) => READABLE_STRUCTURE[step] ?? step.replace(/_/g, ' ');

export function summarise(trace: TurnTrace): TurnSummary {
  const flags: TurnSummary['flags'] = [];

  const intent = trace.intent;
  const topic = trace.topic;
  const plan = trace.response_plan;
  const strategy = trace.strategy;
  const policy = trace.teaching_policy;

  // ---- What I understood ---------------------------------------------------
  let understood: string;
  if (!intent) {
    understood = 'Reading the student message…';
  } else {
    const topicName = topic?.topic ?? 'an unresolved topic';
    const level = topic ? ` (${topic.student_level})` : '';
    const state = intent.learner_state === 'confused' ? 'The student is confused and ' : 'The student ';
    understood = `${state}${INTENT_PHRASE[intent.intent]} ${topicName}${level}.`;

    if (intent.special_handling.detected && intent.special_handling.type !== 'none') {
      understood +=
        intent.special_handling.type === 'homework'
          ? ' They said this is homework.'
          : ' They said this is an assessment.';
    }
    if (intent.fallback_applied) {
      understood += ` Intent confidence was ${intent.confidence.toFixed(2)}, below the ${CONFIDENCE_THRESHOLD} threshold, so the turn fell back to explain.`;
    }
  }

  // ---- What I intend to do -------------------------------------------------
  let intend: string;
  if (!strategy || !plan) {
    intend = 'Planning the response…';
  } else {
    const supporting = strategy.supporting_strategies.map((s) => strategyName(s.strategy_id));
    const steps = plan.structure.map(readable).join(', then ');
    intend = `${strategyName(strategy.primary_strategy.strategy_id)}${
      supporting.length > 0 ? ` supported by ${supporting.join(' and ')}` : ''
    }: ${steps} — at most ${plan.max_words} words.`;
    if (!plan.full_solution_allowed) intend += ' The full solution is withheld this turn.';
    if (plan.wait_for_student) intend += ' Then wait for their attempt.';
  }

  // ---- Flags ---------------------------------------------------------------
  if (intent && intent.confidence < CONFIDENCE_THRESHOLD) {
    flags.push({
      tone: 'amber',
      label: `low confidence ${intent.confidence.toFixed(2)}`,
      title: intent.fallback_reason ?? 'Below the classifier confidence threshold.',
    });
  }
  if (intent?.special_handling.detected && intent.special_handling.type !== 'none') {
    flags.push({ tone: 'red', label: `${intent.special_handling.type} mode` });
  }
  if (policy?.override_applied) {
    flags.push({ tone: 'red', label: 'policy override', title: policy.override_note });
  }
  if (strategy && !strategy.constraints.full_solution_allowed_now) {
    flags.push({ tone: 'red', label: 'answer withheld' });
  }
  if (strategy?.constraints.student_attempt_required) {
    flags.push({ tone: 'blue', label: 'attempt required' });
  }
  if (trace.answer_evaluation) {
    const tone: Tone =
      trace.answer_evaluation.evaluation === 'correct'
        ? 'green'
        : trace.answer_evaluation.evaluation === 'cannot_evaluate'
          ? 'grey'
          : 'amber';
    flags.push({ tone, label: `attempt ${trace.answer_evaluation.evaluation.replace(/_/g, ' ')}` });
  }
  if (trace.guardrail?.some((event) => event.type === 'leak_blocked')) {
    flags.push({ tone: 'red', label: 'leak blocked' });
  } else if (trace.guardrail?.some((event) => event.type === 'fallback_served')) {
    // The student got the safe template instead of the model's draft; that is a
    // guardrail save and must read as loudly as a block.
    flags.push({ tone: 'red', label: 'fallback served' });
  } else if (trace.guardrail?.some((event) => event.type === 'leak_regenerated')) {
    flags.push({ tone: 'amber', label: 'regenerated once' });
  }

  return { understood, intend, flags };
}
