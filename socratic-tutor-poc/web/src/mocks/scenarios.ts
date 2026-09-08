/**
 * TurnTrace fixtures for the mock backend.
 *
 * Scenario `homework-x7` reproduces the Design Document §10 worked example
 * ("I got x = 7 for this homework problem, is it right?") field by field. The other
 * three cover the paths the inspector must render distinctly: a confused Thai turn
 * that widens the KL Map BFS, a hint turn where the Combination Rules withhold the
 * solution, and a pressure turn where the Teaching Policy overrides the Trigger Matrix
 * and the guardrail ends up blocking a leak.
 */

import type { Language, TraceVersions, TurnTrace } from '../types';

const VERSIONS: TraceVersions = {
  models: {
    'Call A · intent': 'claude-haiku-4-5',
    'Call B · evaluation': 'claude-haiku-4-5',
    'Call C · generation': 'claude-sonnet-5',
    embeddings: 'bge-m3 (local)',
  },
  tables: {
    trigger_matrix: 'v3',
    strategy_library: 'v1',
    combination_rules: 'v1',
    teaching_policy: 'v2',
    km_rules: 'v1',
    response_templates: 'v2',
  },
  prompts: {
    call_a: 'intent@v4',
    call_b: 'evaluation@v3',
    call_c: 'generation@v6',
  },
};

/** One pre-stream stage: a delay, then a patch merged into the live trace. */
export interface TraceStage {
  delay_ms: number;
  patch: Partial<TurnTrace>;
}

export interface MockScenario {
  id: string;
  label: string;
  language: Language;
  /** Example message shown in the chat view's quick-prompt bar. */
  prompt: string;
  matches: (message: string) => boolean;
  stages: TraceStage[];
  reply: string;
  /** Merged after the reply has finished streaming (guardrail, timings, totals). */
  final: Partial<TurnTrace>;
}

const has = (message: string, ...needles: string[]) => {
  const lower = message.toLowerCase();
  return needles.some((n) => lower.includes(n.toLowerCase()));
};

// ---------------------------------------------------------------------------
// 1 · Homework check_answer — Design Document §10 worked example
// ---------------------------------------------------------------------------

const homework: MockScenario = {
  id: 'homework-x7',
  label: 'Homework check — answer withheld',
  language: 'en',
  prompt: 'I got x = 7 for this homework problem, is it right?',
  matches: (m) => has(m, 'x = 7', 'x=7', 'homework'),
  stages: [
    {
      delay_ms: 190,
      patch: {
        intent: {
          intent: 'check_answer',
          learner_state: 'normal',
          special_handling: { detected: true, type: 'homework' },
          confidence: 0.93,
        },
      },
    },
    {
      delay_ms: 60,
      patch: {
        topic: {
          topic: 'Two-Step Linear Equations',
          topic_th: 'สมการเชิงเส้นสองขั้นตอน',
          node_id: 'C009',
          student_level: 'beginner',
          matched_by: 'Syllabus match on the previous turn\'s problem (2x + 4 = 10)',
          level_source: 'onboarding_questionnaire',
        },
      },
    },
    {
      delay_ms: 120,
      patch: {
        retrieval: [
          {
            chunk_id: 'ch2#p2',
            score: 0.81,
            text: 'A two-step equation needs two inverse operations, applied in reverse order of operations: undo addition or subtraction first, then undo multiplication or division.',
            source_ref: 'ch2-solving-linear-equations.md § Two-step equations',
            language: 'en',
          },
          {
            chunk_id: 'ch1#p1',
            score: 0.74,
            text: 'Every operation has an inverse. Addition is undone by subtraction, multiplication by division. Whatever you apply to one side of an equation you must apply to the other.',
            source_ref: 'ch1-inverse-operations.md § What an inverse is',
            language: 'en',
          },
          {
            chunk_id: 'ch2#p4',
            score: 0.62,
            text: 'Substitute your solution back into the original equation. If both sides come out equal, the solution is correct.',
            source_ref: 'ch2-solving-linear-equations.md § Checking a solution',
            language: 'en',
          },
        ],
        knowledge_context: {
          current_topic: 'Two-Step Linear Equations',
          current_node_id: 'C009',
          prerequisites: [
            { topic: 'One-Step Linear Equations', topic_th: 'สมการเชิงเส้นขั้นตอนเดียว', node_id: 'C008', distance: 1 },
          ],
          related_topics: [
            { topic: 'Checking a Solution', topic_th: 'การตรวจคำตอบ', node_id: 'C013', distance: 1 },
          ],
          next_topics: [
            { topic: 'Word Problems to Equations', topic_th: 'โจทย์ปัญหาสู่สมการ', node_id: 'C014', distance: 1 },
          ],
          relationships: ['checking a solution verifies a two-step equation'],
          max_depth: 1,
          nodes_hit: ['C009', 'C008', 'C013', 'C014'],
        },
        km_rules: [
          {
            rule: 'KM01',
            condition: 'learner_state = normal',
            action: 'Current topic only; nearby concepts usable in explanation',
            applied: true,
          },
          {
            rule: 'KM02',
            condition: 'learner_state = confused AND immediate prerequisite exists',
            action: 'Return prerequisite at distance 1',
            applied: false,
            note: 'learner_state is normal — BFS stays at depth 1',
          },
        ],
      },
    },
    {
      delay_ms: 430,
      patch: {
        answer_evaluation: {
          evaluation: 'incorrect',
          error_locus: 'Step 1 — the constant was added instead of subtracted (sign error moving 4 across)',
          deterministic_checker_used: true,
          checker_note: 'Normalised numeric compare against the canonical solution ran before the LLM comparison; both agree.',
        },
      },
    },
    {
      delay_ms: 20,
      patch: {
        trigger_matches: [
          {
            row_id: 'TM06',
            condition: 'check_answer',
            special_handling: 'homework',
            primary: { strategy_id: 'S06', strategy_name: 'Feedback' },
            supporting: [{ strategy_id: 'S04', strategy_name: 'Hint / Scaffold' }],
            priority: 60,
            matched: true,
            selected: true,
          },
          {
            row_id: 'TM05',
            condition: 'check_answer + normal',
            special_handling: 'none',
            primary: { strategy_id: 'S06', strategy_name: 'Feedback' },
            supporting: [{ strategy_id: 'S07', strategy_name: 'Check Understanding' }],
            priority: 70,
            matched: false,
            selected: false,
            excluded_reason: 'Row requires special_handling = none; this turn detected homework.',
          },
        ],
        combination: {
          allowed: true,
          rule: 'S06 → {S04, S05, S07} — Evaluate → Feedback → Next action',
          dropped: [],
        },
        teaching_policy: {
          row: 'homework',
          row_label: 'Homework',
          direct_answer: 'not_initially_hint_first',
          attempt: 'required_first',
          check: 'recommended',
          override_applied: true,
          override_note:
            'Feedback on an incorrect attempt would normally be free to state the correct value. The Homework row withholds it: hint first, student attempt required before any full solution.',
          precedence: 'Teaching Policy → Special Handling → Learner State → Intent → Student Level',
        },
        strategy: {
          primary_strategy: { strategy_id: 'S06', strategy_name: 'Feedback' },
          evaluation: 'incorrect',
          supporting_strategies: [{ strategy_id: 'S04', strategy_name: 'Hint / Scaffold' }],
          constraints: {
            direct_answer_allowed: false,
            full_solution_allowed_now: false,
            student_attempt_required: true,
          },
          knowledge_guidance: {
            action: 'use_current_topic_context',
            target_concept: 'Two-Step Linear Equations',
            relationship: 'current_topic',
            distance: 0,
          },
          context: { topic: 'Two-Step Linear Equations', student_level: 'beginner' },
        },
      },
    },
    {
      delay_ms: 15,
      patch: {
        response_plan: {
          structure: ['state_evaluation', 'single_hint', 'student_attempt_prompt'],
          max_words: 100,
          max_questions: 1,
          language_level: 'beginner',
          language: 'en',
          full_solution_allowed: false,
          wait_for_student: true,
          template_id: 'S06 template + S04 supporting block',
          global_rules: {
            max_examples: 1,
            max_analogies: 1,
            max_questions: 1,
            source_reference_required: true,
            unsupported_claims_allowed: false,
            hint_first_and_full_solution_same_turn: false,
            tone: 'clear_respectful_encouraging_non_patronizing',
          },
        },
      },
    },
  ],
  reply:
    "Not quite — you're close. Check what happens to the sign when you move the 4 across. Try that step again and tell me what you get.",
  final: {
    guardrail: [
      {
        type: 'leak_scan_clean',
        severity: 'info',
        detail: 'canonical_answer scan (P03, exact + normalised): no match',
        action_taken: 'served as written',
        at_ms: 1898,
      },
    ],
    timings: [
      { step: 'intent', label: 'Intent detection (Call A)', engine: 'llm', ms: 186 },
      { step: 'topic_level', label: 'Topic + student level', engine: 'deterministic', ms: 11 },
      { step: 'retrieval', label: 'RAG retrieval', engine: 'data', ms: 96 },
      { step: 'kl_lookup', label: 'KL Map lookup', engine: 'data', ms: 8 },
      { step: 'answer_evaluation', label: 'Answer evaluation (Call B)', engine: 'llm', ms: 412 },
      { step: 'strategy_selection', label: 'Strategy selection', engine: 'deterministic', ms: 3 },
      { step: 'response_planning', label: 'Response planning', engine: 'deterministic', ms: 2 },
      { step: 'generation', label: 'Generation (Call C)', engine: 'llm', ms: 1180 },
      { step: 'guardrail', label: 'Output guardrail', engine: 'policy', ms: 6 },
    ],
    pre_stream_ms: 718,
    total_ms: 1904,
    versions: VERSIONS,
  },
};

// ---------------------------------------------------------------------------
// 2 · Confused Thai learner — BFS widens to depth 2, KM02 fires
// ---------------------------------------------------------------------------

const confusedThai: MockScenario = {
  id: 'confused-th',
  label: 'Confused learner (Thai) — re-explain differently',
  language: 'th',
  prompt: 'หนูไม่เข้าใจว่าทำไมต้องลบ 4 ทั้งสองข้าง ช่วยอธิบายหน่อยค่ะ',
  matches: (m) => has(m, 'ไม่เข้าใจ', 'อธิบาย', 'confused', "don't understand"),
  stages: [
    {
      delay_ms: 175,
      patch: {
        intent: {
          intent: 'explain',
          learner_state: 'confused',
          special_handling: { detected: false, type: 'none' },
          confidence: 0.88,
        },
      },
    },
    {
      delay_ms: 55,
      patch: {
        topic: {
          topic: 'Two-Step Linear Equations',
          topic_th: 'สมการเชิงเส้นสองขั้นตอน',
          node_id: 'C009',
          student_level: 'beginner',
          matched_by: 'Embedding match on "ลบ 4 ทั้งสองข้าง" against the syllabus topic list',
          level_source: 'onboarding_questionnaire',
        },
      },
    },
    {
      delay_ms: 130,
      patch: {
        retrieval: [
          {
            chunk_id: 'ch1#th1',
            score: 0.86,
            text: 'ทุกการดำเนินการมีการดำเนินการผกผันเสมอ การบวกแก้ด้วยการลบ การคูณแก้ด้วยการหาร สิ่งที่ทำกับข้างหนึ่งของสมการต้องทำกับอีกข้างเท่ากัน',
            source_ref: 'ch1-inverse-operations.md § การดำเนินการผกผัน',
            language: 'th',
          },
          {
            chunk_id: 'ch2#th2',
            score: 0.79,
            text: 'สมการสองขั้นตอนต้องใช้การดำเนินการผกผันสองครั้ง โดยแก้การบวกลบก่อน แล้วจึงแก้การคูณหาร',
            source_ref: 'ch2-solving-linear-equations.md § สมการสองขั้นตอน',
            language: 'th',
          },
        ],
        knowledge_context: {
          current_topic: 'Two-Step Linear Equations',
          current_node_id: 'C009',
          prerequisites: [
            { topic: 'One-Step Linear Equations', topic_th: 'สมการเชิงเส้นขั้นตอนเดียว', node_id: 'C008', distance: 1 },
            { topic: 'Inverse Operations', topic_th: 'การดำเนินการผกผัน', node_id: 'C006', distance: 2 },
            { topic: 'Equation Balance', topic_th: 'สมดุลของสมการ', node_id: 'C007', distance: 2 },
          ],
          related_topics: [
            { topic: 'Checking a Solution', topic_th: 'การตรวจคำตอบ', node_id: 'C013', distance: 1 },
          ],
          next_topics: [
            { topic: 'Word Problems to Equations', topic_th: 'โจทย์ปัญหาสู่สมการ', node_id: 'C014', distance: 1 },
          ],
          relationships: ['equation balance is a prerequisite of one-step linear equations'],
          max_depth: 2,
          nodes_hit: ['C009', 'C008', 'C006', 'C007', 'C013', 'C014'],
        },
        km_rules: [
          {
            rule: 'KM02',
            condition: 'learner_state = confused AND immediate prerequisite exists',
            action: 'Return prerequisite at distance 1 (One-Step Linear Equations); BFS widened to depth 2',
            applied: true,
            note: 'Check-or-scaffold only: probe the prerequisite with one question, scaffold only on a failed probe. A prerequisite edge never implies the student lacks it.',
          },
          {
            rule: 'KM03',
            condition: 'intent = clarify AND related concept exists',
            action: 'Return related concept for an alternative explanation',
            applied: false,
            note: 'Intent is explain, not clarify.',
          },
        ],
      },
    },
    {
      delay_ms: 18,
      patch: {
        trigger_matches: [
          {
            row_id: 'TM02',
            condition: 'explain + confused + beginner',
            special_handling: 'none',
            primary: { strategy_id: 'S05', strategy_name: 'Re-explain Differently' },
            supporting: [
              { strategy_id: 'S02', strategy_name: 'Example / Analogy' },
              { strategy_id: 'S07', strategy_name: 'Check Understanding' },
            ],
            priority: 70,
            matched: true,
            selected: true,
          },
          {
            row_id: 'TM01',
            condition: 'explain + normal + beginner',
            special_handling: 'none',
            primary: { strategy_id: 'S01', strategy_name: 'Direct Answer / Explanation' },
            supporting: [
              { strategy_id: 'S02', strategy_name: 'Example / Analogy' },
              { strategy_id: 'S07', strategy_name: 'Check Understanding' },
            ],
            priority: 50,
            matched: false,
            selected: false,
            excluded_reason: 'Row requires learner_state = normal; this turn detected confused.',
          },
        ],
        combination: {
          allowed: true,
          rule: 'S05 → {S02, S07} — Re-explain → Example → Check',
          dropped: [],
          note: 'Declared extension in combination_rules.yaml: §3.3 lists rows for four primaries, the Trigger Matrix pairing governs for the rest.',
        },
        teaching_policy: {
          row: 'normal_learning',
          row_label: 'Normal learning',
          direct_answer: 'allowed',
          attempt: 'optional',
          check: 'recommended',
          override_applied: false,
          precedence: 'Teaching Policy → Special Handling → Learner State → Intent → Student Level',
        },
        strategy: {
          primary_strategy: { strategy_id: 'S05', strategy_name: 'Re-explain Differently' },
          supporting_strategies: [
            { strategy_id: 'S02', strategy_name: 'Example / Analogy' },
            { strategy_id: 'S07', strategy_name: 'Check Understanding' },
          ],
          constraints: {
            direct_answer_allowed: true,
            full_solution_allowed_now: true,
            student_attempt_required: false,
          },
          knowledge_guidance: {
            action: 'check_or_scaffold_prerequisite',
            target_concept: 'One-Step Linear Equations · สมการเชิงเส้นขั้นตอนเดียว',
            relationship: 'prerequisite_of',
            distance: 1,
          },
          context: { topic: 'Two-Step Linear Equations', student_level: 'beginner' },
        },
      },
    },
    {
      delay_ms: 12,
      patch: {
        response_plan: {
          structure: ['reframed_explanation', 'one_example_or_analogy', 'single_check_question'],
          max_words: 120,
          max_questions: 1,
          language_level: 'beginner',
          language: 'th',
          full_solution_allowed: true,
          wait_for_student: false,
          template_id: 'S05 template',
          global_rules: {
            max_examples: 1,
            max_analogies: 1,
            max_questions: 1,
            source_reference_required: true,
            unsupported_claims_allowed: false,
            tone: 'clear_respectful_encouraging_non_patronizing',
          },
        },
      },
    },
  ],
  reply:
    'ลองคิดว่าสมการคือตาชั่งสองข้างที่ต้องหนักเท่ากันเสมอนะคะ ถ้าเราหยิบของออกจากข้างซ้าย 4 หน่วย ข้างขวาก็ต้องหยิบออก 4 หน่วยเท่ากัน ตาชั่งถึงจะยังสมดุล การลบ 4 ทั้งสองข้างใน 2x + 4 = 10 ก็คือการหยิบออกให้เท่ากันแบบนั้นค่ะ แล้วถ้าโจทย์เปลี่ยนเป็น 2x + 6 = 10 เราต้องหยิบอะไรออกจากทั้งสองข้างคะ',
  final: {
    guardrail: [
      {
        type: 'leak_scan_clean',
        severity: 'info',
        detail: 'full_solution_allowed = true; quiz-key scan ran anyway: no match',
        action_taken: 'served as written',
        at_ms: 2094,
      },
    ],
    timings: [
      { step: 'intent', label: 'Intent detection (Call A)', engine: 'llm', ms: 171 },
      { step: 'topic_level', label: 'Topic + student level', engine: 'deterministic', ms: 9 },
      { step: 'retrieval', label: 'RAG retrieval', engine: 'data', ms: 104 },
      { step: 'kl_lookup', label: 'KL Map lookup (depth 2)', engine: 'data', ms: 14 },
      { step: 'strategy_selection', label: 'Strategy selection', engine: 'deterministic', ms: 4 },
      { step: 'response_planning', label: 'Response planning', engine: 'deterministic', ms: 2 },
      { step: 'generation', label: 'Generation (Call C)', engine: 'llm', ms: 1790 },
      { step: 'guardrail', label: 'Output guardrail', engine: 'policy', ms: 5 },
    ],
    pre_stream_ms: 304,
    total_ms: 2099,
    versions: VERSIONS,
  },
};

// ---------------------------------------------------------------------------
// 3 · Hint request — hint-first beats an immediate full answer
// ---------------------------------------------------------------------------

const hint: MockScenario = {
  id: 'hint-en',
  label: 'Hint request — solution withheld by Combination Rules',
  language: 'en',
  prompt: 'Give me a hint for 3x - 7 = 8',
  matches: (m) => has(m, 'hint', 'clue', 'ใบ้'),
  stages: [
    {
      delay_ms: 165,
      patch: {
        intent: {
          intent: 'hint',
          learner_state: 'normal',
          special_handling: { detected: false, type: 'none' },
          confidence: 0.95,
        },
      },
    },
    {
      delay_ms: 50,
      patch: {
        topic: {
          topic: 'Two-Step Linear Equations',
          topic_th: 'สมการเชิงเส้นสองขั้นตอน',
          node_id: 'C009',
          student_level: 'beginner',
          matched_by: 'Syllabus match on the problem form 3x - 7 = 8',
          level_source: 'onboarding_questionnaire',
        },
      },
    },
    {
      delay_ms: 98,
      patch: {
        retrieval: [
          {
            chunk_id: 'ch1#p1',
            score: 0.77,
            text: 'Every operation has an inverse. Addition is undone by subtraction, multiplication by division.',
            source_ref: 'ch1-inverse-operations.md § What an inverse is',
            language: 'en',
          },
          {
            chunk_id: 'ch2#p2',
            score: 0.71,
            text: 'Undo addition or subtraction first, then undo multiplication or division.',
            source_ref: 'ch2-solving-linear-equations.md § Two-step equations',
            language: 'en',
          },
        ],
        knowledge_context: {
          current_topic: 'Two-Step Linear Equations',
          current_node_id: 'C009',
          prerequisites: [
            { topic: 'One-Step Linear Equations', topic_th: 'สมการเชิงเส้นขั้นตอนเดียว', node_id: 'C008', distance: 1 },
          ],
          related_topics: [
            { topic: 'Checking a Solution', topic_th: 'การตรวจคำตอบ', node_id: 'C013', distance: 1 },
          ],
          next_topics: [
            { topic: 'Word Problems to Equations', topic_th: 'โจทย์ปัญหาสู่สมการ', node_id: 'C014', distance: 1 },
          ],
          relationships: [],
          max_depth: 1,
          nodes_hit: ['C009', 'C008', 'C013', 'C014'],
        },
        km_rules: [
          {
            rule: 'KM01',
            condition: 'learner_state = normal',
            action: 'Current topic only',
            applied: true,
          },
        ],
      },
    },
    {
      delay_ms: 16,
      patch: {
        trigger_matches: [
          {
            row_id: 'TM08',
            condition: 'hint + any',
            special_handling: 'any',
            primary: { strategy_id: 'S04', strategy_name: 'Hint / Scaffold' },
            supporting: [{ strategy_id: 'S07', strategy_name: 'Check Understanding' }],
            priority: 65,
            matched: true,
            selected: true,
          },
        ],
        combination: {
          allowed: true,
          rule: 'S04 → {S03, S07} — Hint → Student tries → Guide',
          dropped: [{ strategy_id: 'S01', strategy_name: 'Direct Answer / Explanation' }],
          note: 'Hint-first always wins over an immediate full answer (§3.3), so the full solution is withheld this turn even though the policy row permits direct answers.',
        },
        teaching_policy: {
          row: 'normal_learning',
          row_label: 'Normal learning',
          direct_answer: 'allowed',
          attempt: 'optional',
          check: 'recommended',
          override_applied: false,
          override_note: 'Policy permits a direct answer; the restriction here comes from the Combination Rules, not the policy.',
          precedence: 'Teaching Policy → Special Handling → Learner State → Intent → Student Level',
        },
        strategy: {
          primary_strategy: { strategy_id: 'S04', strategy_name: 'Hint / Scaffold' },
          supporting_strategies: [{ strategy_id: 'S07', strategy_name: 'Check Understanding' }],
          constraints: {
            direct_answer_allowed: false,
            full_solution_allowed_now: false,
            student_attempt_required: true,
          },
          knowledge_guidance: {
            action: 'use_current_topic_context',
            target_concept: 'Inverse Operations',
            relationship: 'uses',
            distance: 1,
          },
          context: { topic: 'Two-Step Linear Equations', student_level: 'beginner' },
        },
      },
    },
    {
      delay_ms: 10,
      patch: {
        response_plan: {
          structure: ['single_hint', 'student_attempt_prompt'],
          max_words: 50,
          max_questions: 1,
          language_level: 'beginner',
          language: 'en',
          full_solution_allowed: false,
          wait_for_student: true,
          template_id: 'S04 template',
          global_rules: {
            max_examples: 1,
            max_questions: 1,
            hint_first_and_full_solution_same_turn: false,
            source_reference_required: true,
          },
        },
      },
    },
  ],
  reply:
    "Look at what is being done to x first: something is subtracted, then x is multiplied. Undo the subtraction before the multiplication. Try that first step and tell me what you get.",
  final: {
    guardrail: [
      {
        type: 'leak_scan_clean',
        severity: 'info',
        detail: 'canonical_answer scan (P04, exact + normalised): no match',
        action_taken: 'served as written',
        at_ms: 1246,
      },
      {
        type: 'plan_violation',
        severity: 'warning',
        detail: 'word_count 42 against max_words 50 (within the 25% tolerance)',
        action_taken: 'served, logged for prompt tuning',
        at_ms: 1247,
      },
    ],
    timings: [
      { step: 'intent', label: 'Intent detection (Call A)', engine: 'llm', ms: 158 },
      { step: 'topic_level', label: 'Topic + student level', engine: 'deterministic', ms: 8 },
      { step: 'retrieval', label: 'RAG retrieval', engine: 'data', ms: 92 },
      { step: 'kl_lookup', label: 'KL Map lookup', engine: 'data', ms: 7 },
      { step: 'strategy_selection', label: 'Strategy selection', engine: 'deterministic', ms: 3 },
      { step: 'response_planning', label: 'Response planning', engine: 'deterministic', ms: 2 },
      { step: 'generation', label: 'Generation (Call C)', engine: 'llm', ms: 972 },
      { step: 'guardrail', label: 'Output guardrail', engine: 'policy', ms: 4 },
    ],
    pre_stream_ms: 270,
    total_ms: 1246,
    versions: VERSIONS,
  },
};

// ---------------------------------------------------------------------------
// 4 · Pressure turn — low-confidence fallback, policy override, leak blocked
// ---------------------------------------------------------------------------

const pressure: MockScenario = {
  id: 'pressure-leak',
  label: 'Pressure turn — policy override + leak blocked',
  language: 'en',
  prompt: "Just tell me the answer to 2x + 4 = 10, it's due tomorrow and I'm out of time",
  matches: (m) => has(m, 'just tell me', 'give me the answer', 'บอกคำตอบ'),
  stages: [
    {
      delay_ms: 210,
      patch: {
        intent: {
          intent: 'explain',
          learner_state: 'normal',
          special_handling: { detected: true, type: 'homework' },
          confidence: 0.58,
          fallback_applied: true,
          fallback_reason:
            'Confidence 0.58 below the 0.60 threshold — classifier was split between solve and check_answer. Fell back to explain and logged the turn for review (§2.1).',
        },
      },
    },
    {
      delay_ms: 58,
      patch: {
        topic: {
          topic: 'Two-Step Linear Equations',
          topic_th: 'สมการเชิงเส้นสองขั้นตอน',
          node_id: 'C009',
          student_level: 'beginner',
          matched_by: 'Problem string 2x + 4 = 10 matched the syllabus topic directly',
          level_source: 'default_beginner',
        },
      },
    },
    {
      delay_ms: 110,
      patch: {
        retrieval: [
          {
            chunk_id: 'ch2#p2',
            score: 0.83,
            text: 'A two-step equation needs two inverse operations, applied in reverse order of operations.',
            source_ref: 'ch2-solving-linear-equations.md § Two-step equations',
            language: 'en',
          },
        ],
        knowledge_context: {
          current_topic: 'Two-Step Linear Equations',
          current_node_id: 'C009',
          prerequisites: [
            { topic: 'One-Step Linear Equations', topic_th: 'สมการเชิงเส้นขั้นตอนเดียว', node_id: 'C008', distance: 1 },
          ],
          related_topics: [
            { topic: 'Checking a Solution', topic_th: 'การตรวจคำตอบ', node_id: 'C013', distance: 1 },
          ],
          next_topics: [
            { topic: 'Word Problems to Equations', topic_th: 'โจทย์ปัญหาสู่สมการ', node_id: 'C014', distance: 1 },
          ],
          relationships: [],
          max_depth: 1,
          nodes_hit: ['C009', 'C008', 'C013', 'C014'],
        },
        km_rules: [
          { rule: 'KM01', condition: 'learner_state = normal', action: 'Current topic only', applied: true },
        ],
      },
    },
    {
      delay_ms: 20,
      patch: {
        trigger_matches: [
          {
            row_id: 'TM01',
            condition: 'explain + normal + beginner',
            special_handling: 'none',
            primary: { strategy_id: 'S01', strategy_name: 'Direct Answer / Explanation' },
            supporting: [
              { strategy_id: 'S02', strategy_name: 'Example / Analogy' },
              { strategy_id: 'S07', strategy_name: 'Check Understanding' },
            ],
            priority: 50,
            matched: true,
            selected: false,
            excluded_reason: 'Selected by the Trigger Matrix, then overridden by the Teaching Policy homework row.',
          },
          {
            row_id: 'TM10',
            condition: 'any + any',
            special_handling: 'assessment',
            primary: { strategy_id: 'S04', strategy_name: 'Hint / Scaffold' },
            supporting: [],
            priority: 90,
            matched: false,
            selected: false,
            excluded_reason: 'Assessment mode is disabled in the POC (decision D4) and was not detected.',
          },
        ],
        combination: {
          allowed: true,
          rule: 'S04 → {S03, S07} — Hint → Student tries → Guide',
          dropped: [
            { strategy_id: 'S01', strategy_name: 'Direct Answer / Explanation' },
            { strategy_id: 'S02', strategy_name: 'Example / Analogy' },
          ],
          note: 'S01 dropped by the policy override; hint-first and a full solution may never share a turn.',
        },
        teaching_policy: {
          row: 'homework',
          row_label: 'Homework',
          direct_answer: 'not_initially_hint_first',
          attempt: 'required_first',
          check: 'recommended',
          override_applied: true,
          override_note:
            'Trigger Matrix selected S01 Direct Answer / Explanation. The Homework policy row sits above it in precedence: no initial direct answer, student attempt required. The planner re-routed the turn to S04 Hint / Scaffold and the canonical solution was excluded from the generation prompt.',
          precedence: 'Teaching Policy → Special Handling → Learner State → Intent → Student Level',
        },
        strategy: {
          primary_strategy: { strategy_id: 'S04', strategy_name: 'Hint / Scaffold' },
          supporting_strategies: [{ strategy_id: 'S07', strategy_name: 'Check Understanding' }],
          constraints: {
            direct_answer_allowed: false,
            full_solution_allowed_now: false,
            student_attempt_required: true,
          },
          knowledge_guidance: {
            action: 'use_current_topic_context',
            target_concept: 'Two-Step Linear Equations',
            relationship: 'current_topic',
            distance: 0,
          },
          context: { topic: 'Two-Step Linear Equations', student_level: 'beginner' },
        },
      },
    },
    {
      delay_ms: 14,
      patch: {
        response_plan: {
          structure: ['single_hint', 'student_attempt_prompt'],
          max_words: 50,
          max_questions: 1,
          language_level: 'beginner',
          language: 'en',
          full_solution_allowed: false,
          wait_for_student: true,
          template_id: 'S04 template (policy-substituted)',
          global_rules: {
            hint_first_and_full_solution_same_turn: false,
            quiz_answer_before_student_attempt: false,
            source_reference_required: true,
          },
        },
      },
    },
  ],
  reply: "Let's work through it — what would you try first?",
  final: {
    guardrail: [
      {
        type: 'leak_regenerated',
        severity: 'warning',
        detail: 'canonical_answer matched as exact (P03, 1 char) in draft 1',
        action_taken: 'regenerated once with an appended system note',
        at_ms: 1489,
      },
      {
        type: 'leak_blocked',
        severity: 'critical',
        detail: 'canonical_answer matched as normalised (P03, 1 char) in draft 2',
        action_taken: 'generation discarded',
        at_ms: 2604,
      },
      {
        type: 'fallback_served',
        severity: 'warning',
        detail: 'template fallback (hint_first) served in place of the draft',
        action_taken: 'served to the student; counts as a guardrail save in E3',
        at_ms: 2606,
      },
    ],
    timings: [
      { step: 'intent', label: 'Intent detection (Call A)', engine: 'llm', ms: 204 },
      { step: 'topic_level', label: 'Topic + student level', engine: 'deterministic', ms: 10 },
      { step: 'retrieval', label: 'RAG retrieval', engine: 'data', ms: 101 },
      { step: 'kl_lookup', label: 'KL Map lookup', engine: 'data', ms: 8 },
      { step: 'strategy_selection', label: 'Strategy selection', engine: 'deterministic', ms: 4 },
      { step: 'response_planning', label: 'Response planning', engine: 'deterministic', ms: 2 },
      { step: 'generation', label: 'Generation (Call C, incl. 1 regeneration)', engine: 'llm', ms: 2260 },
      { step: 'guardrail', label: 'Output guardrail (2 scans)', engine: 'policy', ms: 15 },
    ],
    pre_stream_ms: 329,
    total_ms: 2604,
    versions: VERSIONS,
  },
};

// ---------------------------------------------------------------------------
// 5 · Fallback — plain explain turn used when nothing else matches
// ---------------------------------------------------------------------------

const explain: MockScenario = {
  id: 'explain-default',
  label: 'Explain — direct answer permitted',
  language: 'en',
  prompt: 'What is a linear equation?',
  matches: () => true,
  stages: [
    {
      delay_ms: 168,
      patch: {
        intent: {
          intent: 'explain',
          learner_state: 'normal',
          special_handling: { detected: false, type: 'none' },
          confidence: 0.91,
        },
      },
    },
    {
      delay_ms: 48,
      patch: {
        topic: {
          topic: 'Variables and Expressions',
          topic_th: 'ตัวแปรและนิพจน์',
          node_id: 'C004',
          student_level: 'beginner',
          matched_by: 'Embedding match against the syllabus topic list',
          level_source: 'default_beginner',
        },
      },
    },
    {
      delay_ms: 102,
      patch: {
        retrieval: [
          {
            chunk_id: 'ch2#p1',
            score: 0.75,
            text: 'A linear equation states that two expressions are equal, and the variable appears only to the first power.',
            source_ref: 'ch2-solving-linear-equations.md § What a linear equation is',
            language: 'en',
          },
        ],
        knowledge_context: {
          current_topic: 'Variables and Expressions',
          current_node_id: 'C004',
          prerequisites: [],
          related_topics: [
            { topic: 'Evaluating Expressions', topic_th: 'การหาค่านิพจน์', node_id: 'C005', distance: 1 },
          ],
          next_topics: [],
          relationships: ['evaluating expressions is part of variables and expressions'],
          max_depth: 1,
          nodes_hit: ['C004', 'C005'],
        },
        km_rules: [
          { rule: 'KM01', condition: 'learner_state = normal', action: 'Current topic only', applied: true },
        ],
      },
    },
    {
      delay_ms: 15,
      patch: {
        trigger_matches: [
          {
            row_id: 'TM01',
            condition: 'explain + normal + beginner',
            special_handling: 'none',
            primary: { strategy_id: 'S01', strategy_name: 'Direct Answer / Explanation' },
            supporting: [
              { strategy_id: 'S02', strategy_name: 'Example / Analogy' },
              { strategy_id: 'S07', strategy_name: 'Check Understanding' },
            ],
            priority: 50,
            matched: true,
            selected: true,
          },
        ],
        combination: {
          allowed: true,
          rule: 'S01 → {S02, S07} — Answer → Example → Check',
          dropped: [],
        },
        teaching_policy: {
          row: 'normal_learning',
          row_label: 'Normal learning',
          direct_answer: 'allowed',
          attempt: 'optional',
          check: 'recommended',
          override_applied: false,
          precedence: 'Teaching Policy → Special Handling → Learner State → Intent → Student Level',
        },
        strategy: {
          primary_strategy: { strategy_id: 'S01', strategy_name: 'Direct Answer / Explanation' },
          supporting_strategies: [
            { strategy_id: 'S02', strategy_name: 'Example / Analogy' },
            { strategy_id: 'S07', strategy_name: 'Check Understanding' },
          ],
          constraints: {
            direct_answer_allowed: true,
            full_solution_allowed_now: true,
            student_attempt_required: false,
          },
          knowledge_guidance: {
            action: 'use_current_topic_context',
            target_concept: 'Variables and Expressions',
            relationship: 'current_topic',
            distance: 0,
          },
          context: { topic: 'Variables and Expressions', student_level: 'beginner' },
        },
      },
    },
    {
      delay_ms: 11,
      patch: {
        response_plan: {
          structure: ['direct_answer', 'explanation', 'source_reference', 'check_understanding'],
          max_words: 120,
          max_questions: 1,
          language_level: 'beginner',
          language: 'en',
          full_solution_allowed: true,
          wait_for_student: false,
          template_id: 'S01 template',
          global_rules: {
            max_examples: 1,
            max_questions: 1,
            source_reference_required: true,
            unsupported_claims_allowed: false,
          },
        },
      },
    },
  ],
  reply:
    'A linear equation says two expressions are equal, and the unknown appears only to the first power — like 2x + 4 = 10. Solving it means finding the value of x that makes both sides equal. (Source: ch2-solving-linear-equations.md) Which part of that would you like to try on an example first?',
  final: {
    guardrail: [
      {
        type: 'leak_scan_clean',
        severity: 'info',
        detail: 'full_solution_allowed = true; no withheld content applies to this turn',
        action_taken: 'served as written',
        at_ms: 1522,
      },
    ],
    timings: [
      { step: 'intent', label: 'Intent detection (Call A)', engine: 'llm', ms: 162 },
      { step: 'topic_level', label: 'Topic + student level', engine: 'deterministic', ms: 8 },
      { step: 'retrieval', label: 'RAG retrieval', engine: 'data', ms: 95 },
      { step: 'kl_lookup', label: 'KL Map lookup', engine: 'data', ms: 6 },
      { step: 'strategy_selection', label: 'Strategy selection', engine: 'deterministic', ms: 3 },
      { step: 'response_planning', label: 'Response planning', engine: 'deterministic', ms: 2 },
      { step: 'generation', label: 'Generation (Call C)', engine: 'llm', ms: 1242 },
      { step: 'guardrail', label: 'Output guardrail', engine: 'policy', ms: 4 },
    ],
    pre_stream_ms: 276,
    total_ms: 1522,
    versions: VERSIONS,
  },
};

/** Order matters: the first scenario whose `matches` returns true wins. */
export const SCENARIOS: MockScenario[] = [homework, confusedThai, hint, pressure, explain];

export function pickScenario(message: string): MockScenario {
  return SCENARIOS.find((s) => s.matches(message)) ?? explain;
}
