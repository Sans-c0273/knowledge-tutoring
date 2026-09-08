/**
 * R18 fixtures: two extracted drafts awaiting review.
 *
 * Issue codes, messages and `location` strings follow `domain/klmap/loader.py`
 * (`KL0xx` errors, `KL1xx` warnings) and `domain/rag/kl_extract.py` (`KX1xx`
 * warnings for proposals the extractor dropped). The first draft is deliberately
 * unapprovable — a prerequisite cycle, a relation declared both ways, and an edge
 * pointing at a concept that does not exist — because the reviewer's main job is
 * seeing exactly that, not admiring a clean graph.
 */

import type { KlMapDraft } from '../types';

const PHYS: KlMapDraft = {
  course_id: 'PHYS-101',
  course_name: 'Newtonian Motion (extracted draft)',
  path: '/Users/tanat/src/socratic-tutor-poc/content/drafts/klmap-PHYS-101.yaml',
  extracted_at: '2026-09-01T14:22:09Z',
  status: 'draft',
  sources: [
    'phys101-week1-kinematics.pdf',
    'phys101-week2-newtons-laws.pptx',
    'lecture-03-friction-transcript.vtt',
  ],
  nodes: [
    { id: 'C001', name: 'Scalars and Vectors', name_th: 'ปริมาณสเกลาร์และเวกเตอร์' },
    { id: 'C002', name: 'Displacement', name_th: 'การกระจัด' },
    { id: 'C003', name: 'Velocity', name_th: 'ความเร็ว' },
    { id: 'C004', name: 'Acceleration', name_th: 'ความเร่ง' },
    { id: 'C005', name: "Newton's First Law", name_th: 'กฎข้อที่หนึ่งของนิวตัน' },
    { id: 'C006', name: "Newton's Second Law", name_th: 'กฎข้อที่สองของนิวตัน' },
    { id: 'C007', name: 'Force Diagrams', name_th: 'แผนภาพแรง' },
    { id: 'C008', name: 'Mass and Weight', name_th: 'มวลและน้ำหนัก' },
    { id: 'C009', name: 'Friction', name_th: 'แรงเสียดทาน' },
    { id: 'C010', name: 'Momentum', name_th: 'โมเมนตัม' },
  ],
  edges: [
    { from: 'C001', to: 'C002', relation: 'prerequisite_of' },
    { from: 'C002', to: 'C003', relation: 'prerequisite_of' },
    { from: 'C003', to: 'C004', relation: 'prerequisite_of' },
    { from: 'C004', to: 'C006', relation: 'prerequisite_of' },
    { from: 'C006', to: 'C004', relation: 'prerequisite_of' },
    { from: 'C005', to: 'C006', relation: 'next_topic' },
    { from: 'C007', to: 'C006', relation: 'uses' },
    { from: 'C008', to: 'C006', relation: 'part_of' },
    { from: 'C009', to: 'C007', relation: 'related_to' },
    { from: 'C011', to: 'C009', relation: 'prerequisite_of' },
    { from: 'C001', to: 'C002', relation: 'prerequisite_of' },
  ],
  report: {
    source: '/Users/tanat/src/socratic-tutor-poc/content/drafts/klmap-PHYS-101.yaml',
    errors: [
      {
        code: 'KL009',
        location: 'edges',
        message:
          "'prerequisite_of' edges form a cycle: C004 (Acceleration) -> C006 (Newton's Second Law) -> C004 (Acceleration). Prerequisites must form a DAG; reverse or remove one edge in that path.",
      },
      {
        code: 'KL008',
        location: 'edges[4]',
        message:
          "Relation 'prerequisite_of' is declared in both directions between C004 (Acceleration) and C006 (Newton's Second Law) (also at edges[3]). Each relation gets one declared direction; keep the correct one and delete the other.",
      },
      {
        code: 'KL006',
        location: 'edges[9]',
        message:
          "Edge 'C011' -> 'C009' (prerequisite_of) references unknown node id 'C011'. Add the node or fix the id.",
      },
    ],
    warnings: [
      {
        code: 'KL100',
        location: 'edges[10]',
        message: "Duplicate edge 'C001' -> 'C002' (prerequisite_of); the repeat has no effect.",
      },
      {
        code: 'KL101',
        location: 'nodes[9]',
        message:
          'Concept C010 (Momentum) has no edges, so lookup can never reach it from another topic.',
      },
      {
        code: 'KX102',
        location: null,
        message:
          "Dropped proposed edge 'C012' -related_to-> 'C003': no concept has id 'C012'.",
      },
      {
        code: 'KX103',
        location: null,
        message:
          "Dropped proposed edge: C007 (Force Diagrams) points at itself via 'related_to'.",
      },
    ],
  },
};

const MATH: KlMapDraft = {
  course_id: 'MATH-SEED-01',
  course_name: 'Linear Equations (re-extraction)',
  path: '/Users/tanat/src/socratic-tutor-poc/content/drafts/klmap-MATH-SEED-01.yaml',
  extracted_at: '2026-09-01T15:04:41Z',
  status: 'draft',
  sources: ['ch1-inverse-operations.md', 'ch2-solving-linear-equations.md', 'video-transcript-distributive.md'],
  nodes: [
    { id: 'C001', name: 'Basic Arithmetic', name_th: 'เลขคณิตพื้นฐาน' },
    { id: 'C002', name: 'Order of Operations', name_th: 'ลำดับการดำเนินการ' },
    { id: 'C004', name: 'Variables and Expressions', name_th: 'ตัวแปรและนิพจน์' },
    { id: 'C005', name: 'Evaluating Expressions', name_th: 'การหาค่านิพจน์' },
    { id: 'C006', name: 'Inverse Operations', name_th: 'การดำเนินการผกผัน' },
    { id: 'C008', name: 'One-Step Linear Equations', name_th: 'สมการเชิงเส้นขั้นตอนเดียว' },
    { id: 'C009', name: 'Two-Step Linear Equations', name_th: 'สมการเชิงเส้นสองขั้นตอน' },
    { id: 'C013', name: 'Checking a Solution', name_th: 'การตรวจคำตอบ' },
    { id: 'C015', name: 'Linear Inequalities', name_th: 'อสมการเชิงเส้น' },
  ],
  edges: [
    { from: 'C001', to: 'C002', relation: 'prerequisite_of' },
    { from: 'C002', to: 'C005', relation: 'prerequisite_of' },
    { from: 'C004', to: 'C005', relation: 'prerequisite_of' },
    { from: 'C005', to: 'C008', relation: 'prerequisite_of' },
    { from: 'C006', to: 'C008', relation: 'prerequisite_of' },
    { from: 'C008', to: 'C009', relation: 'prerequisite_of' },
    { from: 'C013', to: 'C009', relation: 'related_to' },
    { from: 'C005', to: 'C004', relation: 'part_of' },
  ],
  report: {
    source: '/Users/tanat/src/socratic-tutor-poc/content/drafts/klmap-MATH-SEED-01.yaml',
    errors: [],
    warnings: [
      {
        code: 'KL101',
        location: 'nodes[8]',
        message:
          'Concept C015 (Linear Inequalities) has no edges, so lookup can never reach it from another topic.',
      },
      {
        code: 'KX104',
        location: null,
        message:
          'Dropped duplicate proposed edge C006 (Inverse Operations) -prerequisite_of-> C008 (One-Step Linear Equations).',
      },
    ],
  },
};

export const MOCK_DRAFTS: KlMapDraft[] = [PHYS, MATH];
