/**
 * Single switch between the real FastAPI backend and the browser-side mocks.
 *
 * The real API is the default; mocks are opt-in with `VITE_USE_MOCKS=true npm run dev`
 * for UI work that must not touch the backend. The polarity is load-bearing because
 * this is a build-time constant: when it folds to `true`, Rollup tree-shakes
 * `src/api/http.ts` out of `web/dist` and the served app is a simulation that never
 * calls `fetch`, while looking exactly like a working client. The previous default
 * ("mocks unless told otherwise") shipped precisely that bundle (2026-09-02);
 * `scripts/check-bundle.mjs` now fails the build if it happens again.
 *
 * Only `src/api/index.ts` selects on this flag; the header badge and the scenario
 * bar read it for display.
 */
export const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true';

export const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '::1', '[::1]', '']);

/**
 * Operator mode gates every diagnostic surface: the glass-box inspector, the
 * knowledge-map panels, ingestion, and KL Map review.
 *
 * **Why this is a boundary and not a preference.** The trace is an operator
 * artefact. Guardrail events describe what was withheld and why, the retrieval
 * section carries course evidence, and the review screen can make a map live —
 * none of that belongs in front of a student, and the inspector sits beside the
 * chat where a student would be reading. Defaulting off anywhere that is not a
 * local single-user machine means nobody has to remember to hide it.
 *
 * **What this flag is not:** a security control on the data itself. Hiding a
 * panel does not stop the trace arriving over the wire, so anyone with the
 * browser's network tab still sees it. In a multi-user deployment the server
 * must be the boundary — send `trace` frames only to an authenticated operator
 * session, and keep the review and ingestion endpoints behind the same check.
 * This flag is the second layer, not the first.
 */
export const OPERATOR_MODE = resolveOperatorMode();

function resolveOperatorMode(): boolean {
  const configured = import.meta.env.VITE_OPERATOR_MODE;
  if (configured === 'true') return true;
  if (configured === 'false') return false;
  // Unset: on for a local single-user run, off everywhere else.
  return typeof window !== 'undefined' && LOCAL_HOSTS.has(window.location.hostname);
}

/**
 * Fixed identifiers for the single-user, no-auth POC (addendum §1, D5).
 *
 * `course_id` is what every upload is filed under and every chat turn is asked
 * about. Its default is the seed course, so the first approved upload replaces the
 * seed's syllabus and map in chat for that id (the seed answer keys stay armed —
 * TECH_DEBT TD6). Set `VITE_COURSE_ID` at build or dev time to work in a course of
 * your own instead; there is no in-app way to pick one yet.
 */
export const SESSION = {
  session_id: 'sess-local-01',
  student_id: 'student-local-01',
  // `||`, not `??`: an exported-but-empty `VITE_COURSE_ID=` must fall back too,
  // or every turn is a 422 from the backend's id validation.
  course_id: import.meta.env.VITE_COURSE_ID || 'MATH-SEED-01',
};
