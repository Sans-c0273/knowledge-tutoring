#!/usr/bin/env node
/**
 * Post-build guard: the shipped bundle must contain the real HTTP client.
 *
 * Why this exists (diagnosis 2026-09-02, bug A / S7): `USE_MOCKS` is a build-time
 * constant. When it folds to `true`, Rollup tree-shakes `src/api/http.ts` away and
 * `web/dist` becomes a browser-side simulation that never calls the backend — while
 * looking, and passing `tsc`, exactly like a working client. No unit test can see
 * that, because unit tests never run the bundle. So the build itself checks that
 * the route strings only `http.ts` emits survived into `dist/assets/*.js`.
 *
 * Exit 1 with a message naming the missing markers; exit 0 when every marker is
 * present. Run from `web/` (or anywhere — paths resolve from this file).
 */
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const assets = resolve(here, '..', 'dist', 'assets');

/** Literals that appear in `src/api/http.ts` and nowhere in `src/mocks/`. */
const REQUIRED_MARKERS = ['ingest/upload', 'chat/turn'];

if (!existsSync(assets)) {
  console.error(`check-bundle: ${assets} does not exist — run \`vite build\` first.`);
  process.exit(1);
}

const bundles = readdirSync(assets).filter((name) => name.endsWith('.js'));
if (bundles.length === 0) {
  console.error(`check-bundle: no .js bundle under ${assets}.`);
  process.exit(1);
}

const source = bundles.map((name) => readFileSync(join(assets, name), 'utf8')).join('\n');
const missing = REQUIRED_MARKERS.filter((marker) => !source.includes(marker));

if (missing.length > 0) {
  console.error(
    `check-bundle: FAIL — ${bundles.join(', ')} lacks ${missing.map((m) => `'${m}'`).join(', ')}.\n` +
      'The real API client (src/api/http.ts) was tree-shaken out: USE_MOCKS folded to true ' +
      'at build time. The served SPA would never talk to the backend.',
  );
  process.exit(1);
}

console.log(`check-bundle: OK — ${bundles.join(', ')} contains ${REQUIRED_MARKERS.join(', ')}.`);
