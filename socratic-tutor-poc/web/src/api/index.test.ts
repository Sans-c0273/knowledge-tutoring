import { afterEach, describe, expect, it, vi } from 'vitest';

/**
 * Regression for the "UI looks great but does not really work" report (2026-09-02).
 *
 * `USE_MOCKS` defaulted to true whenever `VITE_USE_MOCKS` was unset, and `npm run build`
 * never set it — so Vite constant-folded the switch, tree-shook `httpApi` out of the
 * bundle, and the served `web/dist` was a browser-side simulation with no `fetch` in it.
 * The real backend must be the default; mocks are opt-in.
 */
async function loadApi() {
  vi.resetModules();
  const [{ api }, { httpApi }, { mockApi }] = await Promise.all([
    import('./index'),
    import('./http'),
    import('../mocks'),
  ]);
  return { api, httpApi, mockApi };
}

describe('api selection (fix: real API is the default)', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('test_fix_real_api_is_default: an unset VITE_USE_MOCKS selects the HTTP client', async () => {
    vi.stubEnv('VITE_USE_MOCKS', undefined);

    const { api, httpApi, mockApi } = await loadApi();

    expect(api).not.toBe(mockApi);
    expect(api).toBe(httpApi);
  });

  it('test_fix_real_api_is_default: VITE_USE_MOCKS=true opts into the mocks', async () => {
    vi.stubEnv('VITE_USE_MOCKS', 'true');

    const { api, mockApi } = await loadApi();

    expect(api).toBe(mockApi);
  });

  it('test_fix_real_api_is_default: VITE_USE_MOCKS=false still selects the HTTP client', async () => {
    vi.stubEnv('VITE_USE_MOCKS', 'false');

    const { api, httpApi } = await loadApi();

    expect(api).toBe(httpApi);
  });
});
