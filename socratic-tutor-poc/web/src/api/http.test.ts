import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SESSION } from '../config';
import { httpApi } from './http';

/**
 * Regression for sibling S4 (diagnosis 2026-09-02): the SPA never sent a `course_id`
 * with an upload or a URL, so `ingest_routes` filed every source under its
 * `DEFAULT_COURSE_ID`. The client already knows the course (`SESSION.course_id`) and
 * must say so explicitly, so that what is ingested, reviewed and taught all share one id.
 */
function okJson(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('httpApi ingestion requests carry the course id (fix: S4)', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('test_fix_upload_carries_explicit_course_id: uploadFiles puts course_id in the form', async () => {
    fetchMock.mockResolvedValue(okJson([{ file_id: 'f-1', filename: 'notes.pdf' }]));
    const file = new File(['%PDF-1.4'], 'notes.pdf', { type: 'application/pdf' });

    await httpApi.uploadFiles([file]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/ingest\/upload$/);
    const form = init?.body;
    expect(form).toBeInstanceOf(FormData);
    expect((form as FormData).getAll('files')).toHaveLength(1);
    expect((form as FormData).get('course_id')).toBe(SESSION.course_id);
  });

  it('test_fix_upload_carries_explicit_course_id: submitUrl puts course_id in the JSON body', async () => {
    fetchMock.mockResolvedValue(okJson({ file_id: 'u-1', filename: 'https://example.org/x' }));

    await httpApi.submitUrl('https://example.org/x');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/ingest\/url$/);
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    expect(body.url).toBe('https://example.org/x');
    expect(body.course_id).toBe(SESSION.course_id);
  });
});
