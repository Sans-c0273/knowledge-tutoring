import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { IngestEvent, IngestFile } from '../types';

/**
 * Holds the ingestion file list and keeps it in sync with the SSE progress stream
 * (R17). Events are the only source of stage/percent — uploads just register files.
 */
export function useIngest() {
  const [files, setFiles] = useState<IngestFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const orderRef = useRef<string[]>([]);

  useEffect(() => {
    const unsubscribe = api.subscribeIngest(
      (event: IngestEvent) => {
        setFiles((current) => applyEvent(current, event));
      },
      (streamError) => setError(streamError.message),
    );
    return unsubscribe;
  }, []);

  const uploadFiles = useCallback(async (selected: File[]) => {
    if (selected.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const accepted = await api.uploadFiles(selected);
      orderRef.current.push(...accepted.map((a) => a.file_id));
      setFiles((current) => seed(current, accepted.map((a) => ({ ...a, source: 'upload' as const }))));
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : String(uploadError));
    } finally {
      setBusy(false);
    }
  }, []);

  const submitUrl = useCallback(async (url: string) => {
    setBusy(true);
    setError(null);
    try {
      const accepted = await api.submitUrl(url);
      orderRef.current.push(accepted.file_id);
      setFiles((current) => seed(current, [{ ...accepted, source: 'url' as const }]));
    } catch (urlError) {
      setError(urlError instanceof Error ? urlError.message : String(urlError));
    } finally {
      setBusy(false);
    }
  }, []);

  return { files, error, busy, uploadFiles, submitUrl, dismissError: () => setError(null) };
}

function seed(
  current: IngestFile[],
  accepted: { file_id: string; filename: string; source: 'upload' | 'url' }[],
): IngestFile[] {
  const known = new Set(current.map((f) => f.file_id));
  const added: IngestFile[] = accepted
    .filter((a) => !known.has(a.file_id))
    .map((a) => ({
      file_id: a.file_id,
      filename: a.filename,
      source: a.source,
      stage: 'parse',
      percent: 0,
      status: 'queued',
      stage_progress: {},
      updated_at: new Date().toISOString(),
    }));
  return [...added.reverse(), ...current];
}

function applyEvent(current: IngestFile[], event: IngestEvent): IngestFile[] {
  const existing = current.find((f) => f.file_id === event.file_id);
  const base: IngestFile = existing ?? {
    file_id: event.file_id,
    filename: event.filename,
    source: 'upload',
    stage: event.stage,
    percent: 0,
    status: 'queued',
    stage_progress: {},
    updated_at: new Date().toISOString(),
  };

  const updated: IngestFile = {
    ...base,
    filename: event.filename || base.filename,
    stage: event.stage,
    percent: event.percent,
    status: event.status,
    message: event.message,
    stage_progress: { ...base.stage_progress, [event.stage]: event.percent },
    updated_at: new Date().toISOString(),
  };

  return existing
    ? current.map((f) => (f.file_id === event.file_id ? updated : f))
    : [updated, ...current];
}
