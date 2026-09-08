import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import type { DraftDecision, KlMapDraft } from '../types';

/**
 * The KL Map review queue (R18). Decisions and re-validations return the updated
 * draft, which replaces the local copy — the server stays the authority on whether
 * a map is approvable.
 */
export function useDrafts() {
  const [drafts, setDrafts] = useState<KlMapDraft[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listDrafts()
      .then((loaded) => {
        if (cancelled) return;
        setDrafts(loaded);
        setSelectedId((current) => current ?? loaded[0]?.course_id ?? null);
      })
      .catch((loadError: unknown) => {
        if (!cancelled) setError(loadError instanceof Error ? loadError.message : String(loadError));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const replace = useCallback((updated: KlMapDraft) => {
    setDrafts((current) =>
      current.map((draft) => (draft.course_id === updated.course_id ? updated : draft)),
    );
  }, []);

  const decide = useCallback(
    async (courseId: string, decision: DraftDecision) => {
      setBusy(true);
      setError(null);
      setNotice(null);
      try {
        const updated = await api.decideDraft(courseId, decision);
        replace(updated);
        setNotice(
          decision.approve
            ? `${courseId} approved — it is now the live map for that course.`
            : `${courseId} rejected. The draft stays on disk for editing.`,
        );
      } catch (decideError) {
        setError(decideError instanceof Error ? decideError.message : String(decideError));
      } finally {
        setBusy(false);
      }
    },
    [replace],
  );

  const revalidate = useCallback(
    async (courseId: string) => {
      setBusy(true);
      setError(null);
      setNotice(null);
      try {
        const previous = drafts.find((draft) => draft.course_id === courseId);
        const updated = await api.revalidateDraft(courseId);
        replace(updated);
        const before = previous ? previous.report.errors.length : -1;
        const after = updated.report.errors.length;
        setNotice(
          before === after
            ? `Re-read ${updated.path.split('/').pop()} — ${after} blocking error(s), unchanged.`
            : `Re-read ${updated.path.split('/').pop()} — blocking errors ${before} → ${after}.`,
        );
      } catch (revalidateError) {
        setError(
          revalidateError instanceof Error ? revalidateError.message : String(revalidateError),
        );
      } finally {
        setBusy(false);
      }
    },
    [drafts, replace],
  );

  return {
    drafts,
    selected: drafts.find((draft) => draft.course_id === selectedId) ?? null,
    selectedId,
    select: setSelectedId,
    loading,
    error,
    busy,
    notice,
    decide,
    revalidate,
    dismissNotice: () => setNotice(null),
  };
}
