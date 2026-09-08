import { useEffect, useState } from 'react';
import { api } from '../api';
import type { KlMap } from '../types';

/** Loads the KL Map for `courseId`, re-fetching whenever it changes (the
 * Knowledge Map view's course switcher passes a different id than the active
 * session's). */
export function useKlMap(courseId: string, enabled = true) {
  const [map, setMap] = useState<KlMap | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setMap(null);
    setError(null);
    api
      .getKlMap(courseId)
      .then((loaded) => {
        if (!cancelled) setMap(loaded);
      })
      .catch((loadError: unknown) => {
        if (cancelled) return;
        const message = loadError instanceof Error ? loadError.message : String(loadError);
        // No approved map is the normal state of a fresh install, not a failure:
        // R18 says an unapproved map never serves traffic.
        setError(
          /no kl map/i.test(message)
            ? 'No approved map for this course yet. Approve one in Map review to make it live.'
            : message,
        );
      });
    return () => {
      cancelled = true;
    };
  }, [courseId, enabled]);

  return { map, error };
}
