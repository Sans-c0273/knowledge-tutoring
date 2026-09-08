import { useEffect, useState } from 'react';
import { OPERATOR_MODE, USE_MOCKS } from './config';
import { useChat } from './hooks/useChat';
import { useKlMap } from './hooks/useKlMap';
import { IngestView } from './components/IngestView';
import { TutorView } from './components/TutorView';
import { KlMapView } from './components/klmap/KlMapView';
import { ReviewView } from './components/review/ReviewView';

const ROUTES = {
  '#/ingest': 'ingest',
  '#/tutor': 'tutor',
  '#/klmap': 'klmap',
  '#/review': 'review',
} as const;

type Route = (typeof ROUTES)[keyof typeof ROUTES];

/** Everything except the conversation is an operator surface (see `OPERATOR_MODE`). */
const OPERATOR_ROUTES = new Set<Route>(['ingest', 'klmap', 'review']);

function currentRoute(): Route {
  const route = ROUTES[window.location.hash as keyof typeof ROUTES] ?? 'tutor';
  // A student reaching an operator route by URL lands on the chat, not a blank page.
  return !OPERATOR_MODE && OPERATOR_ROUTES.has(route) ? 'tutor' : route;
}

export function App() {
  const [route, setRoute] = useState<Route>(currentRoute);
  // Chat and map state live above the router so a turn survives navigation.
  const chat = useChat();
  const { map, error: mapError } = useKlMap(OPERATOR_MODE);

  useEffect(() => {
    const onHashChange = () => {
      const resolved = currentRoute();
      // A redirected operator route rewrites the hash, so the URL never claims to
      // be somewhere the view is not.
      if (window.location.hash !== `#/${resolved}`) {
        window.history.replaceState(null, '', `#/${resolved}`);
      }
      setRoute(resolved);
    };
    onHashChange();
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const go = (next: Route) => {
    window.location.hash = `#/${next}`;
    setRoute(next);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">
          Socratic AI Tutor<span>{OPERATOR_MODE ? 'glass-box POC' : 'tutor'}</span>
        </div>
        <nav className="nav">
          {OPERATOR_MODE && (
            <button type="button" aria-current={route === 'ingest' ? 'page' : undefined} onClick={() => go('ingest')}>
              Ingestion
            </button>
          )}
          <button type="button" aria-current={route === 'tutor' ? 'page' : undefined} onClick={() => go('tutor')}>
            Tutor
          </button>
          {OPERATOR_MODE && (
            <>
              <button type="button" aria-current={route === 'klmap' ? 'page' : undefined} onClick={() => go('klmap')}>
                Knowledge map
              </button>
              <button type="button" aria-current={route === 'review' ? 'page' : undefined} onClick={() => go('review')}>
                Map review
              </button>
            </>
          )}
        </nav>
        <div className="header-right">
          <span className={`badge ${USE_MOCKS ? 'mock' : 'live'}`}>{USE_MOCKS ? 'mock data' : 'live API'}</span>
          {OPERATOR_MODE ? (
            <span className="badge operator" title="Diagnostic surfaces are visible. Not for a student-facing screen.">
              operator view
            </span>
          ) : (
            <span className="badge">student view</span>
          )}
        </div>
      </header>

      <main className="app-main">
        {route === 'ingest' && <IngestView />}
        {route === 'tutor' && (
          <TutorView chat={chat} map={map} mapError={mapError} onOpenFullMap={() => go('klmap')} />
        )}
        {route === 'klmap' && <KlMapView map={map} error={mapError} trace={chat.activeTrace} />}
        {route === 'review' && <ReviewView />}
      </main>
    </div>
  );
}
