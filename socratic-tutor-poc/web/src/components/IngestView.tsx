import { useRef, useState } from 'react';
import { useIngest } from '../hooks/useIngest';
import type { IngestFile, IngestStage } from '../types';
import { INGEST_STAGES } from '../types';
import { Pill } from './ui';

const ACCEPT = '.pdf,.pptx,.docx,.md,.txt,.vtt,.srt';

/** R17 — upload / URL intake with per-file stage + percentage progress from SSE. */
export function IngestView() {
  const { files, error, busy, uploadFiles, submitUrl, dismissError } = useIngest();
  const [over, setOver] = useState(false);
  const [url, setUrl] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="ingest">
      <div className="ingest-inner">
        <h1>Content ingestion</h1>
        <p className="lead">
          Course materials enter the pipeline here: parsed, chunked, embedded with local BGE-M3, then mined for
          knowledge-map candidates that wait for human review before they can serve traffic.
        </p>

        <div
          className={`dropzone${over ? ' over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            void uploadFiles(Array.from(e.dataTransfer.files));
          }}
        >
          <strong>Drop course files here</strong>
          <span className="muted small">or</span>
          <div style={{ marginTop: 10 }}>
            <button type="button" className="btn primary" disabled={busy} onClick={() => inputRef.current?.click()}>
              Choose files
            </button>
          </div>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT}
            hidden
            onChange={(e) => {
              void uploadFiles(Array.from(e.target.files ?? []));
              e.target.value = '';
            }}
          />
          <div className="formats">
            PDF · PPTX · DOCX · Markdown · TXT · video transcripts (.vtt, .srt). Videos are transcript-only in the
            POC — there is no speech recognition.
          </div>
        </div>

        <form
          className="url-row"
          onSubmit={(e) => {
            e.preventDefault();
            const trimmed = url.trim();
            if (!trimmed) return;
            void submitUrl(trimmed).then(() => setUrl(''));
          }}
        >
          <input
            type="url"
            placeholder="https://… page or transcript URL"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            aria-label="Content URL"
          />
          <button type="submit" className="btn" disabled={busy || url.trim().length === 0}>
            Ingest URL
          </button>
        </form>

        {error && (
          <div className="notice red" style={{ marginTop: 14 }}>
            {error}{' '}
            <button type="button" className="reveal-btn" onClick={dismissError}>
              dismiss
            </button>
          </div>
        )}

        <div className="file-list">
          {files.length === 0 && (
            <p className="muted small" style={{ marginTop: 18 }}>
              No files yet. Progress appears here per file as the pipeline advances through its five stages.
            </p>
          )}
          {files.map((file) => (
            <FileCard key={file.file_id} file={file} />
          ))}
        </div>
      </div>
    </div>
  );
}

const STAGE_TITLE: Record<IngestStage, string> = {
  parse: 'parse',
  chunk: 'chunk',
  embed: 'embed',
  'kl-extract': 'kl-extract',
  'review-ready': 'review-ready',
};

function FileCard({ file }: { file: IngestFile }) {
  const currentIndex = INGEST_STAGES.indexOf(file.stage);

  return (
    <div className={`file-card ${file.status}`}>
      <div className="file-head">
        <span className="file-name" title={file.filename}>
          {file.filename}
        </span>
        <Pill tone={file.source === 'url' ? 'blue' : 'grey'}>{file.source}</Pill>
        {file.status === 'error' && <Pill tone="red">failed</Pill>}
        {file.status === 'done' && <Pill tone="green">review-ready</Pill>}
        <span className="file-percent">{file.percent}%</span>
      </div>

      <div className="progress">
        <div style={{ width: `${file.percent}%` }} />
      </div>

      <div className="stage-rail">
        {INGEST_STAGES.map((stage, index) => {
          const failed = file.status === 'error' && index === currentIndex;
          const complete = file.status !== 'error' && (index < currentIndex || file.status === 'done');
          const active = file.status === 'running' && index === currentIndex;
          const className = failed ? 'failed' : complete ? 'complete' : active ? 'active' : '';
          return (
            <div key={stage} className={`stage ${className}`}>
              {STAGE_TITLE[stage]}
            </div>
          );
        })}
      </div>

      {file.message && (
        <div className={`file-message ${file.status === 'error' ? 'notice red' : 'muted small'}`}>{file.message}</div>
      )}
    </div>
  );
}
