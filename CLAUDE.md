# socratic-tutor-azure

This is the active workspace for the Azure AI Foundry-wired copies of three
originally-independent projects: `kg_reasoner`, `knowledge-graph-poc`,
`socratic-tutor-poc`, plus the shared `azure-foundry-adapter` client code.

## Working conventions

- **Make all code edits inside this folder**, not in the sibling
  `../kg_reasoner`, `../knowledge-graph-poc`, `../socratic-tutor-poc`, or
  `../azure-foundry-adapter` directories at `/home/sans/socratic_tutor/`. Those
  are either the pristine originals (untouched on purpose — separate GitHub
  remotes owned by other accounts) or the earlier zero-edit adapter-only
  folder. This repo's copies have no `.git` history/remote ties back to the
  originals, so edits here are safe to push freely.
- **After a code change, commit and push to `origin` (`main` branch)
  automatically** — this repo's remote is
  `https://github.com/Sans-c0273/knowledge-tutoring.git`, the user's own repo,
  standing push authorization given 2026-09-08. No need to ask before each
  push; do still surface what was pushed.
- `.env` files in each project subfolder hold real Azure credentials and are
  gitignored — never add/commit them.

## Push authentication

No persistent GitHub credential is configured on this machine (no `gh` CLI,
no working SSH key registered with `github.com`, no stored credential
helper). The last push used a personal access token pasted directly in chat
for one-time use (not stored anywhere on disk) — that token should be treated
as revoked/rotated by the user afterward. If a future push is needed and no
stored credential exists, ask the user for a fresh token (or offer to set up
SSH/`gh auth login` once, so this stops being a per-push ask).
