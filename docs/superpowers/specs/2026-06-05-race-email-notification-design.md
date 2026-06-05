# Email notification for RACE runs (Resend)

**Date:** 2026-06-05
**Status:** approved

## Problem

RACE runs take hours; today the user must keep the RACE hub session and
laptop open, then manually copy the driver's terminal output ("paste
everything above") back into Claude for analysis. When a run crashes
overnight, nobody notices until morning.

## Goal

When any `race-*.sh` driver finishes (success **or** crash), automatically
email the user a markdown run report (human- and AI-readable) plus the full
log, via the Resend API. The user then pastes/forwards the report for
analysis.

## Approach

**Generic wrapper script** — zero edits to the 9 existing drivers; crash
coverage via bash trap; works for all future drivers.

```
nohup bash scripts/notify-run.sh bash scripts/race-revisit.sh --tag wide-1 &
```

## Components

### 1. `scripts/notify-run.sh`

- `set -uo pipefail`; resolve repo root from script path (same pattern as
  the existing `race-*.sh`).
- Record `START_TS`, `GIT_COMMIT` (short), the full command string, and a
  TAG (first `--tag` value in args, else the wrapped script basename).
- Run `"$@" 2>&1 | tee runs/notify-${TAG}-<timestamp>.log`; capture exit
  code via `PIPESTATUS[0]`.
- `trap ... EXIT` calls `python scripts/notify_email.py --exit-code $EC
  --log $LOG --command "$CMD" --start-ts $START_TS --commit $GIT_COMMIT
  --tag $TAG` — fires on normal exit, crash, and Ctrl-C/SIGTERM.
- Wrapper exits with the wrapped command's exit code; a notifier failure
  never changes it (`|| true`).

### 2. `scripts/notify_email.py` (stdlib-only)

No new deps on the RACE conda env — `urllib.request`, `json`, `base64`,
`gzip`, `argparse`, `pathlib`, `socket`, `subprocess` only.

**Config:** parse `.env` at repo root (simple `KEY=VALUE`, ignore
comments/blanks, strip optional quotes). Real environment variables
override `.env`. Keys: `RESEND_API_KEY` (required), `NOTIFY_EMAIL_TO`
(required), `NOTIFY_EMAIL_FROM` (default `onboarding@resend.dev`),
`NOTIFY_LOG_TAIL_LINES` (default 400). Missing key/recipient → one-line
warning, exit 0. **Notification is never load-bearing.**

**Report (`build_report`, pure function):**
1. Header: ✅/❌ by exit code, command, exit code, duration (humanized),
   git commit, hostname, start/end ISO timestamps.
2. Run digests: `runs/*/summary.json` with mtime > start_ts → markdown
   table with dir, `ablation.setting`, episodes completed/attempted, mean
   `soft_spl` over `episodes[]`, `n_memory_chosen`, `ltm_counts_final`,
   failed `pass_conditions`. Malformed/missing summary.json → skip with a
   note.
3. Terminal output: last N lines of the log in a fenced code block — on
   success this is the analyzer output verbatim (what the user currently
   pastes); on failure, the traceback. Verbatim tail = no fragile parsing.

**Send:** `POST https://api.resend.com/emails`, Bearer auth. Subject
`{✅|❌} [ltm] {tag} — exit {code} ({duration})`; `text` = the markdown
report; attachments = `report.md` + `{logname}.log.gz` (base64). If the
gzipped log > ~35 MB (Resend total limit 40 MB), gzip only the tail and
note the truncation. 3 attempts with 2s/8s backoff; on final failure write
`report.md` next to the log and print its path; always exit 0.

### 3. `.env.example` (committed)

```
RESEND_API_KEY=re_xxxxxxxx
NOTIFY_EMAIL_TO=you@example.com
# Free tier: onboarding@resend.dev can send to your own Resend account email, no domain needed
NOTIFY_EMAIL_FROM=onboarding@resend.dev
# NOTIFY_LOG_TAIL_LINES=400
```

`.env` is gitignored (verified). Real `.env` is created once on RACE by hand.

### 4. Convenience alias

`scripts/race-setup.sh` gains `nrun() { bash "$REPO_ROOT/scripts/notify-run.sh" "$@"; }`
so the RACE command becomes `nrun bash scripts/race-revisit.sh --tag t`.

## Not changing

The 9 `race-*.sh` drivers, analyzers, and Python runners — zero edits
(drivers already tee their own logs; the wrapper's tee is additive and is
the one the notifier reads).

## Testing

`scripts/test_notify_email.py` — repo-convention standalone assert suite:
env parser, build_report (fixture summary.json on the `runs/abl-s3-qwen`
schema), run-dir discovery by mtime, mocked Resend payload (URL, auth,
subject, base64 attachments, retry, missing-config exit 0), and a local
wrapper smoke (`notify-run.sh false` → exit 1 preserved, warning, log
created).

End-to-end: real key in local `.env`, `bash scripts/notify-run.sh echo
hello`, confirm email lands. Then on RACE: pull, create `.env`, wrap a
short `race-smoke.sh` run.
