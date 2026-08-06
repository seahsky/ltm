#!/usr/bin/env python3
"""
Email a RACE run report via the Resend API.

Called by scripts/notify-run.sh from its EXIT trap — fires on normal
completion, crash, and Ctrl-C/SIGTERM. Builds a markdown report (header +
runs/*/summary.json digests + verbatim log tail), emails it with the
gzipped log attached, and ALWAYS exits 0: notification is never
load-bearing for the wrapped run.

stdlib-only on purpose — no new deps on the RACE conda env.

Config (.env at repo root; real environment variables override; empty
env var falls back to .env):
    RESEND_API_KEY          required to send
    NOTIFY_EMAIL_TO         required to send (comma-separated ok)
    NOTIFY_EMAIL_FROM       default onboarding@resend.dev
    NOTIFY_LOG_TAIL_LINES   default 400
    NOTIFY_DISABLE          set to 1 to skip sending entirely

Usage (normally via notify-run.sh):
    python scripts/notify_email.py --exit-code 0 --log runs/notify-x.log \
        --command "bash scripts/race-revisit.sh --tag x" \
        --start-ts 1770000000 --commit 2d518a3 --tag x
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

RESEND_URL = "https://api.resend.com/emails"
# Resend caps the whole request at 40 MB; leave headroom for report + JSON.
MAX_GZ_BYTES = 35 * 1024 * 1024
RETRY_BACKOFF_S = [2, 8]  # 3 attempts total


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

def parse_env_file(path: Path) -> dict:
    """Minimal KEY=VALUE parser: comments/blank lines ignored, optional
    single/double quotes stripped, whitespace around key/value stripped."""
    out = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def load_config(repo_root: Path, environ=None) -> dict:
    """Merge .env with real environment; non-empty env vars win."""
    if environ is None:
        environ = os.environ
    file_cfg = parse_env_file(Path(repo_root) / ".env")
    cfg = {}
    for key, default in (
        ("RESEND_API_KEY", ""),
        ("NOTIFY_EMAIL_TO", ""),
        ("NOTIFY_EMAIL_FROM", "onboarding@resend.dev"),
        ("NOTIFY_LOG_TAIL_LINES", "400"),
    ):
        val = environ.get(key) or file_cfg.get(key) or default
        cfg[key] = val
    try:
        cfg["NOTIFY_LOG_TAIL_LINES"] = int(cfg["NOTIFY_LOG_TAIL_LINES"])
    except (TypeError, ValueError):
        cfg["NOTIFY_LOG_TAIL_LINES"] = 400
    return cfg


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def humanize_duration(seconds) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def tail_lines(path: Path, n: int) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as e:
        return f"(log unreadable: {e})"
    lines = text.splitlines()
    return "\n".join(lines[-n:])


# ----------------------------------------------------------------------
# run digests
# ----------------------------------------------------------------------

# The funnel stages `RunSummary.as_dict` writes, and the two the digest leads with.
# `T_ANOM_REACHED` is the denominator the run's own summary() prints against (§6), so the
# rates here are the same rates the terminal showed rather than a second convention.
FUNNEL_DENOMINATOR = "T_ANOM_REACHED"
FUNNEL_REPORTED = ("ONSET_FIRED", "SOURCE_REACHED", "PRIMARY_RESUMED")


def _rate(numerator, denominator) -> str:
    if not isinstance(numerator, int) or not isinstance(denominator, int) or denominator <= 0:
        return "?"
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})"


def _digest_one(run_dir: Path, summary_path: Path, *, name: str = "") -> dict:
    """One `summary.json` as a report row.

    Reads the keys `RunSummary.as_dict` actually writes. It used to read
    `ablation.setting`, `n_memory_chosen` and `ltm_counts_final`, which belonged to the
    tree the 2026-08-06 reset deleted — so after the rebuild every column would have
    rendered `?` even once the file was found. A digest of question marks is worse than
    no digest: it reads as a run that produced nothing.
    """
    d = {"name": name or run_dir.name, "error": None}
    try:
        s = json.loads(summary_path.read_text())
    except (OSError, ValueError) as e:
        d["error"] = f"malformed summary.json ({e.__class__.__name__})"
        return d
    if not isinstance(s, dict):
        d["error"] = "malformed summary.json (not an object)"
        return d
    d["scene"] = s.get("scene", "?")
    built = s.get("n_episodes")
    skipped = s.get("n_skipped")
    d["built"] = built if isinstance(built, int) else "?"
    d["skipped"] = skipped if isinstance(skipped, int) else "?"
    offered = (built + skipped) if isinstance(built, int) and isinstance(skipped, int) else 0
    # None, not 0%, when nothing was offered — `yield_report.aggregate` draws the same
    # line: a yield of zero and no data are different claims.
    d["yield"] = f"{built / offered:.0%}" if offered else "?"
    funnel = s.get("funnel") if isinstance(s.get("funnel"), dict) else {}
    denominator = funnel.get(FUNNEL_DENOMINATOR)
    for stage in FUNNEL_REPORTED:
        d[stage] = _rate(funnel.get(stage), denominator)
    return d


# How deep under `runs/` a summary.json may sit. A single run writes
# runs/<tag>/summary.json; a SWEEP writes runs/<tag>/<scene>/summary.json, one level
# further down. Scanning only depth 1 is why the yield-1 sweep emailed "No summary.json
# updated during this run — none found" on the same page as "records:
# runs/yield-1/<scene>/summary.json": the digest section was structurally blind to every
# sweep this repo has ever run. Bounded rather than a recursive walk, so a deep runs/
# tree cannot turn a notifier into a filesystem crawl.
MAX_SUMMARY_DEPTH = 2

# `report.artifacts.RUN_SUMMARY_NAME`, spelled again. This module is stdlib-only and
# standalone by design so it cannot import that constant, and
# `test_report_artifacts.test_the_notifier_looks_for_exactly_this_name` is what keeps the
# two spellings honest. Named rather than inlined into the glob pattern below: the last
# version buried it inside a `"/".join(...)` expression, which is a literal the seam test
# cannot see and a rename this file would silently survive.
SUMMARY_NAME = "summary.json"


def discover_run_digests(runs_dir: Path, start_ts: float) -> list:
    """Digest every summary.json under runs/, to MAX_SUMMARY_DEPTH, modified after start_ts.
    Tolerates malformed files (digest with an 'error' note); never raises."""
    runs_dir = Path(runs_dir)
    digests = []
    if not runs_dir.is_dir():
        return digests
    for depth in range(1, MAX_SUMMARY_DEPTH + 1):
        pattern = "/".join(["*"] * depth + [SUMMARY_NAME])
        try:
            found = sorted(runs_dir.glob(pattern))
        except OSError:
            continue
        for summary in found:
            try:
                if not summary.is_file() or summary.stat().st_mtime <= start_ts:
                    continue
            except OSError:
                continue
            # Named by the path relative to runs/, not by the leaf: a sweep's rows are
            # all scene labels, and "ziup5kvtCCR" alone does not say which sweep.
            digests.append(_digest_one(summary.parent, summary,
                                       name=str(summary.parent.relative_to(runs_dir))))
    return digests


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------

def build_report(*, command, exit_code, start_ts, end_ts, commit, hostname,
                 log_tail, digests) -> str:
    ok = exit_code == 0
    marker = "✅" if ok else "❌"
    duration = humanize_duration(end_ts - start_ts)
    iso = lambda ts: datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    lines = [
        f"# {marker} RACE run {'finished' if ok else 'FAILED'}",
        "",
        f"- **Command:** `{command}`",
        f"- **Exit code:** {exit_code}",
        f"- **Duration:** {duration}",
        f"- **Git commit:** `{commit}`",
        f"- **Host:** {hostname}",
        f"- **Start:** {iso(start_ts)}",
        f"- **End:** {iso(end_ts)}",
        "",
        "## Run digests (summary.json updated during this run)",
        "",
    ]
    if not digests:
        lines.append("_No summary.json updated during this run — none found._")
    else:
        lines += [
            "| run | scene | built | skipped | yield "
            "| onset | source reached | resumed |",
            "|---|---|---|---|---|---|---|---|",
            # The funnel columns are fractions of T_ANOM_REACHED, which is the
            # denominator §6 names and the one the run's own terminal summary used.
        ]
        for d in digests:
            if d.get("error"):
                lines.append(f"| {d['name']} | — | — | — | — | — | — "
                             f"| ⚠️ {d['error']} |")
                continue
            lines.append(
                f"| {d['name']} | {d.get('scene', '?')} "
                f"| {d.get('built', '?')} | {d.get('skipped', '?')} "
                f"| {d.get('yield', '?')} | {d.get('ONSET_FIRED', '?')} "
                f"| {d.get('SOURCE_REACHED', '?')} "
                f"| {d.get('PRIMARY_RESUMED', '?')} |")
    lines += [
        "",
        "## Terminal output (tail)",
        "",
        "```",
        log_tail,
        "```",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# payload + send
# ----------------------------------------------------------------------

def _gzip_log(log_path: Path, max_gz_bytes: int):
    """Gzip the log; if the result exceeds max_gz_bytes, keep gzipping a
    shrinking tail until it fits. Returns (gz_bytes, truncated)."""
    try:
        data = Path(log_path).read_bytes()
    except OSError:
        data = b"(log unreadable)"
    gz = gzip.compress(data)
    truncated = False
    while len(gz) > max_gz_bytes and len(data) > 1:
        data = data[len(data) // 2:]
        gz = gzip.compress(data)
        truncated = True
    return gz, truncated


def build_payload(*, config, tag, exit_code, duration_s, report_md, log_path,
                  max_gz_bytes=MAX_GZ_BYTES) -> dict:
    marker = "✅" if exit_code == 0 else "❌"
    subject = (f"{marker} [ltm] {tag} — exit {exit_code} "
               f"({humanize_duration(duration_s)})")
    gz, truncated = _gzip_log(Path(log_path), max_gz_bytes)
    if truncated:
        report_md += ("\n> NOTE: log attachment truncated to its tail "
                      "(gzipped log exceeded the Resend size limit).\n")
    to = [a.strip() for a in str(config["NOTIFY_EMAIL_TO"]).split(",")
          if a.strip()]
    return {
        "from": config["NOTIFY_EMAIL_FROM"],
        "to": to,
        "subject": subject,
        "text": report_md,
        "attachments": [
            {"filename": "report.md",
             "content": base64.b64encode(report_md.encode()).decode()},
            {"filename": f"{Path(log_path).stem}.log.gz",
             "content": base64.b64encode(gz).decode()},
        ],
    }


def send_with_retries(payload, api_key, urlopen=urllib.request.urlopen,
                      sleep=time.sleep) -> bool:
    """POST to Resend; 3 attempts with 2s/8s backoff. Never raises."""
    body = json.dumps(payload).encode()
    for attempt in range(len(RETRY_BACKOFF_S) + 1):
        req = urllib.request.Request(
            RESEND_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json",
                     # Cloudflare 403s (error 1010) the default
                     # Python-urllib UA — send a real one.
                     "User-Agent": "ltm-notify-run/1.0"},
        )
        try:
            with urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if 200 <= status < 300:
                    return True
                print(f"[notify] Resend HTTP {status} "
                      f"(attempt {attempt + 1})", file=sys.stderr)
        except urllib.error.HTTPError as e:
            # surface the response body — Resend puts the actual
            # validation error there (e.g. unverified recipient)
            try:
                detail = e.read().decode(errors="replace")[:500]
            except Exception:  # noqa: BLE001
                detail = ""
            print(f"[notify] send attempt {attempt + 1} failed: {e} {detail}",
                  file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — never load-bearing
            print(f"[notify] send attempt {attempt + 1} failed: {e}",
                  file=sys.stderr)
        if attempt < len(RETRY_BACKOFF_S):
            sleep(RETRY_BACKOFF_S[attempt])
    return False


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main(argv=None, environ=None, urlopen=urllib.request.urlopen,
         sleep=time.sleep) -> int:
    if environ is None:
        environ = os.environ
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exit-code", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--start-ts", type=float, required=True)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--tag", default=None)
    # Three levels up from earshot/tools/notify/. Ticket 10 carried this trio
    # verbatim from `scripts/`, where one level up WAS the repo root; the move to
    # three levels deep left this pointing at earshot/tools, so `.env` was read
    # from a directory that has never held one and every run silently declined to
    # email. Same defect as notify-run.sh's two dispatch paths, third instance.
    ap.add_argument("--repo-root",
                    default=str(Path(__file__).resolve().parents[3]))
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root)
    log_path = Path(args.log)
    tag = args.tag or log_path.stem
    end_ts = time.time()

    if environ.get("NOTIFY_DISABLE"):
        print("[notify] NOTIFY_DISABLE set — skipping email.")
        return 0

    config = load_config(repo_root, environ)
    if not config["RESEND_API_KEY"] or not config["NOTIFY_EMAIL_TO"]:
        print("[notify] RESEND_API_KEY / NOTIFY_EMAIL_TO not configured "
              f"(.env at {repo_root}) — skipping email.")
        return 0

    report = build_report(
        command=args.command,
        exit_code=args.exit_code,
        start_ts=args.start_ts,
        end_ts=end_ts,
        commit=args.commit,
        hostname=socket.gethostname(),
        log_tail=tail_lines(log_path, config["NOTIFY_LOG_TAIL_LINES"]),
        digests=discover_run_digests(repo_root / "runs", args.start_ts),
    )
    payload = build_payload(
        config=config, tag=tag, exit_code=args.exit_code,
        duration_s=end_ts - args.start_ts, report_md=report,
        log_path=log_path,
    )
    if send_with_retries(payload, config["RESEND_API_KEY"],
                         urlopen=urlopen, sleep=sleep):
        print(f"[notify] email sent to {', '.join(payload['to'])} "
              f"({payload['subject']})")
    else:
        fallback = log_path.with_name(f"{log_path.stem}-report.md")
        try:
            fallback.write_text(payload["text"])
            print(f"[notify] send FAILED after retries — report saved to "
                  f"{fallback}", file=sys.stderr)
        except OSError as e:
            print(f"[notify] send FAILED and could not save report: {e}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
