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

def _digest_one(run_dir: Path, summary_path: Path) -> dict:
    d = {"name": run_dir.name, "error": None}
    try:
        s = json.loads(summary_path.read_text())
    except (OSError, ValueError) as e:
        d["error"] = f"malformed summary.json ({e.__class__.__name__})"
        return d
    if not isinstance(s, dict):
        d["error"] = "malformed summary.json (not an object)"
        return d
    abl = s.get("ablation") or {}
    d["setting"] = abl.get("setting", "?")
    d["episodes"] = (f"{s.get('n_episodes_completed', '?')}/"
                     f"{s.get('n_episodes_attempted', '?')}")
    sspls = [ep.get("soft_spl") for ep in s.get("episodes", [])
             if isinstance(ep, dict)
             and isinstance(ep.get("soft_spl"), (int, float))]
    d["mean_soft_spl"] = (sum(sspls) / len(sspls)) if sspls else None
    d["n_memory_chosen"] = s.get("n_memory_chosen", "?")
    ltm = s.get("ltm_counts_final") or {}
    d["ltm_counts"] = (f"{ltm.get('fine', '?')}/{ltm.get('mid', '?')}/"
                       f"{ltm.get('coarse', '?')}")
    fails = [k for k, v in (s.get("pass_conditions") or {}).items() if v is False]
    d["gate_fails"] = ", ".join(fails) if fails else "—"
    return d


def discover_run_digests(runs_dir: Path, start_ts: float) -> list:
    """Digest every runs/*/summary.json modified after start_ts.
    Tolerates malformed files (digest with an 'error' note); never raises."""
    runs_dir = Path(runs_dir)
    digests = []
    if not runs_dir.is_dir():
        return digests
    for run_dir in sorted(runs_dir.iterdir()):
        summary = run_dir / "summary.json"
        try:
            if not summary.is_file() or summary.stat().st_mtime <= start_ts:
                continue
        except OSError:
            continue
        digests.append(_digest_one(run_dir, summary))
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
        "## Run digests (runs/*/summary.json updated during this run)",
        "",
    ]
    if not digests:
        lines.append("_No summary.json updated during this run — none found._")
    else:
        lines += [
            "| run | setting | eps | mean soft_spl | mem_chosen "
            "| ltm f/m/c | gate fails |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in digests:
            if d.get("error"):
                lines.append(f"| {d['name']} | — | — | — | — | — "
                             f"| ⚠️ {d['error']} |")
                continue
            mss = d.get("mean_soft_spl")
            mss = f"{mss:.4g}" if isinstance(mss, (int, float)) else "?"
            lines.append(
                f"| {d['name']} | {d.get('setting', '?')} "
                f"| {d.get('episodes', '?')} | {mss} "
                f"| {d.get('n_memory_chosen', '?')} "
                f"| {d.get('ltm_counts', '?')} | {d.get('gate_fails', '—')} |")
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
                     "Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if 200 <= status < 300:
                    return True
                print(f"[notify] Resend HTTP {status} "
                      f"(attempt {attempt + 1})", file=sys.stderr)
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
    ap.add_argument("--repo-root",
                    default=str(Path(__file__).resolve().parent.parent))
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
