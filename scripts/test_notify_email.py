"""
Sanity tests for scripts/notify_email.py + scripts/notify-run.sh.

Standalone assert suite (style of embodied_memory/scripts/test_*.py) —
stdlib-only, no habitat/torch/faiss, no network (urlopen is mocked).

Covers:
1. .env parser — comments, blank lines, quotes, KEY=VALUE.
2. Config precedence — real env vars override .env; empty string = unset;
   defaults for NOTIFY_EMAIL_FROM / NOTIFY_LOG_TAIL_LINES.
3. humanize_duration / tail_lines helpers.
4. Run-dir discovery — only summary.json with mtime > start_ts picked;
   malformed JSON skipped with a note, never raises.
5. build_report — header fields, digest table values, fenced log tail.
6. build_payload — subject format, recipients, base64 report.md +
   gzipped log attachments; oversize-log tail truncation note.
7. send_with_retries — mocked urlopen: URL, auth header, JSON body;
   retry/backoff path; all-fail returns False without raising.
8. main() with missing config — warning + exit 0 (never load-bearing).
9. Wrapper smoke — `bash scripts/notify-run.sh false` preserves exit
   code 1, prints the notifier warning, creates the log file.

Invoke with::

    python scripts/test_notify_email.py
"""
from __future__ import annotations

import base64
import gzip
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

spec = importlib.util.spec_from_file_location(
    "notify_email", SCRIPTS_DIR / "notify_email.py"
)
ne = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------

# modeled on runs/abl-s3-qwen/summary.json
SUMMARY_FIXTURE = {
    "n_episodes_attempted": 30,
    "n_episodes_completed": 30,
    "n_successful_episodes": 0,
    "ltm_counts_final": {"fine": 82, "mid": 0, "coarse": 10},
    "n_memory_chosen": 21,
    "ablation": {"backbone": "remembr", "setting": 3},
    "pass_conditions": {
        "fine_layer_nonempty": True,
        "no_crash": True,
        "memory_influences_at_least_once": False,
    },
    "episodes": [
        {"episode_id": "5", "soft_spl": 0.25, "success": False},
        {"episode_id": "6", "soft_spl": 0.75, "success": True},
        {"episode_id": "7", "soft_spl": None, "success": False},  # tolerated
    ],
}


def write_env(path: Path, text: str) -> Path:
    p = path / ".env"
    p.write_text(text)
    return p


# ----------------------------------------------------------------------
# 1. .env parser
# ----------------------------------------------------------------------
print("== .env parser ==")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    envf = write_env(
        td,
        "# a comment\n"
        "\n"
        "RESEND_API_KEY=re_abc123\n"
        "NOTIFY_EMAIL_TO='quoted@example.com'\n"
        'NOTIFY_EMAIL_FROM="dq@example.com"\n'
        "  SPACED_KEY = spaced value  \n"
        "NOT_A_PAIR\n",
    )
    parsed = ne.parse_env_file(envf)
    check("plain value", parsed.get("RESEND_API_KEY") == "re_abc123", parsed)
    check("single quotes stripped", parsed.get("NOTIFY_EMAIL_TO") == "quoted@example.com")
    check("double quotes stripped", parsed.get("NOTIFY_EMAIL_FROM") == "dq@example.com")
    check("whitespace stripped", parsed.get("SPACED_KEY") == "spaced value")
    check("comment ignored", "# a comment" not in parsed)
    check("non-pair line ignored", "NOT_A_PAIR" not in parsed)

    missing = ne.parse_env_file(td / "nope.env")
    check("missing file -> empty dict", missing == {})

# ----------------------------------------------------------------------
# 2. config precedence + defaults
# ----------------------------------------------------------------------
print("== load_config ==")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    write_env(td, "RESEND_API_KEY=re_from_file\nNOTIFY_EMAIL_TO=file@example.com\n")
    cfg = ne.load_config(td, environ={})
    check("from .env", cfg["RESEND_API_KEY"] == "re_from_file")
    check("default FROM", cfg["NOTIFY_EMAIL_FROM"] == "onboarding@resend.dev")
    check("default tail lines", cfg["NOTIFY_LOG_TAIL_LINES"] == 400)

    cfg = ne.load_config(td, environ={"RESEND_API_KEY": "re_from_env"})
    check("env overrides .env", cfg["RESEND_API_KEY"] == "re_from_env")

    cfg = ne.load_config(td, environ={"RESEND_API_KEY": ""})
    check("empty env var = unset (falls back to .env)",
          cfg["RESEND_API_KEY"] == "re_from_file")

    cfg = ne.load_config(td, environ={"NOTIFY_LOG_TAIL_LINES": "100"})
    check("tail lines from env, int", cfg["NOTIFY_LOG_TAIL_LINES"] == 100)

    cfg = ne.load_config(Path(td / "no-such-dir"), environ={})
    check("no .env at all -> key missing/empty", not cfg.get("RESEND_API_KEY"))

# ----------------------------------------------------------------------
# 3. helpers
# ----------------------------------------------------------------------
print("== helpers ==")
check("45s", ne.humanize_duration(45) == "45s")
check("2m 5s", ne.humanize_duration(125) == "2m 5s")
check("1h 2m", ne.humanize_duration(3720) == "1h 2m")
check("3h 0m", ne.humanize_duration(3 * 3600) == "3h 0m")

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.log"
    p.write_text("".join(f"line{i}\n" for i in range(600)))
    t = ne.tail_lines(p, 400)
    lines = t.splitlines()
    check("tail count", len(lines) == 400, len(lines))
    check("tail starts at 200", lines[0] == "line200", lines[0])
    check("tail ends at 599", lines[-1] == "line599")
    check("tail of missing file is placeholder str",
          isinstance(ne.tail_lines(Path(td) / "nope.log", 10), str))

# ----------------------------------------------------------------------
# 4. run-dir discovery
# ----------------------------------------------------------------------
print("== discover_run_digests ==")
with tempfile.TemporaryDirectory() as td:
    runs = Path(td) / "runs"
    start_ts = 1_000_000.0

    old = runs / "abl-old"
    old.mkdir(parents=True)
    (old / "summary.json").write_text(json.dumps(SUMMARY_FIXTURE))
    os.utime(old / "summary.json", (start_ts - 100, start_ts - 100))

    new = runs / "abl-new"
    new.mkdir()
    (new / "summary.json").write_text(json.dumps(SUMMARY_FIXTURE))
    os.utime(new / "summary.json", (start_ts + 100, start_ts + 100))

    bad = runs / "abl-bad"
    bad.mkdir()
    (bad / "summary.json").write_text("{ not json !!!")
    os.utime(bad / "summary.json", (start_ts + 100, start_ts + 100))

    digests = ne.discover_run_digests(runs, start_ts)
    names = [d["name"] for d in digests]
    check("old run excluded", "abl-old" not in names, names)
    check("new run included", "abl-new" in names, names)
    check("malformed included as note", "abl-bad" in names, names)
    good = next(d for d in digests if d["name"] == "abl-new")
    badd = next(d for d in digests if d["name"] == "abl-bad")
    check("setting extracted", good["setting"] == 3, good)
    check("episodes 30/30", good["episodes"] == "30/30")
    check("mean soft_spl over numeric eps", abs(good["mean_soft_spl"] - 0.5) < 1e-9)
    check("n_memory_chosen", good["n_memory_chosen"] == 21)
    check("ltm counts", good["ltm_counts"] == "82/0/10")
    check("gate fails listed",
          good["gate_fails"] == "memory_influences_at_least_once", good)
    check("malformed flagged", badd.get("error"), badd)

    check("missing runs dir -> empty list",
          ne.discover_run_digests(Path(td) / "norunds", start_ts) == [])

# ----------------------------------------------------------------------
# 5. build_report
# ----------------------------------------------------------------------
print("== build_report ==")
digest = {
    "name": "abl-new", "setting": 3, "episodes": "30/30",
    "mean_soft_spl": 0.5, "n_memory_chosen": 21, "ltm_counts": "82/0/10",
    "gate_fails": "memory_influences_at_least_once", "error": None,
}
rep = ne.build_report(
    command="bash scripts/race-revisit.sh --tag wide-1",
    exit_code=0,
    start_ts=1_000_000.0,
    end_ts=1_000_000.0 + 3720,
    commit="2d518a3",
    hostname="race-node-1",
    log_tail="=== paired deltas ===\nGate A GREEN\nDONE",
    digests=[digest],
)
check("success marker", "✅" in rep)
check("command in header", "race-revisit.sh --tag wide-1" in rep)
check("exit code in header", "0" in rep)
check("duration humanized", "1h 2m" in rep)
check("commit in header", "2d518a3" in rep)
check("hostname in header", "race-node-1" in rep)
check("digest table has run dir", "abl-new" in rep)
check("digest table has soft_spl", "0.5" in rep)
check("digest table has ltm counts", "82/0/10" in rep)
check("digest table has gate fails", "memory_influences_at_least_once" in rep)
check("log tail fenced", "```" in rep and "Gate A GREEN" in rep)

rep_fail = ne.build_report(
    command="bash x.sh", exit_code=137, start_ts=0, end_ts=10,
    commit="abc", hostname="h", log_tail="Traceback...", digests=[],
)
check("failure marker", "❌" in rep_fail)
check("exit 137 shown", "137" in rep_fail)
check("no-digests note", "no" in rep_fail.lower() or "none" in rep_fail.lower())

rep_err = ne.build_report(
    command="c", exit_code=0, start_ts=0, end_ts=1, commit="a",
    hostname="h", log_tail="t",
    digests=[{"name": "abl-bad", "error": "malformed summary.json"}],
)
check("malformed digest noted, no raise", "abl-bad" in rep_err)

# ----------------------------------------------------------------------
# 6. build_payload
# ----------------------------------------------------------------------
print("== build_payload ==")
with tempfile.TemporaryDirectory() as td:
    log = Path(td) / "notify-wide-1-20260605.log"
    log.write_text("hello log\n" * 50)
    cfg = {
        "RESEND_API_KEY": "re_test",
        "NOTIFY_EMAIL_TO": "you@example.com",
        "NOTIFY_EMAIL_FROM": "onboarding@resend.dev",
        "NOTIFY_LOG_TAIL_LINES": 400,
    }
    payload = ne.build_payload(
        config=cfg, tag="wide-1", exit_code=0, duration_s=3720,
        report_md="# report body", log_path=log,
    )
    check("subject format",
          payload["subject"] == "✅ [ltm] wide-1 — exit 0 (1h 2m)",
          payload["subject"])
    check("to is list", payload["to"] == ["you@example.com"])
    check("from", payload["from"] == "onboarding@resend.dev")
    check("text is report", payload["text"].startswith("# report body"))
    att = {a["filename"]: a for a in payload["attachments"]}
    check("report.md attached", "report.md" in att)
    check("report.md b64 roundtrip",
          base64.b64decode(att["report.md"]["content"]).decode().startswith("# report"))
    gz_names = [n for n in att if n.endswith(".log.gz")]
    check("log.gz attached", len(gz_names) == 1, list(att))
    if gz_names:
        gz = base64.b64decode(att[gz_names[0]]["content"])
        check("log gz roundtrip", gzip.decompress(gz).decode().startswith("hello log"))

    payload_f = ne.build_payload(
        config=cfg, tag="wide-1", exit_code=1, duration_s=5,
        report_md="r", log_path=log,
    )
    check("failure subject",
          payload_f["subject"] == "❌ [ltm] wide-1 — exit 1 (5s)",
          payload_f["subject"])

    # oversize log -> tail-only gzip + truncation note
    big = Path(td) / "big.log"
    big.write_text("".join(f"unique-line-{i} {os.urandom(8).hex()}\n"
                           for i in range(5000)))
    payload_t = ne.build_payload(
        config=cfg, tag="t", exit_code=0, duration_s=1,
        report_md="r", log_path=big, max_gz_bytes=4096,
    )
    att_t = {a["filename"]: a for a in payload_t["attachments"]}
    gz_t = [n for n in att_t if n.endswith(".log.gz")]
    check("oversize: gz under cap",
          gz_t and len(base64.b64decode(att_t[gz_t[0]]["content"])) <= 4096)
    check("oversize: truncation noted", "truncat" in payload_t["text"].lower())

# ----------------------------------------------------------------------
# 7. send_with_retries (mocked urlopen)
# ----------------------------------------------------------------------
print("== send_with_retries ==")


class FakeResponse(io.BytesIO):
    def __init__(self, body=b'{"id":"x"}'):
        super().__init__(body)
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_urlopen(fail_times=0, captured=None):
    state = {"calls": 0}

    def fake(req, timeout=None):
        state["calls"] += 1
        if captured is not None:
            captured.append(req)
        if state["calls"] <= fail_times:
            raise urllib.error.URLError("boom")
        return FakeResponse()

    fake.state = state
    return fake


payload = {"from": "f@x.com", "to": ["t@x.com"], "subject": "s",
           "text": "b", "attachments": []}

captured = []
slept = []
ok = ne.send_with_retries(payload, "re_test",
                          urlopen=make_urlopen(0, captured),
                          sleep=slept.append)
check("send ok", ok is True)
check("one call only", len(captured) == 1)
req = captured[0]
check("resend url", req.get_full_url() == "https://api.resend.com/emails",
      req.get_full_url())
check("bearer auth", req.get_header("Authorization") == "Bearer re_test",
      dict(req.headers))
check("content-type json",
      (req.get_header("Content-type") or "").startswith("application/json"))
body = json.loads(req.data.decode())
check("body is payload", body["subject"] == "s" and body["to"] == ["t@x.com"])
check("no sleep on success", slept == [])

slept = []
fake = make_urlopen(2)
ok = ne.send_with_retries(payload, "k", urlopen=fake, sleep=slept.append)
check("retries then succeeds", ok is True and fake.state["calls"] == 3,
      fake.state)
check("backoff 2s then 8s", slept == [2, 8], slept)

slept = []
fake = make_urlopen(99)
ok = ne.send_with_retries(payload, "k", urlopen=fake, sleep=slept.append)
check("all fail -> False, no raise", ok is False and fake.state["calls"] == 3,
      fake.state)

# ----------------------------------------------------------------------
# 8. main() missing config -> warning + exit 0
# ----------------------------------------------------------------------
print("== main missing config ==")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "runs").mkdir()
    log = td / "runs" / "notify-t-x.log"
    log.write_text("some output\n")
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = ne.main([
            "--exit-code", "1", "--log", str(log),
            "--command", "bash scripts/race-smoke.sh",
            "--start-ts", str(int(time.time()) - 60),
            "--commit", "abc1234", "--tag", "t",
            "--repo-root", str(td),
        ], environ={})
    out = buf.getvalue()
    check("exit 0 when unconfigured", rc == 0, rc)
    check("warning printed", "RESEND_API_KEY" in out or "not configured" in out.lower(), out)

# ----------------------------------------------------------------------
# 8b. main() send-failure -> report.md written next to log, exit 0
# ----------------------------------------------------------------------
print("== main send failure fallback ==")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "runs").mkdir()
    log = td / "runs" / "notify-t2-x.log"
    log.write_text("boom\n")
    write_env(td, "RESEND_API_KEY=re_x\nNOTIFY_EMAIL_TO=a@b.c\n")
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = ne.main([
            "--exit-code", "0", "--log", str(log),
            "--command", "echo hi",
            "--start-ts", str(int(time.time()) - 5),
            "--commit", "abc", "--tag", "t2",
            "--repo-root", str(td),
        ], environ={}, urlopen=make_urlopen(99), sleep=lambda s: None)
    check("send-fail still exit 0", rc == 0, rc)
    reports = list((td / "runs").glob("*report*.md"))
    check("local report.md written on send failure", len(reports) == 1,
          list((td / "runs").iterdir()))

# ----------------------------------------------------------------------
# 9. wrapper smoke: exit code preserved, warning, log created
# ----------------------------------------------------------------------
print("== notify-run.sh smoke ==")
with tempfile.TemporaryDirectory() as td:
    env = dict(os.environ)
    env["NOTIFY_RUN_LOG_DIR"] = td
    # force the no-send path even if a real .env exists locally
    env["NOTIFY_DISABLE"] = "1"
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "notify-run.sh"), "false"],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )
    check("wrapped exit code preserved", proc.returncode == 1, proc.returncode)
    logs = list(Path(td).glob("notify-*.log"))
    check("log file created", len(logs) == 1, logs)
    combined = proc.stdout + proc.stderr
    check("notifier warning printed",
          "not configured" in combined.lower()
          or "disable" in combined.lower()
          or "RESEND_API_KEY" in combined,
          combined[-500:])

    proc0 = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "notify-run.sh"), "echo", "ok-marker"],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )
    check("success exit code 0", proc0.returncode == 0, proc0.returncode)
    check("wrapped stdout passes through", "ok-marker" in proc0.stdout,
          proc0.stdout[-300:])

# ----------------------------------------------------------------------
print()
print(f"{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
