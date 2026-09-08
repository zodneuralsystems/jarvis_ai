"""Zod's Universe — persistent project and job store.

A job must survive model unloading, worker changes, service restarts, agent
handoff and UI refresh (directive section 15), so state lives on disk rather
than in process memory. JSON under ``server/universe_state/`` keeps it
inspectable by hand; writes are atomic (temp file + replace) under a lock so a
crash cannot leave a half-written ledger.

Nothing in here invents data. The store starts with exactly the records that are
independently verifiable on this machine, each carrying a ``source`` field so
its provenance is visible in the UI.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

STATE_DIR = Path(__file__).resolve().parent / "universe_state"
PROJECTS_PATH = STATE_DIR / "projects.json"
JOBS_PATH = STATE_DIR / "jobs.json"
EVIDENCE_PATH = STATE_DIR / "evidence.jsonl"

_LOCK = threading.RLock()

# Project lifecycle from directive section 6.
STAGES = ["IDEA", "RESEARCH", "VALIDATE", "BUILD", "TEST", "LAUNCH", "GROW",
          "OPERATE", "OPTIMISE"]

JOB_STATES = ["PLANNED", "RUNNING", "BLOCKED", "AWAITING_APPROVAL", "COMPLETED",
              "FAILED", "CANCELLED"]

# Optional mission-observability fields accepted by the existing job seam. They
# are deliberately structured facts only. Private model reasoning is never
# persisted by this store or returned to the HUD observatory.
OBSERVABILITY_FIELDS = (
    "mission", "role", "assignment_reason", "executor", "provider", "model",
    "dependencies", "parallel_group", "heartbeat", "context", "budget", "quota",
    "health", "retries", "handoffs", "failover", "supervisor_interventions",
    "machine_acceptance", "stop", "scarce_tier", "agents",
)
_PRIVATE_REASONING_KEYS = {
    "chain_of_thought", "chain-of-thought", "cot", "reasoning", "raw_reasoning",
    "hidden_reasoning", "thinking",
}


def _now() -> float:
    return time.time()


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_observability(value: Any, depth: int = 0) -> Any:
    """Bound and redact observability payloads before they become durable state."""
    if depth > 5:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1200]
    if isinstance(value, list):
        return [_safe_observability(v, depth + 1) for v in value[:80]]
    if isinstance(value, dict):
        out = {}
        for raw_key, raw_value in list(value.items())[:80]:
            key = str(raw_key)[:120]
            if key.strip().lower() in _PRIVATE_REASONING_KEYS:
                continue
            out[key] = _safe_observability(raw_value, depth + 1)
        return out
    return str(value)[:1200]


# --------------------------------------------------------------------- seeding
# These two records are real and independently checkable: the commit hashes
# exist in the two repositories and the gate results were produced by the
# harnesses under server/scripts and the Phase 3 scratch harnesses. They are
# seeded so the Universe opens with true state instead of an empty shell.
def _seed_projects() -> list[dict]:
    return [
        {
            "id": "zod-operator",
            "name": "Zod Computer Operator",
            "objective": "A HUD-driven local operator that can converse, speak, "
                         "inspect this machine read-only, and execute "
                         "state-changing actions only behind strict approval.",
            "stage": "OPERATE",
            "progress_note": "Physically accepted: typed conversation, second "
                             "turn, audible local voice, telemetry, strict "
                             "approval, read-only inspection, HUD lifecycle.",
            "owner": "Zod Orchestrator",
            "repos": ["/Users/zod/Zod-HUD", "/Users/zod/.hermes/hermes-agent"],
            "blockers": [],
            "next_action": "Physical acceptance of the Universe interface.",
            "evidence": [
                "Zod-HUD checkpoint 7bbe11eccf402a6dc19d86694b2266362f29114e",
                "Hermes checkpoint 5e373e1ad0070e90666b306723dc38e2e579d2f2",
                "Strict approval 7/7; HUD lifecycle 9/9; read-only 4/4; Hermes suites 42/42",
            ],
            "source": "recorded during the Phase 2 operator milestone",
            "created_at": _now(),
            "updated_at": _now(),
        },
        {
            "id": "zod-universe",
            "name": "Zod's Universe",
            "objective": "An executive command centre over the proven operator "
                         "core: one persistent Zod, real project/job/agent "
                         "state, and honest capability reporting.",
            "stage": "BUILD",
            "progress_note": "First functional shell under construction. "
                             "Registries are truthful; autonomy is not claimed.",
            "owner": "Zod Orchestrator",
            "repos": ["/Users/zod/Zod-HUD"],
            "blockers": [],
            "next_action": "Nick's physical acceptance of /universe/ and the male voice.",
            "evidence": [],
            "source": "recorded during the Phase 3 Universe build",
            "created_at": _now(),
            "updated_at": _now(),
        },
    ]


def _seed_jobs() -> list[dict]:
    return [
        {
            "id": "job-operator-milestone",
            "project": "zod-operator",
            "objective": "Repair typed chat and prove strict approval safety end to end.",
            "state": "COMPLETED",
            "worker": "Claude Code (sole writer)",
            "stages": ["diagnose typed-chat regression", "wire hybrid strict path",
                       "close the ungated-write gap", "prove approval safety",
                       "prove HUD lifecycle", "regression", "checkpoint"],
            "completed_stages": ["diagnose typed-chat regression",
                                 "wire hybrid strict path",
                                 "close the ungated-write gap",
                                 "prove approval safety", "prove HUD lifecycle",
                                 "regression", "checkpoint"],
            "checkpoint": "7bbe11eccf402a6dc19d86694b2266362f29114e",
            "evidence": [
                "Root cause: HUD process held a pre-rotation token; 101,884 WS 403 rejections",
                "Gate D 7/7 on the real production path",
                "Gate E 9/9 visible lifecycle",
                "Hermes suites 42/42",
            ],
            "blocker": None,
            "next_action": None,
            "source": "recorded by Claude Code during the Phase 2 milestone",
            "created_at": _now(),
            "updated_at": _now(),
        },
        {
            "id": "job-universe-phase3",
            "project": "zod-universe",
            "objective": "Build the first usable Zod's Universe shell over the proven core.",
            "state": "RUNNING",
            "worker": "Claude Code (sole writer)",
            "stages": ["freeze operator recovery state", "inspect prior UI read-only",
                       "create /universe/ shell", "executive home + persistent Zod",
                       "male local voice", "native projects/jobs/org/approvals",
                       "model + resource view", "registries", "regression",
                       "physical acceptance", "checkpoint"],
            "completed_stages": ["freeze operator recovery state",
                                 "inspect prior UI read-only"],
            "checkpoint": None,
            "evidence": [
                "Operator recovery verified: both checkpoints reachable, 0 dirty tracked files",
                "Voice candidates measured by median F0 on identical text",
            ],
            "blocker": None,
            "next_action": "Nick's physical acceptance test.",
            "source": "recorded by Claude Code during the Phase 3 build",
            "created_at": _now(),
            "updated_at": _now(),
        },
    ]


def ensure_seeded() -> None:
    with _LOCK:
        if not PROJECTS_PATH.exists():
            _atomic_write(PROJECTS_PATH, {"projects": _seed_projects()})
        if not JOBS_PATH.exists():
            _atomic_write(JOBS_PATH, {"jobs": _seed_jobs()})


# ---------------------------------------------------------------------- reads
def projects() -> list[dict]:
    ensure_seeded()
    return list(_read(PROJECTS_PATH, {"projects": []}).get("projects") or [])


def jobs() -> list[dict]:
    ensure_seeded()
    return list(_read(JOBS_PATH, {"jobs": []}).get("jobs") or [])


def evidence(limit: int = 100) -> list[dict]:
    if not EVIDENCE_PATH.exists():
        return []
    out = []
    try:
        for line in EVIDENCE_PATH.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return list(reversed(out))


# --------------------------------------------------------------------- writes
def append_evidence(entry: dict) -> dict:
    """Append one real action/result record. Callers supply verified facts only."""
    record = {
        "id": f"ev-{uuid.uuid4().hex[:10]}",
        "at": _now(),
        "action": str(entry.get("action") or "")[:400],
        "worker": str(entry.get("worker") or "")[:120],
        "tool": str(entry.get("tool") or "")[:80],
        "result": str(entry.get("result") or "")[:800],
        "job": entry.get("job"),
        "project": entry.get("project"),
    }
    with _LOCK:
        EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVIDENCE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    return record


def upsert_job(payload: dict) -> dict:
    """Create or update one job. Unknown states are rejected rather than coerced."""
    state = str(payload.get("state") or "PLANNED").upper()
    if state not in JOB_STATES:
        raise ValueError(f"unsupported job state: {state}")
    with _LOCK:
        current = jobs()
        job_id = str(payload.get("id") or f"job-{uuid.uuid4().hex[:10]}")
        existing = next((j for j in current if j.get("id") == job_id), None)
        record = dict(existing or {
            "id": job_id,
            "source": "created through the Universe API",
            "created_at": _now(),
        })
        for key in ("project", "objective", "worker", "checkpoint", "blocker",
                    "next_action"):
            if key in payload:
                record[key] = payload[key]
        for key in ("stages", "completed_stages", "evidence"):
            if key in payload and isinstance(payload[key], list):
                record[key] = [str(x)[:400] for x in payload[key]]
        for key in OBSERVABILITY_FIELDS:
            if key in payload:
                record[key] = _safe_observability(payload[key])
        record["state"] = state
        record["updated_at"] = _now()
        record.setdefault("stages", [])
        record.setdefault("completed_stages", [])
        record.setdefault("evidence", [])
        others = [j for j in current if j.get("id") != job_id]
        _atomic_write(JOBS_PATH, {"jobs": others + [record]})
        return record


def upsert_project(payload: dict) -> dict:
    stage = str(payload.get("stage") or "IDEA").upper()
    if stage not in STAGES:
        raise ValueError(f"unsupported project stage: {stage}")
    with _LOCK:
        current = projects()
        pid = str(payload.get("id") or f"proj-{uuid.uuid4().hex[:8]}")
        existing = next((p for p in current if p.get("id") == pid), None)
        record = dict(existing or {
            "id": pid,
            "source": "created through the Universe API",
            "created_at": _now(),
        })
        for key in ("name", "objective", "progress_note", "owner", "next_action"):
            if key in payload:
                record[key] = payload[key]
        for key in ("repos", "blockers", "evidence"):
            if key in payload and isinstance(payload[key], list):
                record[key] = [str(x)[:400] for x in payload[key]]
        record["stage"] = stage
        record["updated_at"] = _now()
        record.setdefault("repos", [])
        record.setdefault("blockers", [])
        record.setdefault("evidence", [])
        others = [p for p in current if p.get("id") != pid]
        _atomic_write(PROJECTS_PATH, {"projects": others + [record]})
        return record
