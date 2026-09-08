"""Zod's Universe — API router mounted at /api/universe/*.

Additive by design: this module owns every Universe endpoint so the proven
operator surface in server.py is left alone. It inherits the existing
``/api/`` auth middleware, so all of these require the HUD token.

Honesty contract: every endpoint returns either real measured data or an
explicit state string (NOT_CONNECTED / UNKNOWN / PLANNED). No endpoint
synthesises progress, revenue, agent activity or test results.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from universe_registry import (
    LARGE_MODEL_GB,
    ONE_LARGE_MODEL_RULE,
    agent_registry,
    capability_registry,
    git_snapshot,
    ollama_state,
    worker_registry,
)
import universe_state

router = APIRouter(prefix="/api/universe")

_CFG: dict = {}
_HERMES_BASE = "http://127.0.0.1:8642"

CODEX_WEB_EXECUTOR = "Codex Web Harness"
CODEX_WEB_MODEL = "ChatGPT Web — High"

_CONTINUATION_ROOT = Path(
    os.environ.get("ZOD_CONTINUATION_ROOT", str(Path.home() / "Zod-Continuation"))
).expanduser()
_CONTINUATION_BIN = _CONTINUATION_ROOT / "bin"


def _continuation_api():
    """Load the authoritative Continuation read API when it is present locally.

    The HUD consumes Continuation's existing public live-view contract instead
    of reinterpreting raw job state. Import failure is a read-only degradation:
    the existing HUD store remains available as history/fallback.
    """
    package = _CONTINUATION_BIN / "zod_continuation" / "control_plane_api.py"
    if not package.is_file():
        return None
    bin_path = str(_CONTINUATION_BIN)
    if bin_path not in sys.path:
        sys.path.insert(0, bin_path)
    try:
        return importlib.import_module("zod_continuation.control_plane_api")
    except Exception:
        return None


def _continuation_evidence(view: dict) -> list[str]:
    """Expose bounded machine facts only; never forward free-form worker prose."""
    facts: list[str] = []
    acceptance = view.get("acceptance") or {}
    total = acceptance.get("criteria_total")
    passed = acceptance.get("criteria_passed")
    if total is not None:
        facts.append(f"machine acceptance: {passed if passed is not None else '?'} / {total}")
    receipts = view.get("receipts") or {}
    if receipts.get("total") is not None:
        facts.append(
            "side-effect receipts: "
            f"{receipts.get('confirmed', 0)} confirmed / {receipts.get('total', 0)} total"
        )
    if view.get("state_dir"):
        facts.append(f"durable state directory: {view['state_dir']}")
    return facts[:8]


def _continuation_observatory_job(view: dict) -> dict:
    """Translate Continuation's authoritative live view into HUD observability."""
    worker = view.get("current_worker")
    route = view.get("current_route")
    is_codex_web = worker == "codex-web-harness" or route == "codex-web-harness"
    executor = CODEX_WEB_EXECUTOR if is_codex_web else worker
    model = CODEX_WEB_MODEL if is_codex_web else view.get("current_model")
    role = view.get("current_role") or "implementation"
    assignment_reason = (
        f"Continuation selected route {route}" if route else "Continuation route decision"
    )
    state = view.get("state") or "UNKNOWN"
    history_agents: list[dict] = []
    previous_id: str | None = None
    for idx, item in enumerate((view.get("worker_history") or [])[-12:]):
        if not isinstance(item, dict):
            continue
        kind = item.get("client_kind") or item.get("route")
        hist_is_web = kind == "codex-web-harness" or item.get("route") == "codex-web-harness"
        hist_id = str(item.get("slice_id") or f"{view.get('job_id') or 'job'}-history-{idx + 1}")
        hist_route = item.get("route")
        history_agents.append({
            "id": hist_id,
            "name": kind or "Continuation worker",
            "role": "implementation",
            "state": "HISTORICAL",
            "assignment_reason": (
                f"Continuation selected route {hist_route}" if hist_route else "Continuation route decision"
            ),
            "executor": CODEX_WEB_EXECUTOR if hist_is_web else kind,
            "provider": item.get("provider"),
            "model": CODEX_WEB_MODEL if hist_is_web else item.get("model"),
            "dependencies": [previous_id] if previous_id else [],
            "parallel_group": None,
            "heartbeat": item.get("at"),
            "context": {"slice_id": item.get("slice_id")},
            "budget": {},
            "quota": {},
            "health": item.get("end_reason") or "ENDED",
            "evidence": [],
            "blocker": None,
            "retries": 0,
            "handoff": {"end_reason": item.get("end_reason")},
            "failover": None,
            "supervisor": None,
            "machine_acceptance": {},
            "stop": {"state": "ENDED"},
            "scarce_tier": {"used": [], "avoided": []},
        })
        previous_id = hist_id

    agent = {
        "id": f"{view.get('job_id') or 'job'}-active",
        "name": worker or route or "Continuation worker",
        "role": role,
        "state": state,
        "assignment_reason": assignment_reason,
        "executor": executor,
        "provider": view.get("current_provider"),
        "model": model,
        "dependencies": [previous_id] if previous_id else [],
        "parallel_group": None,
        "heartbeat": view.get("last_meaningful_progress") or view.get("updated_at"),
        "context": {"slice": view.get("current_slice"), "slice_id": view.get("current_slice_id")},
        "budget": view.get("budgets") or {},
        "quota": {"remaining_budget": view.get("remaining_budget") or {}},
        "health": "ACTIVE" if state in {"RUNNING", "CONTINUING"} else state,
        "evidence": _continuation_evidence(view),
        "blocker": view.get("blocker"),
        "retries": (view.get("spend") or {}).get("failures", 0),
        "handoff": {"count": view.get("handoffs", 0)},
        "failover": {"count": (view.get("spend") or {}).get("provider_failovers", 0)},
        "supervisor": None,
        "machine_acceptance": view.get("acceptance") or {},
        "stop": {
            "state": "STOPPED" if state == "STOPPED" else "AVAILABLE_VIA_CONTINUATION",
            "reason": view.get("stop_reason"),
        },
        "scarce_tier": {"used": [], "avoided": []},
    }
    return {
        "id": view.get("job_id") or "unknown-job",
        "mission": view.get("objective") or "mission not recorded",
        "state": state,
        "role": role,
        "assignment_reason": assignment_reason,
        "executor": executor,
        "provider": view.get("current_provider"),
        "model": model,
        "dependencies": [],
        "parallel_group": None,
        "heartbeat": view.get("last_meaningful_progress") or view.get("updated_at"),
        "context": {
            "workdir": view.get("workdir"),
            "current_step": view.get("current_step"),
            "current_slice": view.get("current_slice"),
            "slice_count": view.get("slice_count"),
        },
        "budget": view.get("budgets") or {},
        "quota": {"remaining_budget": view.get("remaining_budget") or {}},
        "health": "ACTIVE" if state in {"RUNNING", "CONTINUING"} else state,
        "evidence": _continuation_evidence(view),
        "blocker": view.get("blocker"),
        "retries": (view.get("spend") or {}).get("failures", 0),
        "handoffs": [{"count": view.get("handoffs", 0)}] if view.get("handoffs") else [],
        "failover": {"count": (view.get("spend") or {}).get("provider_failovers", 0)},
        "supervisor_interventions": [],
        "machine_acceptance": view.get("acceptance") or {},
        "stop": agent["stop"],
        "scarce_tier": {"used": [], "avoided": []},
        "agents": history_agents + ([agent] if worker or route else []),
        "source": "Zod-Continuation authoritative live view",
    }


def _continuation_jobs() -> tuple[list[dict], str]:
    api = _continuation_api()
    if api is None:
        return [], "UNAVAILABLE"
    try:
        rows = api.list_jobs(limit=100)
    except Exception:
        return [], "DEGRADED"
    return [_continuation_observatory_job(row) for row in rows if isinstance(row, dict)], "LIVE"


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _display_identity(value: Any, kind: str) -> Any:
    """Keep the subscription-backed Web harness distinct from native Codex/Astra."""
    if not isinstance(value, str):
        return value
    folded = value.strip().lower().replace("_", "-")
    if kind == "executor" and folded in {
        "codex-web-harness", "codex web harness", "codex-chatgpt-web",
    }:
        return CODEX_WEB_EXECUTOR
    if kind == "model" and folded in {
        "chatgpt-web/high", "chatgpt web high", "chatgpt-web-high",
        "chatgpt web — high", "chatgpt web - high",
    }:
        return CODEX_WEB_MODEL
    return value


def _observatory_agent(agent: dict) -> dict:
    """Allow-list structured execution facts; never forward free-form reasoning."""
    worker_name = agent.get("worker_name")
    return {
        "id": agent.get("id") or agent.get("name") or "agent",
        "name": agent.get("name") or agent.get("id") or "Agent",
        "role": agent.get("role") or agent.get("capability") or "unassigned",
        "state": agent.get("state") or "UNKNOWN",
        "assignment_reason": agent.get("assignment_reason") or agent.get("note") or "not recorded",
        "executor": _display_identity(agent.get("executor") or ("Zod local worker" if worker_name else None), "executor"),
        "provider": agent.get("provider"),
        "model": _display_identity(agent.get("model") or worker_name, "model"),
        "dependencies": _as_list(agent.get("dependencies")),
        "parallel_group": agent.get("parallel_group"),
        "heartbeat": agent.get("heartbeat"),
        "context": agent.get("context") or {},
        "budget": agent.get("budget") or {},
        "quota": agent.get("quota") or {},
        "health": agent.get("health") or agent.get("worker_state") or "UNKNOWN",
        "evidence": _as_list(agent.get("evidence")),
        "blocker": agent.get("blocker"),
        "retries": agent.get("retries") or 0,
        "handoff": agent.get("handoff"),
        "failover": agent.get("failover"),
        "supervisor": agent.get("supervisor"),
        "machine_acceptance": agent.get("machine_acceptance"),
        "stop": agent.get("stop"),
        "scarce_tier": agent.get("scarce_tier"),
    }


def _observatory_job(job: dict, recorded_evidence: list[dict]) -> dict:
    job_id = str(job.get("id") or "unknown-job")
    agents = [_observatory_agent(a) for a in (job.get("agents") or []) if isinstance(a, dict)]
    evidence = [e for e in recorded_evidence if e.get("job") == job_id][:30]
    return {
        "job_id": job_id,
        "mission": job.get("mission") or job.get("objective") or "mission not recorded",
        "state": job.get("state") or "UNKNOWN",
        "role": job.get("role"),
        "assignment_reason": job.get("assignment_reason") or "not recorded",
        "executor": _display_identity(job.get("executor") or job.get("worker"), "executor"),
        "provider": job.get("provider"),
        "model": _display_identity(job.get("model"), "model"),
        "dependencies": _as_list(job.get("dependencies")),
        "parallel_group": job.get("parallel_group"),
        "heartbeat": job.get("heartbeat") or job.get("updated_at"),
        "context": job.get("context") or {},
        "budget": job.get("budget") or {},
        "quota": job.get("quota") or {},
        "health": job.get("health") or "UNKNOWN",
        "evidence": _as_list(job.get("evidence")),
        "recorded_evidence": evidence,
        "blocker": job.get("blocker"),
        "retries": job.get("retries") or 0,
        "handoffs": _as_list(job.get("handoffs")),
        "failover": job.get("failover"),
        "supervisor_interventions": _as_list(job.get("supervisor_interventions")),
        "machine_acceptance": job.get("machine_acceptance"),
        "stop": job.get("stop") or {"state": "AVAILABLE_VIA_HUD"},
        "scarce_tier": job.get("scarce_tier") or {"used": [], "avoided": []},
        "stages": _as_list(job.get("stages")),
        "completed_stages": _as_list(job.get("completed_stages")),
        "agents": agents,
        "source": job.get("source") or "unknown",
    }


def configure(cfg: dict) -> None:
    """Called once from server.py so this module never re-reads config itself."""
    global _CFG, _HERMES_BASE
    _CFG = cfg or {}
    _HERMES_BASE = str((_CFG.get("hermes") or {}).get("base_url") or _HERMES_BASE)


# ------------------------------------------------------------------- overview
def _memory() -> dict:
    vm = psutil.virtual_memory()
    total_gb = round(vm.total / 1e9, 1)
    return {
        "total_gb": total_gb,
        "available_gb": round(vm.available / 1e9, 1),
        "used_percent": vm.percent,
        # The measured constraint on this machine, surfaced rather than assumed.
        "rule": ONE_LARGE_MODEL_RULE,
        "rule_note": f"Only one model above ~{LARGE_MODEL_GB:.0f}GB fits resident "
                     f"in {total_gb:.0f}GB of unified memory. Measured cold load "
                     f"was 264.5s under contention versus 9.5s uncontended.",
    }


def _service_blocking(name: str, url: str) -> dict:
    """Probe one external service. Must run off the event loop (see _services)."""
    started = time.perf_counter()
    try:
        r = requests.get(url, timeout=4)
        ms = int((time.perf_counter() - started) * 1000)
        if r.status_code != 200:
            return {"name": name, "state": "DEGRADED", "detail": f"HTTP {r.status_code}", "latency_ms": ms}
        body: Any
        try:
            body = r.json()
        except ValueError:
            body = {}
        return {"name": name, "state": "ONLINE", "latency_ms": ms,
                "detail": body.get("version") or body.get("status") or "ok"}
    except Exception:
        return {"name": name, "state": "OFFLINE", "detail": "unreachable", "latency_ms": None}


async def _services(oll: dict) -> list[dict]:
    """Service health.

    The HUD is never probed over HTTP: this code runs *inside* the HUD process,
    so a self-request cannot be served while the handler is waiting on it and
    would always report OFFLINE. If this function is executing, the HUD is up.
    External probes run in a worker thread so they never block the event loop
    that also carries Zod's WebSocket audio.
    """
    hermes = await asyncio.to_thread(_service_blocking, "Hermes", f"{_HERMES_BASE}/health")
    return [
        {"name": "Zod HUD", "state": "ONLINE", "detail": "serving this request", "latency_ms": None},
        hermes,
        {"name": "Ollama",
         "state": "ONLINE" if oll.get("reachable") else "OFFLINE",
         "detail": f"{len(oll.get('installed', []))} models installed",
         "latency_ms": None},
    ]


@router.get("/overview")
async def overview() -> JSONResponse:
    """Executive home data. Every field is measured or explicitly stated."""
    oll = await asyncio.to_thread(ollama_state)
    workers = worker_registry(oll, _CFG)
    caps = capability_registry(workers)
    agents = agent_registry(caps)
    projects = universe_state.projects()
    jobs = universe_state.jobs()

    hermes_cfg = _CFG.get("hermes") or {}
    voice_cfg = _CFG.get("voice") or {}
    services = await _services(oll)

    active_jobs = [j for j in jobs if j.get("state") in ("RUNNING", "AWAITING_APPROVAL")]
    blocked = [j for j in jobs if j.get("state") == "BLOCKED" or j.get("blocker")]
    project_blockers = [
        {"project": p.get("id"), "name": p.get("name"), "blocker": b}
        for p in projects for b in (p.get("blockers") or [])
    ]

    return JSONResponse({
        "brief": {
            # An honest brief: counts of real records, not a generated narrative.
            "projects_total": len(projects),
            "jobs_total": len(jobs),
            "jobs_active": len(active_jobs),
            "jobs_blocked": len(blocked),
            "agents_total": agents["total"],
            "agents_active": agents["counts"].get("ACTIVE", 0),
            "agents_available": agents["counts"].get("AVAILABLE", 0),
            "agents_planned": agents["counts"].get("PLANNED", 0),
            "agents_interface_only": agents["counts"].get("INTERFACE_ONLY", 0),
            "generated_at": time.time(),
        },
        "decisions_required": {
            # Approvals are live over the WebSocket; there is no server-side
            # queue to poll, so this is stated rather than faked.
            "state": "LIVE_OVER_WEBSOCKET",
            "note": "Approval requests arrive on the Zod channel in real time and "
                    "render in the Approvals view. Nothing is queued server-side.",
        },
        "opportunities": {
            "state": "NOT_YET_CONNECTED",
            "note": "Opportunity Miner is PLANNED. No opportunity data exists yet.",
        },
        "revenue": {
            "state": "NOT_YET_CONNECTED",
            "note": "No revenue source is connected. Nothing is estimated.",
        },
        "overnight": {
            "state": "NOT_YET_CONNECTED",
            "note": "Change summarisation is PLANNED; no overnight digest is produced yet.",
        },
        "jobs_active": active_jobs,
        "blockers": project_blockers,
        "services": services,
        "memory": _memory(),
        "brain": {
            "model": hermes_cfg.get("model"),
            "provider": hermes_cfg.get("provider"),
            "fallback": hermes_cfg.get("fallback_provider") or "None",
            "resident": [m["name"] for m in oll.get("resident", [])],
            "voice": voice_cfg.get("macos_voice") or "system-default",
            "stt": (_CFG.get("stt") or {}).get("model"),
        },
    })


# ------------------------------------------------------------ projects / jobs
def _projects_with_git() -> list[dict]:
    items = universe_state.projects()
    for p in items:
        p["repo_state"] = {repo: git_snapshot(repo) for repo in (p.get("repos") or [])}
    return items


@router.get("/projects")
async def get_projects() -> JSONResponse:
    # git + file reads run off the event loop: this handler shares the loop with
    # Zod's WebSocket audio, which must never wait on a subprocess.
    items = await asyncio.to_thread(_projects_with_git)
    return JSONResponse({"projects": items, "stages": universe_state.STAGES})


@router.post("/projects")
async def post_project(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    try:
        return JSONResponse(universe_state.upsert_project(body))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@router.get("/jobs")
async def get_jobs() -> JSONResponse:
    return JSONResponse({"jobs": universe_state.jobs(), "states": universe_state.JOB_STATES})


@router.get("/observatory")
async def observatory() -> JSONResponse:
    """Live mission/agent observability from durable job facts and registries."""
    hud_jobs = universe_state.jobs()
    continuation_jobs, continuation_state = await asyncio.to_thread(_continuation_jobs)
    live_ids = {j.get("id") for j in continuation_jobs}
    jobs = continuation_jobs + [j for j in hud_jobs if j.get("id") not in live_ids]
    evidence = universe_state.evidence(240)
    workers = await asyncio.to_thread(worker_registry, None, _CFG)
    registry = agent_registry(capability_registry(workers))
    return JSONResponse({
        "jobs": [_observatory_job(j, evidence) for j in jobs],
        "registry_agents": [_observatory_agent(a) for a in registry.get("agents", [])],
        "identity": {
            "codex_web_executor": CODEX_WEB_EXECUTOR,
            "codex_web_model": CODEX_WEB_MODEL,
        },
        "continuation": {
            "state": continuation_state,
            "authority": "Zod-Continuation" if continuation_state == "LIVE" else "HUD fallback/history",
            "jobs": len(continuation_jobs),
        },
        "reasoning_policy": "Structured reasons and evidence only; raw chain-of-thought is never exposed.",
        "generated_at": time.time(),
    })


@router.post("/jobs")
async def post_job(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    try:
        return JSONResponse(universe_state.upsert_job(body))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


# ---------------------------------------------------- organisation / registry
@router.get("/organisation")
async def organisation() -> JSONResponse:
    caps = await asyncio.to_thread(capability_registry)
    return JSONResponse(agent_registry(caps))


@router.get("/capabilities")
async def capabilities() -> JSONResponse:
    workers = await asyncio.to_thread(worker_registry, None, _CFG)
    return JSONResponse({
        "capabilities": capability_registry(workers),
        "contract": "Zod requests a capability, never a model name. UNROUTED means "
                    "the contract exists but no worker serves it yet.",
    })


@router.get("/resources")
async def resources() -> JSONResponse:
    """Real machine, model and worker state — the Resource Governor view."""
    oll = await asyncio.to_thread(ollama_state)
    workers = await asyncio.to_thread(worker_registry, oll, _CFG)
    resident_large = [m for m in oll.get("resident", []) if (m.get("size_gb") or 0) >= LARGE_MODEL_GB]
    return JSONResponse({
        "memory": _memory(),
        "ollama": oll,
        "workers": workers,
        "resident_large_count": len(resident_large),
        "rule_satisfied": len(resident_large) <= 1,
        "swap_automation": {
            "state": "INTERFACE_ONLY",
            "note": "Model residency is reported here. Automatic checkpoint / "
                    "unload / pre-warm swapping is not implemented yet, so no "
                    "swap is performed on your behalf.",
        },
        "opencode": {
            "installed": bool(shutil.which("opencode")),
            "running": await asyncio.to_thread(_opencode_running),
        },
    })


def _opencode_running() -> list[dict]:
    """Real OpenCode processes, if any. Empty list means none are running."""
    out = []
    try:
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            cmd = " ".join(p.info.get("cmdline") or [])
            if "opencode" in cmd and "grep" not in cmd:
                model = None
                for token in cmd.split():
                    if "devstral" in token or "qwen" in token:
                        model = token
                out.append({"pid": p.info["pid"], "model": model, "cmd": cmd[:140]})
    except Exception:
        return []
    return out


@router.get("/evidence")
async def get_evidence() -> JSONResponse:
    return JSONResponse({
        "evidence": universe_state.evidence(120),
        "note": "Recorded actions and results only — never model reasoning.",
    })


@router.post("/evidence")
async def post_evidence(request: Request) -> JSONResponse:
    """Append one action/result record.

    Append-only on purpose: the evidence trail is the record of what actually
    happened, so it is never edited or deleted through the API.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not str(body.get("action") or "").strip():
        return JSONResponse({"error": "action is required"}, status_code=400)
    return JSONResponse(universe_state.append_evidence(body))


@router.get("/memory")
async def memory_view() -> JSONResponse:
    return JSONResponse({
        "state": "INTERFACE_ONLY",
        "note": "Project and job records are the only persistent Zod context "
                "today. A dedicated memory/knowledge store is PLANNED. Secrets "
                "are never surfaced here.",
        "projects": len(universe_state.projects()),
        "jobs": len(universe_state.jobs()),
    })


# ---------------------------------------------------------------------- voice
@router.get("/voice")
async def voice_state() -> JSONResponse:
    """Installed local voices actually usable by `say`, plus the current choice.

    Only voices `say -v '?'` reports are listed: naming an absent voice makes
    `say` fall back to the system default while still exiting 0, which would
    silently give Zod the wrong voice.
    """
    voices: list[dict] = []
    try:
        p = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10)
        for line in p.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            idx = next((i for i, t in enumerate(parts) if "_" in t and len(t) == 5), None)
            if idx is None:
                continue
            voices.append({"name": " ".join(parts[:idx]), "locale": parts[idx]})
    except Exception:
        return JSONResponse({"state": "UNKNOWN", "voices": [], "current": None})
    cfg_voice = str((_CFG.get("voice") or {}).get("macos_voice") or "")
    return JSONResponse({
        "state": "ONLINE",
        "current": cfg_voice or "system-default",
        "rate": str((_CFG.get("voice") or {}).get("macos_rate") or ""),
        "provider": str((_CFG.get("voice") or {}).get("provider") or "macos"),
        "voices": voices,
        "measured": {
            # Real measurements from this machine, kept so the choice is auditable.
            "Ralph": 70.4, "Rocko (English (US))": 82.6, "Grandpa": 95.9,
            "Fred": 100.7, "Reed (English (US))": 110.8, "Daniel": 112.5,
            "Rishi": 114.2, "Eddy (English (US))": 124.2, "Aman": 134.9,
        },
        "measured_note": "Median fundamental frequency in Hz on one identical "
                         "sentence; lower is deeper. Daniel was selected as a "
                         "full-quality long-form male voice.",
    })


@router.post("/voice/preview")
async def voice_preview(request: Request) -> JSONResponse:
    """Speak a short sample locally through `say` so a voice can be auditioned.

    Audio-only and local: it writes nothing and calls no network service.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    name = str(body.get("voice") or "").strip()
    if not name or len(name) > 60 or any(c in name for c in ";|&$`\n"):
        return JSONResponse({"error": "invalid voice name"}, status_code=400)
    text = "Good evening, Nick. Zod is online. All systems nominal."
    say_bin = shutil.which("say") or "/usr/bin/say"
    try:
        subprocess.run([say_bin, "-v", name, text], check=True, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as exc:
        return JSONResponse({"error": f"preview failed: {type(exc).__name__}"}, status_code=500)
    return JSONResponse({"spoken": True, "voice": name})
