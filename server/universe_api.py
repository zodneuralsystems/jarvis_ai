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
import shutil
import subprocess
import time
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
