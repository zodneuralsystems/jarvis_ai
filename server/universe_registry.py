"""Zod's Universe — capability, worker and agent registries.

Design rule from the Phase 3 directive: this file must never claim capability it
does not have. Every entry carries an explicit, honest state:

  worker state    RESIDENT | INSTALLED | NOT_INSTALLED | UNKNOWN
  agent state     ACTIVE | AVAILABLE | INTERFACE_ONLY | PLANNED
  routing         a capability is either routed to a real worker or UNROUTED

"Installed" is not "active" and "interface exists" is not "implemented". The
Universe UI renders these states verbatim rather than smoothing them over,
because runtime testing has already caught a local model fabricating machine
state when guidance was disabled.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import requests

OLLAMA_BASE = "http://127.0.0.1:11434"

# --------------------------------------------------------------- worker states
RESIDENT = "RESIDENT"
INSTALLED = "INSTALLED"
NOT_INSTALLED = "NOT_INSTALLED"
UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------- agent states
ACTIVE = "ACTIVE"                  # proven working on this machine right now
AVAILABLE = "AVAILABLE"            # a real worker exists but the role is not running
INTERFACE_ONLY = "INTERFACE_ONLY"  # registry/UI contract exists, no autonomy yet
PLANNED = "PLANNED"                # named only; nothing implemented

# One large local model at a time — measured on this 36 GB Mac Studio: with a
# 24 GB model already resident a cold load took 264.5s, versus 9.5s once the
# other model was unloaded. The Universe surfaces this; it does not swap yet.
ONE_LARGE_MODEL_RULE = "ONE_LARGE_LOCAL_MODEL_AT_A_TIME"
LARGE_MODEL_GB = 12.0


def _ollama(path: str, timeout: float = 4.0) -> dict:
    try:
        r = requests.get(f"{OLLAMA_BASE}{path}", timeout=timeout)
        if r.status_code != 200:
            return {}
        return r.json() or {}
    except Exception:
        return {}


def ollama_state() -> dict:
    """Real Ollama state: what is installed and what is actually resident."""
    tags = _ollama("/api/tags")
    ps = _ollama("/api/ps")
    reachable = bool(tags) or bool(ps)
    installed = {}
    for m in (tags.get("models") or []):
        name = m.get("name") or ""
        if name:
            installed[name] = {
                "name": name,
                "size_gb": round(float(m.get("size") or 0) / 1e9, 1),
                "state": INSTALLED,
            }
    resident = []
    for m in (ps.get("models") or []):
        name = m.get("name") or m.get("model") or ""
        entry = {
            "name": name,
            "size_gb": round(float(m.get("size") or 0) / 1e9, 1),
            "vram_gb": round(float(m.get("size_vram") or 0) / 1e9, 1),
            "context_length": m.get("context_length"),
            "expires_at": m.get("expires_at"),
            "state": RESIDENT,
        }
        resident.append(entry)
        if name in installed:
            installed[name]["state"] = RESIDENT
    return {
        "reachable": reachable,
        "installed": sorted(installed.values(), key=lambda x: x["name"]),
        "resident": resident,
        "rule": ONE_LARGE_MODEL_RULE,
        "large_model_gb_threshold": LARGE_MODEL_GB,
    }


# ------------------------------------------------------------ worker registry
# Registered truthfully per directive section 13. `probe` is the ollama tag we
# check for real installation; None means the worker is not an ollama model.
WORKERS: list[dict[str, Any]] = [
    {
        "id": "zod-qwen3-coder-30b",
        "name": "zod-qwen3-coder:30b",
        "probe": "zod-qwen3-coder:30b",
        "kind": "local_llm",
        "role": "Primary Zod brain — reasoning, orchestration, local tool decisions",
        "large": True,
    },
    {
        "id": "opencode",
        "name": "OpenCode",
        "probe": None,
        "kind": "engineering_environment",
        "role": "Engineering execution environment for bounded implementation jobs",
        "large": False,
    },
    {
        "id": "zod-devstral-opencode",
        "name": "zod-devstral-opencode",
        "probe": "zod-devstral-opencode:latest",
        "kind": "local_llm",
        "role": "Primary bounded local implementation worker inside OpenCode",
        "large": True,
    },
    {
        "id": "qwen3-coder-30b",
        "name": "qwen3-coder:30b",
        "probe": "qwen3-coder:30b",
        "kind": "local_llm",
        "role": "Independent read-only reviewer when not acting as writer",
        "large": True,
    },
    {
        "id": "qwen2.5-coder-14b",
        "name": "qwen2.5-coder:14b",
        "probe": "qwen2.5-coder:14b",
        "kind": "local_llm",
        "role": "Lighter coding / mechanical transformation worker",
        "large": False,
    },
    {
        "id": "qwen3-vl-4b",
        "name": "qwen3-vl:4b",
        "probe": "qwen3-vl:4b",
        "kind": "local_vlm",
        "role": "Local vision capability",
        "large": False,
    },
    {
        "id": "stt-small-en",
        "name": "faster-whisper small.en",
        "probe": None,
        "kind": "local_stt",
        "role": "Local speech to text",
        "large": False,
    },
    {
        "id": "tts-macos",
        "name": "macOS local TTS",
        "probe": None,
        "kind": "local_tts",
        "role": "Local Zod voice (male, configurable)",
        "large": False,
    },
]


def worker_registry(oll: dict | None = None, cfg: dict | None = None) -> list[dict]:
    """Workers with real, probed state. Never asserts an unverified worker."""
    oll = oll if oll is not None else ollama_state()
    cfg = cfg or {}
    by_name = {m["name"]: m for m in oll.get("installed", [])}
    resident_names = {m["name"] for m in oll.get("resident", [])}
    out = []
    for w in WORKERS:
        entry = dict(w)
        probe = w.get("probe")
        if probe:
            if probe in resident_names:
                entry["state"] = RESIDENT
            elif probe in by_name:
                entry["state"] = INSTALLED
            elif not oll.get("reachable"):
                entry["state"] = UNKNOWN
                entry["note"] = "Ollama not reachable"
            else:
                entry["state"] = NOT_INSTALLED
            entry["size_gb"] = by_name.get(probe, {}).get("size_gb")
        elif w["id"] == "opencode":
            entry["state"] = INSTALLED if shutil.which("opencode") else NOT_INSTALLED
            entry["note"] = "Suspended while Zod's brain is resident (one-large-model rule)"
        elif w["id"] == "tts-macos":
            entry["state"] = INSTALLED if shutil.which("say") else NOT_INSTALLED
            entry["voice"] = str((cfg.get("voice") or {}).get("macos_voice") or "system-default")
        elif w["id"] == "stt-small-en":
            entry["state"] = INSTALLED
            entry["note"] = str((cfg.get("stt") or {}).get("model") or "")
        else:
            entry["state"] = UNKNOWN
        out.append(entry)
    return out


# -------------------------------------------------------- capability registry
# Directive section 16. `worker` is the worker id a capability routes to today;
# None means UNROUTED — the contract exists, nothing serves it yet.
CAPABILITIES: list[dict[str, Any]] = [
    {"id": "general_reasoning", "worker": "zod-qwen3-coder-30b", "state": ACTIVE,
     "note": "Serves live typed and spoken Zod turns"},
    {"id": "computer_read", "worker": "zod-qwen3-coder-30b", "state": ACTIVE,
     "note": "Read-only inspection, auto-allowed, verified 4/4"},
    {"id": "computer_write", "worker": "zod-qwen3-coder-30b", "state": ACTIVE,
     "note": "Strict approval required before execution, verified 7/7"},
    {"id": "text_to_speech", "worker": "tts-macos", "state": ACTIVE,
     "note": "Local macOS voice, no cloud"},
    {"id": "speech_to_text", "worker": "stt-small-en", "state": ACTIVE,
     "note": "Local faster-whisper"},
    {"id": "code_review", "worker": "qwen3-coder-30b", "state": AVAILABLE,
     "note": "Used read-only for the Phase 2 independent review"},
    {"id": "code_implementation", "worker": "opencode", "state": AVAILABLE,
     "note": "OpenCode + devstral installed; not driven by Zod yet"},
    {"id": "small_code_task", "worker": "qwen2.5-coder-14b", "state": AVAILABLE,
     "note": "Model installed; no routing implemented"},
    {"id": "vision", "worker": "qwen3-vl-4b", "state": AVAILABLE,
     "note": "Model installed; no routing implemented"},
    {"id": "research", "worker": None, "state": PLANNED},
    {"id": "document_analysis", "worker": None, "state": PLANNED},
    {"id": "testing", "worker": None, "state": PLANNED},
    {"id": "deployment", "worker": None, "state": PLANNED},
    {"id": "security_review", "worker": None, "state": PLANNED},
    {"id": "business_analysis", "worker": None, "state": PLANNED},
    {"id": "marketing", "worker": None, "state": PLANNED},
    {"id": "sales", "worker": None, "state": PLANNED},
]


def capability_registry(workers: list[dict] | None = None) -> list[dict]:
    """Capabilities with the real state of the worker each one routes to."""
    workers = workers if workers is not None else worker_registry()
    wmap = {w["id"]: w for w in workers}
    out = []
    for c in CAPABILITIES:
        entry = dict(c)
        wid = c.get("worker")
        if not wid:
            entry["routing"] = "UNROUTED"
            entry["worker_name"] = None
            entry["worker_state"] = None
        else:
            w = wmap.get(wid) or {}
            entry["routing"] = "ROUTED"
            entry["worker_name"] = w.get("name")
            entry["worker_state"] = w.get("state", UNKNOWN)
        out.append(entry)
    return out


# ------------------------------------------------------------- agent registry
# Directive section 11. Every role is listed so the organisation is visible, but
# state is truthful: only what is genuinely proven is ACTIVE.
DEPARTMENTS = [
    "Executive", "Strategy", "Research", "Product", "Engineering", "QA",
    "Security", "Design", "Marketing", "Sales", "Finance", "Operations",
]

AGENTS: list[dict[str, Any]] = [
    {"id": "zod_orchestrator", "name": "Zod Orchestrator", "dept": "Executive",
     "state": ACTIVE, "capability": "general_reasoning",
     "note": "Live: typed + spoken turns, tool use, strict approval"},
    {"id": "approval_safety_governor", "name": "Approval & Safety Governor", "dept": "Security",
     "state": ACTIVE, "capability": "computer_write",
     "note": "Strict approval proven: deny=0 exec, approve-once=exactly 1, "
             "duplicate/wrong/stale rejected, reconnect fail-closed, stop-before-spawn"},
    {"id": "capability_router", "name": "Capability Router", "dept": "Executive",
     "state": INTERFACE_ONLY, "capability": None,
     "note": "Registry + strict/non-strict routing contract exist; no model selection yet"},
    {"id": "resource_governor", "name": "Resource Governor", "dept": "Operations",
     "state": INTERFACE_ONLY, "capability": None,
     "note": "Surfaces model residency and the one-large-model rule; no automatic swap"},
    {"id": "job_manager", "name": "Job Manager", "dept": "Operations",
     "state": INTERFACE_ONLY, "capability": None,
     "note": "Persistent job store + API exist; no autonomous scheduler"},
    {"id": "evidence_qa_controller", "name": "Evidence / QA Controller", "dept": "QA",
     "state": INTERFACE_ONLY, "capability": None,
     "note": "Evidence is recorded against jobs; no autonomous verification"},
    {"id": "independent_code_reviewer", "name": "Independent Code Reviewer", "dept": "QA",
     "state": AVAILABLE, "capability": "code_review",
     "note": "qwen3-coder read-only; ran the Phase 2 safety review"},
    {"id": "implementation_engineer", "name": "Implementation Engineer", "dept": "Engineering",
     "state": AVAILABLE, "capability": "code_implementation",
     "note": "OpenCode + devstral installed; currently suspended, not driven by Zod"},
    {"id": "fast_coding_worker", "name": "Fast Coding Worker", "dept": "Engineering",
     "state": AVAILABLE, "capability": "small_code_task",
     "note": "qwen2.5-coder:14b installed; no routing implemented"},
    {"id": "vision_agent", "name": "Vision Agent", "dept": "Research",
     "state": AVAILABLE, "capability": "vision",
     "note": "qwen3-vl installed; no routing implemented"},
    {"id": "memory_context_manager", "name": "Memory & Context Manager", "dept": "Executive",
     "state": PLANNED, "capability": None},
    {"id": "planner_architect", "name": "Planner / Architect", "dept": "Strategy",
     "state": PLANNED, "capability": None},
    {"id": "senior_software_architect", "name": "Senior Software Architect", "dept": "Engineering",
     "state": PLANNED, "capability": None},
    {"id": "test_engineer", "name": "Test Engineer", "dept": "QA", "state": PLANNED, "capability": None},
    {"id": "debug_incident_engineer", "name": "Debug / Incident Engineer", "dept": "Engineering",
     "state": PLANNED, "capability": None},
    {"id": "devops_engineer", "name": "DevOps Engineer", "dept": "Operations",
     "state": PLANNED, "capability": None},
    {"id": "git_repository_steward", "name": "Git / Repository Steward", "dept": "Engineering",
     "state": PLANNED, "capability": None},
    {"id": "security_engineer", "name": "Security Engineer", "dept": "Security",
     "state": PLANNED, "capability": None},
    {"id": "performance_finops_engineer", "name": "Performance / FinOps Engineer", "dept": "Finance",
     "state": PLANNED, "capability": None},
    {"id": "deep_research_agent", "name": "Deep Research Agent", "dept": "Research",
     "state": PLANNED, "capability": "research"},
    {"id": "opportunity_miner", "name": "Opportunity Miner", "dept": "Strategy",
     "state": PLANNED, "capability": None},
    {"id": "data_analyst", "name": "Data Analyst", "dept": "Research",
     "state": PLANNED, "capability": None},
    {"id": "quant_simulation_agent", "name": "Quant / Simulation Agent", "dept": "Research",
     "state": PLANNED, "capability": None},
    {"id": "document_intelligence_agent", "name": "Document Intelligence Agent", "dept": "Research",
     "state": PLANNED, "capability": "document_analysis"},
    {"id": "product_manager", "name": "Product Manager", "dept": "Product",
     "state": PLANNED, "capability": None},
    {"id": "commercial_analyst", "name": "Commercial Analyst", "dept": "Finance",
     "state": PLANNED, "capability": "business_analysis"},
    {"id": "operations_agent", "name": "Operations Agent", "dept": "Operations",
     "state": PLANNED, "capability": None},
    {"id": "creative_brand_agent", "name": "Creative / Brand Agent", "dept": "Design",
     "state": PLANNED, "capability": None},
    {"id": "marketing_agent", "name": "Marketing Agent", "dept": "Marketing",
     "state": PLANNED, "capability": "marketing"},
    {"id": "sales_agent", "name": "Sales Agent", "dept": "Sales",
     "state": PLANNED, "capability": "sales"},
]


def agent_registry(capabilities: list[dict] | None = None) -> dict:
    """Agents grouped by department, each with its honest state."""
    capabilities = capabilities if capabilities is not None else capability_registry()
    cmap = {c["id"]: c for c in capabilities}
    agents = []
    for a in AGENTS:
        entry = dict(a)
        cap = cmap.get(a.get("capability") or "")
        entry["worker_name"] = cap.get("worker_name") if cap else None
        entry["worker_state"] = cap.get("worker_state") if cap else None
        agents.append(entry)
    counts: dict[str, int] = {}
    for a in agents:
        counts[a["state"]] = counts.get(a["state"], 0) + 1
    return {
        "departments": DEPARTMENTS,
        "agents": agents,
        "counts": counts,
        "total": len(agents),
        "disclosure": "Roles are logical. A role is only ACTIVE if it is proven "
                      "running on this machine; AVAILABLE means a real worker "
                      "exists but nothing drives the role yet.",
    }


def git_snapshot(repo: str) -> dict:
    """Real git state for a repository path, or an explicit unknown."""
    def run(*args: str) -> str | None:
        try:
            p = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                               text=True, timeout=8)
            return p.stdout.strip() if p.returncode == 0 else None
        except Exception:
            return None

    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    if branch is None:
        return {"available": False, "state": UNKNOWN}
    porcelain = run("status", "--porcelain") or ""
    return {
        "available": True,
        "branch": branch,
        "commit": (run("rev-parse", "--short", "HEAD") or ""),
        "subject": (run("log", "-1", "--pretty=%s") or ""),
        "dirty_tracked": len([l for l in porcelain.splitlines() if l and not l.startswith("??")]),
        "untracked": len([l for l in porcelain.splitlines() if l.startswith("??")]),
    }
