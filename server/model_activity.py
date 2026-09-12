"""Cross-process ownership markers for Zod's local owner model.

Runtime V2 may need to reclaim unified memory before a heavy MLX job. It must
never unload an Ollama model while Hermes is using it, but it should be able to
release Zod's own model once a real owner run has ended. HUD is the authority
that knows that lifecycle, so it records active runs and completed use here.

The files are deliberately advisory and fail-safe: if recording fails, Runtime
V2 simply keeps the model instead of risking an active run.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

ACTIVE_PATH = Path.home() / ".zod" / "model-run-activity.json"
COMPLETED_PATH = Path.home() / ".zod" / "model-use-hud.json"
HERMES_CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"
_LOCK = threading.Lock()


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _clean_active(value: dict) -> dict:
    return {
        str(run_id): row
        for run_id, row in value.items()
        if isinstance(row, dict) and _pid_alive(row.get("pid", -1))
    }


def resolve_local_model(hud_cfg: dict, hermes_config_path: Path = HERMES_CONFIG_PATH) -> str:
    """Resolve HUD's stable Hermes alias to a local Ollama model, if any."""
    try:
        alias = str(((hud_cfg.get("hermes") or {}).get("model") or "")).strip()
        if alias.startswith("ollama-local/"):
            return alias
        if not alias or not hermes_config_path.is_file():
            return ""
        import yaml
        cfg = yaml.safe_load(hermes_config_path.read_text(encoding="utf-8")) or {}
        api_server = ((cfg.get("gateway") or {}).get("api_server") or {})
        routes = (api_server.get("extra") or {}).get("model_routes") or {}
        route = routes.get(alias) if isinstance(routes, dict) else None
        model = str((route or {}).get("model") or "").strip()
        return model if model.startswith("ollama-local/") else ""
    except Exception:
        return ""


def mark_started(run_id: str, model: str) -> bool:
    if not run_id or not model.startswith("ollama-local/"):
        return False
    try:
        with _LOCK:
            active = _clean_active(_read(ACTIVE_PATH))
            active[run_id] = {
                "model": model,
                "pid": os.getpid(),
                "since": time.time(),
                "owner": "zod-hud-hermes",
            }
            _write(ACTIVE_PATH, active)
        return True
    except Exception:
        return False


def mark_finished(run_id: str, model: str) -> bool:
    """Clear the busy claim and record that the last Zod use completed."""
    if not run_id:
        return False
    try:
        with _LOCK:
            active = _clean_active(_read(ACTIVE_PATH))
            active.pop(run_id, None)
            _write(ACTIVE_PATH, active)
            if model.startswith("ollama-local/"):
                completed = _read(COMPLETED_PATH)
                completed[model.split("/")[-1]] = time.time()
                cutoff = time.time() - 3600
                completed = {
                    str(k): float(v) for k, v in completed.items()
                    if isinstance(v, (int, float)) and float(v) > cutoff
                }
                _write(COMPLETED_PATH, completed)
        return True
    except Exception:
        return False
