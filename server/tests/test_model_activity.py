import json
from pathlib import Path

import model_activity as M


def _paths(monkeypatch, tmp_path):
    active = tmp_path / "active.json"
    completed = tmp_path / "completed.json"
    monkeypatch.setattr(M, "ACTIVE_PATH", active)
    monkeypatch.setattr(M, "COMPLETED_PATH", completed)
    return active, completed


def test_resolve_owner_alias_to_local_model(tmp_path):
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text("""gateway:\n  api_server:\n    extra:\n      model_routes:\n        zod-owner-routed:\n          model: ollama-local/zod-devstral-opencode:latest\n""")
    assert M.resolve_local_model({"hermes": {"model": "zod-owner-routed"}}, cfg) == \
        "ollama-local/zod-devstral-opencode:latest"


def test_nonlocal_route_is_not_marked_as_reclaimable(tmp_path):
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text("""gateway:\n  api_server:\n    extra:\n      model_routes:\n        zod-owner-routed:\n          model: openrouter/some-cloud-model\n""")
    assert M.resolve_local_model({"hermes": {"model": "zod-owner-routed"}}, cfg) == ""


def test_run_lifecycle_records_busy_then_completed(monkeypatch, tmp_path):
    active, completed = _paths(monkeypatch, tmp_path)
    model = "ollama-local/zod-devstral-opencode:latest"
    assert M.mark_started("run-1", model)
    row = json.loads(active.read_text())["run-1"]
    assert row["model"] == model
    assert row["pid"] > 0

    assert M.mark_finished("run-1", model)
    assert "run-1" not in json.loads(active.read_text())
    assert json.loads(completed.read_text())["zod-devstral-opencode:latest"] > 0


def test_dead_owner_run_is_cleaned_before_new_write(monkeypatch, tmp_path):
    active, _ = _paths(monkeypatch, tmp_path)
    active.write_text(json.dumps({"dead": {"model": "ollama-local/x", "pid": 99999999}}))
    assert M.mark_started("live", "ollama-local/y")
    data = json.loads(active.read_text())
    assert "dead" not in data
    assert "live" in data
