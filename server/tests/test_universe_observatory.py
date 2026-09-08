import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

import universe_api
import universe_state


class UniverseObservatoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.path_patch = patch.multiple(
            universe_state,
            STATE_DIR=root,
            PROJECTS_PATH=root / "projects.json",
            JOBS_PATH=root / "jobs.json",
            EVIDENCE_PATH=root / "evidence.jsonl",
        )
        self.path_patch.start()

    def tearDown(self):
        self.path_patch.stop()
        self.tmp.cleanup()

    def test_job_observability_persists_without_private_reasoning(self):
        saved = universe_state.upsert_job({
            "id": "job-web",
            "state": "RUNNING",
            "mission": "Repair one bounded gate",
            "executor": "codex-web-harness",
            "model": "chatgpt-web/high",
            "agents": [{
                "id": "builder",
                "role": "implementation",
                "assignment_reason": "smallest healthy free engineering route",
                "dependencies": ["reviewer"],
                "chain_of_thought": "must never persist",
                "reasoning": "must never persist either",
            }],
        })
        self.assertEqual(saved["agents"][0]["assignment_reason"], "smallest healthy free engineering route")
        self.assertNotIn("chain_of_thought", saved["agents"][0])
        self.assertNotIn("reasoning", saved["agents"][0])

    def test_regression_raw_and_private_reasoning_stripped_evidence_survives(self):
        for field in ("chain_of_thought", "chain-of-thought", "cot",
                      "reasoning", "raw_reasoning", "hidden_reasoning",
                      "thinking"):
            self.assertIn(field, universe_state.OBSERVATORY_PRIVATE_REASONING_FIELDS)
        payload = {
            "id": "job-evidence",
            "state": "COMPLETED",
            "mission": "gate verification",
            "evidence": ["acceptable finding", "verifiable result"],
            "agents": [{
                "id": "auditor",
                "role": "verification",
                "raw_reasoning": "must be stripped",
                "thinking": "also stripped",
            }],
        }
        saved = universe_state.upsert_job(payload)
        self.assertEqual(saved["mission"], "gate verification")
        self.assertEqual(saved["evidence"], ["acceptable finding", "verifiable result"])
        self.assertIn("id", saved["agents"][0])
        self.assertIn("role", saved["agents"][0])
        self.assertNotIn("raw_reasoning", saved["agents"][0])
        self.assertNotIn("thinking", saved["agents"][0])
        rendered = json.dumps(saved)
        for field in universe_state.OBSERVATORY_PRIVATE_REASONING_FIELDS:
            self.assertNotIn(field, rendered)

    def test_observatory_uses_exact_codex_web_identity_and_allowlisted_facts(self):
        job = universe_api._observatory_job({
            "id": "job-web",
            "state": "RUNNING",
            "objective": "Continue same job",
            "executor": "codex-web-harness",
            "model": "chatgpt-web/high",
            "agents": [{"id": "builder", "state": "RUNNING", "executor": "codex-chatgpt-web",
                        "model": "chatgpt web high", "raw_reasoning": "private"}],
            "raw_reasoning": "private",
        }, [])
        self.assertEqual(job["executor"], "Codex Web Harness")
        self.assertEqual(job["model"], "ChatGPT Web — High")
        self.assertEqual(job["agents"][0]["executor"], "Codex Web Harness")
        self.assertEqual(job["agents"][0]["model"], "ChatGPT Web — High")
        self.assertNotIn("raw_reasoning", json.dumps(job))

    def test_observatory_endpoint_reports_structured_reasoning_policy(self):
        universe_state.upsert_job({"id": "job-one", "state": "RUNNING", "objective": "Observe"})
        with patch.object(universe_api, "worker_registry", return_value=[]), \
             patch.object(universe_api, "capability_registry", return_value=[]), \
             patch.object(universe_api, "agent_registry", return_value={"agents": []}), \
             patch.object(universe_api, "_continuation_jobs", return_value=([], "UNAVAILABLE")):
            response = asyncio.run(universe_api.observatory())
        body = json.loads(response.body)
        self.assertIn("job-one", {job["job_id"] for job in body["jobs"]})
        self.assertEqual(body["continuation"]["authority"], "HUD fallback/history")
        self.assertIn("raw chain-of-thought is never exposed", body["reasoning_policy"])

    def test_continuation_live_view_is_primary_and_redacted(self):
        universe_state.upsert_job({
            "id": "same-job",
            "state": "RUNNING",
            "objective": "stale HUD copy",
        })
        live = universe_api._continuation_observatory_job({
            "job_id": "same-job",
            "state": "RUNNING",
            "objective": "authoritative continuation mission",
            "workdir": "/tmp/repo",
            "current_worker": "codex-web-harness",
            "current_provider": "chatgpt-web-subscription",
            "current_model": "chatgpt-web/high",
            "current_route": "codex-web-harness",
            "current_role": "implementation",
            "current_slice": 2,
            "current_slice_id": "slice-2",
            "slice_count": 2,
            "acceptance": {"criteria_total": 4, "criteria_passed": 3},
            "receipts": {"total": 1, "confirmed": 1},
            "budgets": {"max_failures": 2},
            "spend": {"failures": 0, "provider_failovers": 1},
            "remaining_budget": {"max_failures": 2},
            "worker_history": [{
                "slice_id": "slice-1",
                "client_kind": "local-devstral",
                "route": "local-devstral",
                "provider": "ollama",
                "model": "devstral",
                "end_reason": "handoff",
                "at": "2026-09-08T12:00:00+10:00",
            }],
            "raw_reasoning": "private",
        })
        with patch.object(universe_api, "worker_registry", return_value=[]), \
             patch.object(universe_api, "capability_registry", return_value=[]), \
             patch.object(universe_api, "agent_registry", return_value={"agents": []}), \
             patch.object(universe_api, "_continuation_jobs", return_value=([live], "LIVE")):
            response = asyncio.run(universe_api.observatory())
        body = json.loads(response.body)
        same = [job for job in body["jobs"] if job["job_id"] == "same-job"]
        self.assertEqual(len(same), 1)
        self.assertEqual(same[0]["mission"], "authoritative continuation mission")
        self.assertEqual(same[0]["executor"], "Codex Web Harness")
        self.assertEqual(same[0]["model"], "ChatGPT Web — High")
        self.assertEqual(len(same[0]["agents"]), 2)
        self.assertEqual(same[0]["agents"][1]["dependencies"], ["slice-1"])
        self.assertEqual(body["continuation"]["authority"], "Zod-Continuation")
        self.assertNotIn("raw_reasoning", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
