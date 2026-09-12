"""Regression coverage for the Hermes named-session lookup.

The original defect: get_session_id() read only the first page of
/api/sessions. Hermes defaults that page to 50 and CLAMPS `limit` to 200
server-side, so a named conversation sitting past the first page was never
found. Zod then tried to recreate the title, Hermes answered
400 invalid_title, and the raise_for_status() fell through as a bare
"400 Bad Request for url: .../api/sessions". Typed chat and voice both died,
because both resolve their session through this one function.

These tests simulate a Hermes with far more sessions than one page and prove
the lookup still finds a session on the last page.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

MODEL = "zod-qwen3-coder:30b"
PAGE_CAP = 200  # Hermes clamps `limit` to this regardless of what we ask for


class FakeHermes:
    """Minimal stand-in for the Hermes sessions API, with real clamping."""

    def __init__(self, total: int, target_title: str, target_index: int,
                 target_model: str = MODEL):
        self.sessions = [
            {"id": f"api_{i}", "title": f"other-{i}", "model": MODEL}
            for i in range(total)
        ]
        self.sessions[target_index] = {
            "id": "api_target", "title": target_title, "model": target_model,
        }
        self.requests = []

    def get(self, url, headers=None, params=None, timeout=None):
        params = params or {}
        limit = min(int(params.get("limit", 50)), PAGE_CAP)
        offset = int(params.get("offset", 0))
        self.requests.append((limit, offset))
        window = self.sessions[offset:offset + limit]
        body = {
            "data": window,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(window)) < len(self.sessions),
        }
        return mock.Mock(ok=True, status_code=200, json=lambda: body)


def make_api(fake: FakeHermes):
    """Real HermesAPI wired to the fake. `base` is a read-only property derived
    from cfg, so configure it through cfg rather than assigning to it."""
    import server

    api = server.HermesAPI({"hermes": {
        "base_url": "http://127.0.0.1:8642",
        "model": MODEL,
    }})
    api.headers = lambda: {}   # no real credential needed for a stubbed transport
    return api


class SessionPaginationTests(unittest.TestCase):
    def _lookup(self, total, target_index, title="jarvis-main", model=MODEL):
        fake = FakeHermes(total, title, target_index, model)
        api = make_api(fake)
        with mock.patch("server.requests.get", fake.get):
            found = api.find_session_by_title(title)
            taken = False
        return found, taken, fake

    def test_finds_session_on_first_page(self):
        found, _, _ = self._lookup(total=40, target_index=10)
        self.assertEqual(found, "api_target")

    def test_finds_session_past_the_default_50(self):
        """The original bug: anything past the first page was invisible."""
        found, _, _ = self._lookup(total=120, target_index=119)
        self.assertEqual(found, "api_target")

    def test_finds_session_past_the_200_server_cap(self):
        """limit=500 did NOT help: Hermes clamps the page to 200."""
        found, _, fake = self._lookup(total=450, target_index=449)
        self.assertEqual(found, "api_target")
        self.assertTrue(max(l for l, _ in fake.requests) <= PAGE_CAP)

    def test_finds_session_past_500(self):
        """Directive requirement: prove lookup works beyond 500 sessions."""
        found, _, fake = self._lookup(total=1300, target_index=1299)
        self.assertEqual(found, "api_target")
        self.assertGreaterEqual(len(fake.requests), 7)

    def test_pages_are_requested_with_advancing_offsets(self):
        _, _, fake = self._lookup(total=650, target_index=649)
        offsets = [o for _, o in fake.requests]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(len(offsets), len(set(offsets)), "offset must advance")

    def test_absent_title_returns_none_without_infinite_loop(self):
        fake = FakeHermes(700, "not-present", 0)
        api = make_api(fake)
        with mock.patch("server.requests.get", fake.get):
            found = api.find_session_by_title("missing-title")
            taken = False
        self.assertIsNone(found)
        self.assertFalse(taken)
        self.assertLessEqual(len(fake.requests), 10)

    def test_same_title_under_a_different_model_is_the_same_conversation(self):
        found, _, _ = self._lookup(total=600, target_index=599,
                                   model="some-other-model")
        self.assertEqual(found, "api_target")

    def test_persisted_session_is_reused_after_model_change(self):
        import server
        api = make_api(FakeHermes(1, "owner-main", 0, target_model="old-model"))
        api.cfg["model"] = "new-owner-route"
        api._load_state = lambda: {"owner-main": "api_target"}
        saved = []
        api._save_state = lambda state: saved.append(dict(state))

        def get(url, headers=None, params=None, timeout=None):
            if url.endswith("/api/sessions/api_target"):
                return mock.Mock(ok=True, status_code=200, json=lambda: {
                    "session": {"id": "api_target", "title": "owner-main", "model": "old-model"}
                })
            raise AssertionError(f"unexpected GET {url}")

        with mock.patch("server.requests.get", get), \
             mock.patch("server.requests.post") as post:
            self.assertEqual(api.get_session_id("owner-main"), "api_target")
        post.assert_not_called()
        self.assertEqual(saved, [])

    def test_stale_mapping_falls_back_to_exact_title_regardless_of_model(self):
        fake = FakeHermes(450, "owner-main", 449, target_model="old-model")
        api = make_api(fake)
        api.cfg["model"] = "new-owner-route"
        api._load_state = lambda: {"owner-main": "stale-id"}
        saved = []
        api._save_state = lambda state: saved.append(dict(state))

        def get(url, headers=None, params=None, timeout=None):
            if url.endswith("/api/sessions/stale-id"):
                return mock.Mock(ok=False, status_code=404)
            return fake.get(url, headers=headers, params=params, timeout=timeout)

        with mock.patch("server.requests.get", get), \
             mock.patch("server.requests.post") as post:
            self.assertEqual(api.get_session_id("owner-main"), "api_target")
        post.assert_not_called()
        self.assertEqual(saved[-1], {"owner-main": "api_target"})

    def test_stops_when_server_ignores_offset(self):
        """A server that replays page one must not spin us forever."""
        class StuckHermes(FakeHermes):
            def get(self, url, headers=None, params=None, timeout=None):
                self.requests.append((PAGE_CAP, 0))
                return mock.Mock(ok=True, status_code=200, json=lambda: {
                    "data": self.sessions[:PAGE_CAP], "has_more": True,
                })

        fake = StuckHermes(1000, "nope", 0)
        api = make_api(fake)
        with mock.patch("server.requests.get", fake.get):
            found = api.find_session_by_title("missing")
        self.assertIsNone(found)
        self.assertLess(len(fake.requests), 600, "scan cap must terminate the loop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
