#!/usr/bin/env python3
"""Hermes LAN voice pipeline server (v3 — sessions, stop, approvals, partials).

WebSocket protocol (client → server):
  {"type":"start", "sample_rate":16000, "format":"pcm_s16le", "channels":1,
   "conversation": "jarvis-main"?}          begin a turn (mid-turn = barge-in)
  <binary int16 16 kHz mono PCM chunks>
  {"type":"stop"}                            end of speech, process turn
  {"type":"stop_run"}                        halt the running agent turn
  {"type":"approval_decision", "approval_token":..., "decision":"once"|"deny"}
  {"type":"stop_operator_run"}              stop the HUD-owned Runs-API turn

Server → client JSON events:
  status, transcript, partial_transcript, agent_status{thinking|tool_use|speaking},
  run_started{run_id}, operator_run_started{run_id}, approval_request{...},
  operator_run_terminal{completed|failed|timed_out}, error, done{timing}
plus binary 16 kHz mono int16 PCM TTS audio.

Brain: voice uses Hermes Session streaming; typed HUD operator turns use the
approval-capable Hermes Runs API (/v1/runs and /v1/runs/{id}/events). The
session stream is never used to resume a tool approval because it has no
resolvable Runs-API approval contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterator

import requests
import uvicorn
import yaml
import numpy as np
from anthropic import Anthropic
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from RealtimeSTT import AudioToTextRecorder

try:
    import psutil
except ImportError:  # machines panel degrades gracefully
    psutil = None

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "server.yaml"
LOG_PATH = ROOT / "logs" / "latency.jsonl"
STATE_PATH = ROOT / "logs" / "hermes_sessions.json"
USAGE_PATH = ROOT / "logs" / "usage_stats.json"
_USAGE_LOCK = threading.Lock()
_RUNTIME_STATUS_LOCK = threading.Lock()
_LAST_HERMES_RUNTIME: dict = {}


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def record_usage(llm_in: int = 0, llm_out: int = 0, turns: int = 0, tts_chars: int = 0) -> None:
    """Accumulate token/character usage into logs/usage_stats.json (total + per-day)."""
    with _USAGE_LOCK:
        try:
            data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {"total": {}, "days": {}}
        day = data["days"].setdefault(_today(), {})
        for bucket in (data["total"], day):
            bucket["llm_in"] = bucket.get("llm_in", 0) + llm_in
            bucket["llm_out"] = bucket.get("llm_out", 0) + llm_out
            bucket["turns"] = bucket.get("turns", 0) + turns
            bucket["tts_chars"] = bucket.get("tts_chars", 0) + tts_chars
        # keep last 60 days
        for k in sorted(data["days"])[:-60]:
            del data["days"][k]
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        USAGE_PATH.write_text(json.dumps(data), encoding="utf-8")


def read_usage() -> dict:
    with _USAGE_LOCK:
        try:
            data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {"total": {}, "days": {}}
    return {"total": data.get("total", {}), "today": data.get("days", {}).get(_today(), {})}
ENV_PATHS = [Path.home() / ".hermes" / ".env", ROOT / ".env"]
SENTENCE_RE = re.compile(r"(.+?[.!?])(?=\s|$)", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
CODEBLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# Secret-shaped strings are never sent to cloud TTS (privacy filter):
SECRET_RES = [
    re.compile(r"\b(?:api[_-]?key|secret|password|passwd|token|bearer|authorization)\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk|key|tok|ghp|xox[abp])[-_][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9+/_\-]{36,}\b"),          # long opaque blobs (keys, JWT segments)
    re.compile(r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----", re.DOTALL),
]


def load_env() -> None:
    for path in ENV_PATHS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@dataclass
class TurnTiming:
    turn_id: int
    audio_start_monotonic: float | None = None
    end_of_speech_monotonic: float | None = None
    stt_start_monotonic: float | None = None
    stt_final_monotonic: float | None = None
    llm_start_monotonic: float | None = None
    llm_first_token_monotonic: float | None = None
    first_sentence_monotonic: float | None = None
    tts_request_start_monotonic: float | None = None
    first_tts_audio_byte_monotonic: float | None = None
    total_done_monotonic: float | None = None
    transcript: str = ""
    response_text: str = ""
    stt_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    tts_model: str = ""
    voice_id: str = ""
    run_id: str = ""
    interrupted: bool = False
    tools_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        eos = self.end_of_speech_monotonic
        return {
            "turn_id": self.turn_id,
            "transcript": self.transcript,
            "response_text": self.response_text,
            "stt_model": self.stt_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tts_model": self.tts_model,
            "voice_id": self.voice_id,
            "run_id": self.run_id,
            "interrupted": self.interrupted,
            "tools_used": self.tools_used,
            "stt_finalize_seconds": self._delta(self.stt_start_monotonic, self.stt_final_monotonic),
            "llm_time_to_first_token_seconds": self._delta(self.llm_start_monotonic, self.llm_first_token_monotonic),
            "time_to_first_tts_audio_byte_seconds": self._delta(self.tts_request_start_monotonic, self.first_tts_audio_byte_monotonic),
            "end_of_speech_to_first_audio_seconds": self._delta(eos, self.first_tts_audio_byte_monotonic),
            "total_turn_seconds": self._delta(eos, self.total_done_monotonic),
            "errors": self.errors,
        }

    @staticmethod
    def _delta(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round(end - start, 4)


# ===================================================================== Hermes


class HermesAPI:
    """Thin client for the Hermes Agent API server (sessions, runs, approvals)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("hermes") or {}

    @property
    def base(self) -> str:
        return (self.cfg.get("base_url") or "http://127.0.0.1:8642").rstrip("/")

    def headers(self) -> dict:
        key = os.environ.get(self.cfg.get("api_key_env", "API_SERVER_KEY"), "")
        if not key:
            raise RuntimeError("Hermes API key not found in environment")
        h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if self.cfg.get("session_key"):
            h["X-Hermes-Session-Key"] = self.cfg["session_key"]
        return h

    # ---- persistent named sessions ----
    def _load_state(self) -> dict:
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: dict) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state), encoding="utf-8")

    # Hermes paginates /api/sessions: `limit` defaults to 50 and is CLAMPED
    # SERVER-SIDE TO 200, with `offset` and `has_more` for paging. Asking for a
    # larger page therefore does not widen the search — the request is silently
    # truncated — so a named conversation sitting past the first page is simply
    # not found, Zod tries to recreate the title, Hermes answers
    # 400 invalid_title, and the conversation dies. Page properly instead.
    _SESSION_PAGE_SIZE = 200
    # Safety valve so a misbehaving API can never spin this forever.
    _SESSION_SCAN_CAP = 100_000

    def iter_sessions(self):
        """Yield every session, following Hermes's pagination to the end."""
        offset = 0
        scanned = 0
        while True:
            r = requests.get(
                f"{self.base}/api/sessions",
                headers=self.headers(),
                params={"limit": self._SESSION_PAGE_SIZE, "offset": offset},
                timeout=15,
            )
            if not r.ok:
                return
            try:
                body = r.json() or {}
            except ValueError:
                return
            items = body.get("data") or []
            for item in items:
                yield item
            scanned += len(items)
            # Stop on an explicit has_more=False, on an empty page, or if the
            # server ignored `offset` and would replay the same page forever.
            if not items or not body.get("has_more") or scanned >= self._SESSION_SCAN_CAP:
                return
            offset += len(items)

    def find_session_by_title(self, name: str) -> str | None:
        """Return the Hermes session with this exact conversation title.

        Conversation identity is deliberately independent of model identity. A session may
        have been created while Zod used one model and then continue after the owner route
        changes to another model/alias. The run itself pins provider/model; the session is
        only the durable conversation/memory scope.
        """
        for item in self.iter_sessions():
            if item.get("title") == name and item.get("id"):
                return item["id"]
        return None

    def get_session_id(self, name: str, force_new: bool = False) -> str:
        state = self._load_state()
        sid = state.get(name)

        def session_exists(session_id: str) -> bool:
            try:
                check = requests.get(
                    f"{self.base}/api/sessions/{session_id}",
                    headers=self.headers(),
                    timeout=10,
                )
                return bool(check.ok)
            except Exception:
                return False

        # A model change must never mint a new conversation. `start_run` pins the current
        # provider/model independently, so a valid persisted Hermes session is reusable
        # regardless of whichever model created or last touched it.
        if sid and not force_new and session_exists(sid):
            return sid
        if sid:
            state.pop(name, None)
            self._save_state(state)

        if not force_new:
            try:
                found = self.find_session_by_title(name)
                if found:
                    state[name] = found
                    self._save_state(state)
                    return found
            except Exception:
                pass

        create_title = name if not force_new else f"{name}-{int(time.time())}"
        r = requests.post(f"{self.base}/api/sessions", headers=self.headers(),
                          json={"title": create_title}, timeout=15)

        if not force_new and r.status_code == 400 and "invalid_title" in r.text:
            # A race can create the title between our scan and POST. Search the entire
            # paginated list again and reuse it; model metadata is irrelevant to identity.
            found = self.find_session_by_title(name)
            if found:
                state[name] = found
                self._save_state(state)
                return found

        r.raise_for_status()
        data = r.json()
        sid = (data.get("session") or data).get("id")
        if not sid:
            raise RuntimeError(f"Hermes session creation returned no id: {data}")
        state[name] = sid
        self._save_state(state)
        print(f"Created Hermes session '{create_title}' -> {sid}", flush=True)
        return sid

    def stop_run(self, run_id: str) -> dict:
        r = requests.post(f"{self.base}/v1/runs/{run_id}/stop", headers=self.headers(), timeout=15)
        return {"status_code": r.status_code, "body": r.text[:300]}

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        """Load the durable Hermes text transcript for a Runs-API turn."""
        r = requests.get(
            f"{self.base}/api/sessions/{session_id}/messages",
            headers=self.headers(),
            params={"limit": 500, "order": "oldest"},
            timeout=15,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Hermes session history HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError as exc:
            raise RuntimeError("Hermes session history was not JSON") from exc
        messages = data.get("data") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            raise RuntimeError("Hermes session history was malformed")
        # Runs accepts only role/content pairs. Preserve completed user/assistant
        # text, but omit historical tool protocol rows that cannot be replayed
        # safely without their original tool-call IDs.
        return [
            {"role": item["role"], "content": item["content"]}
            for item in messages
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            and item["content"].strip()
        ]

    def get_run_status(self, run_id: str) -> dict:
        """Read Hermes's authoritative terminal status after an SSE failure."""
        r = requests.get(f"{self.base}/v1/runs/{run_id}", headers=self.headers(), timeout=15)
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code != 200 or not isinstance(data, dict):
            raise RuntimeError(f"Hermes run status HTTP {r.status_code}")
        return data

    def start_run(self, input_text: str, session_id: str, strict: bool = False) -> dict:
        """Start one Hermes Runs-API turn using the frozen provider/model route.

        ``strict`` opts the run into Hermes's strict approval mode, which is the
        only mode that carries per-approval correlation identifiers
        (``approval_id`` + ``tool_call_id``) and arms the shared atomic
        stop-before-spawn guard. Consequential actions must run strict; plain
        conversation and read-only inspection stay non-strict so they keep the
        full toolset (strict restricts the run to the terminal toolset).
        """
        model = self.cfg.get("model")
        provider = self.cfg.get("provider")
        if not model or not provider:
            raise RuntimeError("Hermes model and provider must be configured")
        conversation_history = self.get_session_history(session_id)
        payload = {
            "input": input_text,
            "session_id": session_id,
            "conversation_history": conversation_history,
            "provider": provider,
            "model": model,
            "require_model_lock": True,
        }
        if strict:
            payload["strict_approval"] = True
        r = requests.post(f"{self.base}/v1/runs", headers=self.headers(), json=payload, timeout=15)
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code != 202:
            raise RuntimeError(f"Hermes run start HTTP {r.status_code}")
        run_id = data.get("run_id") if isinstance(data, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Hermes run start returned no run_id")
        return {"run_id": run_id}

    def run_events(self, run_id: str) -> Iterator[dict]:
        """Yield the structured SSE events for one real Hermes run.

        Hermes Runs SSE uses JSON in ``data:`` frames instead of named SSE
        events. Parsing failures deliberately abort the observer so its caller
        can fail the run closed before an unresolved action can continue.
        """
        resp = requests.get(
            f"{self.base}/v1/runs/{run_id}/events",
            headers={**self.headers(), "Accept": "text/event-stream"},
            stream=True,
            timeout=(10, 45),  # Hermes emits an SSE keepalive every 30 seconds.
        )
        if resp.status_code != 200:
            resp.close()
            raise RuntimeError(f"Hermes run events HTTP {resp.status_code}")
        resp.encoding = "utf-8"
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None or not raw or raw.startswith(":") or raw.startswith("event:"):
                    continue
                if not raw.startswith("data:"):
                    continue
                data_text = raw[5:].strip()
                if not data_text or len(data_text) > 512_000:
                    raise RuntimeError("Malformed Hermes run event")
                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Malformed Hermes run event") from exc
                if not isinstance(data, dict):
                    raise RuntimeError("Malformed Hermes run event")
                yield data
        finally:
            resp.close()

    def post_run_approval(self, run_id: str, choice: str, approval_id: str = "") -> dict:
        """Resolve exactly one pending Hermes Runs-API approval.

        A strict run requires the ``approval_id`` Hermes issued for that exact
        pending action; it rejects the call otherwise, which is what makes
        duplicate, wrong and stale decisions provably fail closed.
        """
        if choice not in {"once", "deny"}:
            raise ValueError("Unsupported Hermes approval choice")
        body: dict = {"choice": choice}
        if approval_id:
            body["approval_id"] = approval_id
        r = requests.post(f"{self.base}/v1/runs/{run_id}/approval", headers=self.headers(),
                          json=body, timeout=15)
        try:
            data = r.json()
        except ValueError:
            data = {}
        return {"status_code": r.status_code, "data": data if isinstance(data, dict) else {}}

    def chat_stream_events(self, session_id: str, input_text: str, timeout: float) -> Iterator[tuple[str, str]]:
        """Yield ("run"|"text"|"tool"|"approval"|"final", value) from a session turn."""
        model = self.cfg.get("model")
        provider = self.cfg.get("provider")
        if not model or not provider:
            raise RuntimeError("Hermes model and provider must be configured")
        payload = {
            "input": input_text,
            "provider": provider,
            "model": model,
            "require_model_lock": True,
        }
        resp = requests.post(
            f"{self.base}/api/sessions/{session_id}/chat/stream",
            headers={**self.headers(), "Accept": "text/event-stream"},
            json=payload, stream=True, timeout=(10, timeout),
        )
        if resp.status_code >= 400:
            resp.close()
            raise RuntimeError(f"Hermes session chat HTTP {resp.status_code}: {resp.text[:300]}")
        resp.encoding = "utf-8"  # SSE has no charset header; requests would assume latin-1 (mojibake)
        try:
            yield from self._parse_sse(resp)
        finally:
            resp.close()  # leaked FDs killed the server once (launchd limit is tiny)

    @staticmethod
    def _parse_sse(resp) -> Iterator[tuple[str, str]]:
        event_name = ""
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw.startswith("event: "):
                event_name = raw[7:].strip()
                continue
            if not raw.startswith("data: "):
                continue
            data_text = raw[6:].strip()
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            ev = event_name or data.get("event", "")
            runtime = data.get("runtime")
            if isinstance(runtime, dict):
                requested = runtime.get("requested") if isinstance(runtime.get("requested"), dict) else {}
                safe_runtime = {
                    "provider": str(runtime.get("provider") or ""),
                    "model": str(runtime.get("model") or ""),
                    "model_lock": str(runtime.get("model_lock") or ""),
                    "requested": {
                        "provider": str(requested.get("provider") or ""),
                        "model": str(requested.get("model") or ""),
                    },
                }
                if ev in ("run.started", "assistant.completed"):
                    print(f"Hermes {ev} runtime {json.dumps(safe_runtime, sort_keys=True)}", flush=True)
                if ev == "assistant.completed" and safe_runtime["model_lock"] == "confirmed":
                    with _RUNTIME_STATUS_LOCK:
                        _LAST_HERMES_RUNTIME.clear()
                        _LAST_HERMES_RUNTIME.update(safe_runtime)
            if ev == "run.started":
                yield ("run", data.get("run_id") or "")
            elif ev == "assistant.delta":
                d = data.get("delta") or ""
                if d:
                    yield ("text", d)
            elif ev == "tool.started":
                name = data.get("tool_name") or "tool"
                if name.startswith("_"):
                    continue  # internal pseudo-tools like _thinking
                yield ("tool", json.dumps({"name": name, "preview": (data.get("preview") or "")[:200]}))
            elif "approval" in ev:
                # Hermes session-chat has no resumable approval contract. Its
                # stream run_id is not a /v1/runs resource, so forwarding this
                # to a browser would create an unsafe, nonfunctional prompt.
                yield ("approval", "")
            elif ev == "assistant.completed":
                yield ("final", json.dumps({
                    "content": data.get("content") or "",
                    "interrupted": bool(data.get("interrupted")),
                }))
            elif ev in ("run.failed", "error"):
                raise RuntimeError(f"Hermes stream error: {data_text[:300]}")
            elif ev == "run.completed":
                usage = data.get("usage") or {}
                if usage:
                    record_usage(
                        llm_in=int(usage.get("input_tokens") or 0),
                        llm_out=int(usage.get("output_tokens") or 0),
                        turns=1,
                    )
            elif ev == "done":
                pass  # stream closes after this


# ==================================================================== Pipeline


class VoicePipelineServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.turn_counter = 0
        self.hermes = HermesAPI(cfg)
        self.stt_lock = asyncio.Lock()
        # Avoid potential initialization storm by lazy loading for now
        self._recorder = None
        self._recorder_init_failed = False


    @property
    def recorder(self):
        if getattr(self, "_recorder_init_failed", False):
            return None

        if self._recorder is None:
            try:
                initial_prompt = str(self.cfg["stt"].get("initial_prompt") or "").strip()
                self._recorder = AudioToTextRecorder(
                    model=self.cfg["stt"]["model"],
                    use_microphone=False,
                    spinner=False,
                    device=self.cfg["stt"].get("device", "cpu"),
                    compute_type=self.cfg["stt"].get("compute_type", "int8"),
                    sample_rate=int(self.cfg["stt"].get("sample_rate", 16000)),
                    language="en",
                    beam_size=1,
                    initial_prompt=initial_prompt or None,
                    faster_whisper_vad_filter=False,
                    no_log_file=True,
                )
            except Exception as exc:
                self._recorder_init_failed = True
                self._recorder = None
                print(f"STT recorder init failed once: {type(exc).__name__}: {exc}", flush=True)
                return None

        return self._recorder

    def shutdown(self) -> None:
        recorder = self._recorder
        self._recorder = None
        if recorder is not None:
            recorder.shutdown()

    def next_turn_id(self) -> int:
        self.turn_counter += 1
        return self.turn_counter

    async def transcribe(self, audio: bytes, timing: TurnTiming | None = None) -> str:
        if timing:
            timing.stt_start_monotonic = time.perf_counter()
        # 1) GPU worker (if configured and reachable) — big model, ~0.3s
        remote = self.cfg["stt"].get("remote") or {}
        if remote.get("url"):
            text = await asyncio.to_thread(self._remote_stt, audio, remote)
            if text is not None:
                if timing:
                    timing.stt_model = f"remote:{remote.get('name', 'gpu')}"
                    timing.stt_final_monotonic = time.perf_counter()
                return text
        # 2) local Whisper fallback
        sample_rate = int(self.cfg["stt"].get("sample_rate", 16000))
        samples = (np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0).copy()
        try:
            async with self.stt_lock:
                recorder = self.recorder
                if recorder is None:
                    text = ""
                else:
                    recorder.feed_audio(samples, original_sample_rate=sample_rate)
                    text = await asyncio.to_thread(recorder.perform_final_transcription, samples, True)
                    recorder.clear_audio_queue()
        except Exception as exc:
            # near-silent audio can make whisper raise ("No clip timestamps found");
            # treat as empty transcript instead of failing the turn
            print(f"local STT error treated as empty transcript: {exc}", flush=True)
            text = ""
        if timing:
            timing.stt_final_monotonic = time.perf_counter()
        return (text or "").strip()

    def _remote_stt(self, audio: bytes, remote: dict) -> str | None:
        """POST raw PCM to the GPU STT worker. None = unavailable (use fallback)."""
        headers = {"Content-Type": "application/octet-stream"}
        token = os.environ.get(remote.get("token_env", "JARVIS_HUD_TOKEN"), "")
        if token:
            headers["X-Jarvis-Token"] = token
        try:
            r = requests.post(remote["url"], data=audio, headers=headers,
                              timeout=float(remote.get("timeout", 6)))
            if r.ok:
                return (r.json().get("text") or "").strip()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ LLM

    def stream_llm_events_sync(
        self, transcript: str, timing: TurnTiming, conversation: str,
    ) -> Iterator[tuple[str, str]]:
        llm = self.cfg["llm"]
        provider = llm["provider"]
        timing.llm_start_monotonic = time.perf_counter()
        if provider == "hermes":
            try:
                h = self.cfg.get("hermes") or {}
                session_id = self.hermes.get_session_id(conversation)
                gen = self._hermes_turn(session_id, transcript, timing, h, conversation)
                first = next(gen)
            except StopIteration:
                return
            except Exception as exc:
                fb = (self.cfg.get("hermes") or {}).get("fallback_provider", "anthropic")
                print(f"Hermes unavailable ({type(exc).__name__}: {exc}); fallback={fb}", flush=True)
                timing.errors.append(f"hermes_fallback: {exc}")
                if not fb:
                    raise
                yield ("text", "Agent backend offline. Running in basic mode. ")
                provider = fb
            else:
                yield first
                yield from gen
                return
        timing.llm_provider = provider
        timing.llm_model = llm["model"]
        if provider == "anthropic":
            key = os.environ.get(llm.get("api_key_env", "ANTHROPIC_API_KEY"))
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not found")
            client = Anthropic(api_key=key)
            with client.messages.stream(
                model=llm["model"],
                max_tokens=int(llm.get("max_tokens", 220)),
                temperature=float(llm.get("temperature", 0.3)),
                system=self.cfg["persona"]["system_prompt"],
                messages=[{"role": "user", "content": transcript}],
            ) as stream:
                for text in stream.text_stream:
                    if text and timing.llm_first_token_monotonic is None:
                        timing.llm_first_token_monotonic = time.perf_counter()
                    yield ("text", text)
        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

    def _hermes_turn(
        self, session_id: str, transcript: str, timing: TurnTiming, h: dict, conversation: str,
    ) -> Iterator[tuple[str, str]]:
        timing.llm_provider = str(h.get("provider") or "hermes")
        timing.llm_model = str(h.get("model") or "hermes-agent")
        timeout = float(h.get("timeout", 240))
        try:
            it = self.hermes.chat_stream_events(session_id, transcript, timeout)
            for kind, value in it:
                if kind == "text" and timing.llm_first_token_monotonic is None:
                    timing.llm_first_token_monotonic = time.perf_counter()
                yield (kind, value)
        except RuntimeError as exc:
            # stale session id (e.g. Hermes DB reset) -> recreate once
            if "404" in str(exc):
                session_id = self.hermes.get_session_id(conversation, force_new=True)
                for kind, value in self.hermes.chat_stream_events(session_id, transcript, timeout):
                    if kind == "text" and timing.llm_first_token_monotonic is None:
                        timing.llm_first_token_monotonic = time.perf_counter()
                    yield (kind, value)
            else:
                raise

    # ------------------------------------------------------------------ TTS


    def tts_chunks_sync(self, text: str, timing: TurnTiming) -> Iterator[bytes]:
        import shutil
        import subprocess
        import tempfile

        voice = self.cfg.get("voice") or {}
        provider = str(voice.get("provider") or "macos").strip().lower()

        timing.tts_model = "macos-say"
        timing.voice_id = str(voice.get("macos_voice") or "system-default")
        timing.tts_request_start_monotonic = timing.tts_request_start_monotonic or time.perf_counter()
        record_usage(tts_chars=len(text))

        if provider not in ("macos", "macos-say", "say", "local"):
            timing.errors.append(f"unsupported_local_tts_provider:{provider}")
            return

        say_bin = shutil.which("say") or "/usr/bin/say"
        ffmpeg_bin = shutil.which("ffmpeg")

        if not ffmpeg_bin:
            timing.errors.append("local_tts_ffmpeg_missing")
            return

        tmp_name = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
                tmp_name = tmp.name

            cmd = [say_bin]
            macos_voice = str(voice.get("macos_voice") or "").strip()
            if macos_voice:
                cmd += ["-v", macos_voice]
            macos_rate = str(voice.get("macos_rate") or "").strip()
            if macos_rate.isdigit() and 80 <= int(macos_rate) <= 400:
                cmd += ["-r", macos_rate]
            cmd += ["-o", tmp_name, text]

            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=120,
            )

            proc = subprocess.Popen(
                [
                    ffmpeg_bin,
                    "-nostdin",
                    "-loglevel", "error",
                    "-i", tmp_name,
                    "-f", "s16le",
                    "-acodec", "pcm_s16le",
                    "-ac", "1",
                    "-ar", "16000",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            sent = False

            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                if not sent:
                    timing.first_tts_audio_byte_monotonic = time.perf_counter()
                    sent = True
                yield chunk

            stderr = proc.stderr.read().decode("utf-8", "ignore")
            rc = proc.wait(timeout=120)

            if rc != 0:
                timing.errors.append(f"local_tts_ffmpeg_exit:{rc}:{stderr[:200]}")

        except Exception as exc:
            timing.errors.append(f"local_tts_error:{type(exc).__name__}:{exc}")
            print(f"Local TTS nonfatal error: {type(exc).__name__}: {exc}", flush=True)
            return

        finally:
            if tmp_name:
                try:
                    Path(tmp_name).unlink()
                except OSError:
                    pass
    async def stream_response_audio(
        self, ws: WebSocket, transcript: str, timing: TurnTiming, conn: "ConnState",
    ) -> None:
        pending = ""
        full_response: list[str] = []
        spoken = False
        await ws.send_json({"type": "agent_status", "state": "thinking"})

        q: asyncio.Queue = asyncio.Queue()

        async def forward() -> None:
            try:
                async for item in self._async_llm_events(transcript, timing, conn.conversation):
                    await q.put(item)
                await q.put(None)
            except Exception as exc:
                await q.put(exc)

        forward_task = asyncio.create_task(forward())
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                kind, value = item
                if kind == "run":
                    timing.run_id = value
                    conn.current_run_id = value
                    await ws.send_json({"type": "run_started", "run_id": value})
                    continue
                if kind == "tool":
                    info = json.loads(value)
                    timing.tools_used.append(info.get("name", "tool"))
                    await ws.send_json({"type": "agent_status", "state": "tool_use",
                                        "tool": info.get("name"), "preview": info.get("preview", "")})
                    continue
                if kind == "approval":
                    # Only the Runs API carries a resolvable approval path.
                    # Halt an unexpected session-stream approval instead of
                    # accepting a client decision for an uncorrelated run.
                    if conn.current_run_id:
                        await asyncio.to_thread(self.hermes.stop_run, conn.current_run_id)
                    raise RuntimeError("Uncorrelated Hermes approval halted; use typed HUD operator chat.")
                if kind == "final":
                    info = json.loads(value)
                    timing.interrupted = info.get("interrupted", False)
                    continue
                # kind == "text"
                full_response.append(value)
                pending += value
                sentences, pending = self._extract_complete_sentences(pending)
                for sentence in sentences:
                    clean = self._clean_for_tts(sentence)
                    if not clean:
                        continue
                    if conn.spoken_sentences and clean == conn.spoken_sentences[-1]:
                        continue
                    if timing.first_sentence_monotonic is None:
                        timing.first_sentence_monotonic = time.perf_counter()
                    if not spoken:
                        await ws.send_json({"type": "agent_status", "state": "speaking"})
                        spoken = True
                    conn.spoken_sentences.append(clean)
                    await self._send_tts_sentence(ws, clean, timing)
            tail = self._clean_for_tts(self._dedupe_exact_repeat(pending))
            if tail and (not conn.spoken_sentences or tail != conn.spoken_sentences[-1]):
                conn.spoken_sentences.append(tail)
                await self._send_tts_sentence(ws, tail, timing)
        finally:
            if not forward_task.done():
                forward_task.cancel()
        timing.response_text = self._dedupe_exact_repeat("".join(full_response))

    async def _async_llm_events(
        self, transcript: str, timing: TurnTiming, conversation: str,
    ) -> AsyncIterator[tuple[str, str]]:
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for item in self.stream_llm_events_sync(transcript, timing, conversation):
                    loop.call_soon_threadsafe(q.put_nowait, item)
                loop.call_soon_threadsafe(q.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)

        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        await worker_task

    async def _send_tts_sentence(self, ws: WebSocket, sentence: str, timing: TurnTiming) -> None:
        if not sentence:
            return
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in self.tts_chunks_sync(sentence, timing):
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
                loop.call_soon_threadsafe(q.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)

        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            await ws.send_bytes(item)
        await worker_task

    @staticmethod
    def _extract_complete_sentences(text: str) -> tuple[list[str], str]:
        sentences = []
        last_end = 0
        for match in SENTENCE_RE.finditer(text):
            sentences.append(match.group(1).strip())
            last_end = match.end()
        return sentences, text[last_end:]

    @staticmethod
    def _dedupe_exact_repeat(text: str) -> str:
        """Collapse a response that is exactly repeated by the upstream model."""
        text = text.strip()
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1 and len(lines) % 2 == 0:
            half = len(lines) // 2
            if lines[:half] == lines[half:]:
                return "\n".join(lines[:half])
        words = text.split()
        if len(words) >= 4 and len(words) % 2 == 0:
            half = len(words) // 2
            if words[:half] == words[half:]:
                return " ".join(words[:half])
        return text

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        if not text:
            return ""
        text = THINK_RE.sub("", text)
        for pattern in SECRET_RES:                  # privacy: never speak secrets
            text = pattern.sub(" redacted ", text)
        text = CODEBLOCK_RE.sub(" code omitted. ", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"^[\s>*#-]+", "", text)
        text = re.sub(r"[*_#]{1,3}([^*_#]+)[*_#]{1,3}", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def log_turn(self, timing: TurnTiming) -> None:
        timing.total_done_monotonic = timing.total_done_monotonic or time.perf_counter()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary = timing.summary()
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print("TURN TIMING", json.dumps(summary, ensure_ascii=False), flush=True)


load_env()
CFG = load_config()
HERMES = HermesAPI(CFG)   # lightweight API client - independent of the STT pipeline


def _safe_hud_text(value: object, limit: int = 600) -> str:
    """Keep approval metadata bounded and redact it again before browser egress."""
    if not isinstance(value, str):
        return ""
    text = value.replace("\x00", " ")
    for secret_re in SECRET_RES:
        text = secret_re.sub("[REDACTED]", text)
    if len(text) > limit:
        return text[:limit] + " [truncated]"
    return text


@dataclass
class HUDChannel:
    channel_id: str
    ws: WebSocket
    closed: bool = False
    starting: bool = False
    cancel_start: bool = False
    runs: set[str] = field(default_factory=set)


@dataclass
class PendingOperatorApproval:
    token: str
    run_id: str
    channel_id: str
    fingerprint: str
    summary: dict
    allow_once: bool
    # Hermes strict-run correlation identifiers. Empty on a non-strict run,
    # which is exactly why a non-strict run may only ever be denied.
    approval_id: str = ""
    tool_call_id: str = ""
    state: str = "pending"
    decision: str | None = None
    expiry_task: asyncio.Task | None = None


@dataclass
class OperatorRun:
    run_id: str
    session_id: str
    channel_id: str
    done: asyncio.Future
    strict: bool = False
    tools: list[dict] = field(default_factory=list)
    last_tool: str = "Hermes tool"
    pending: PendingOperatorApproval | None = None
    approvals: list[PendingOperatorApproval] = field(default_factory=list)
    seen_approval_fingerprints: set[str] = field(default_factory=set)
    seen_approval_ids: set[str] = field(default_factory=set)
    terminal: bool = False
    terminal_state: str = ""
    timing_out: bool = False
    failing_closed: bool = False
    failure_state: str = ""
    failure_reason: str = ""
    observer_task: asyncio.Task | None = None
    timeout_task: asyncio.Task | None = None


class OperatorBridgeError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class OperatorApprovalBridge:
    """Own one real Hermes Runs-API stream and its fail-closed approvals.

    Hermes v0.20's Runs API exposes a real ``run_id`` but no approval or tool
    invocation identifier. Hermes serializes its per-run approval queue FIFO,
    so this bridge permits exactly one observed pending approval at a time and
    uses an opaque, server-issued browser token only to bind the UI response to
    that run and WebSocket. The token is never forwarded to Hermes.
    """

    def __init__(self, hermes: HermesAPI, cfg: dict):
        self.hermes = hermes
        self._channels: dict[str, HUDChannel] = {}
        self._runs: dict[str, OperatorRun] = {}
        self._pending: dict[str, PendingOperatorApproval] = {}
        self._state_lock = threading.RLock()
        hermes_cfg = cfg.get("hermes") or {}
        try:
            run_timeout = int(hermes_cfg.get("timeout", 240))
        except (TypeError, ValueError):
            run_timeout = 240
        self.run_timeout = max(30, min(run_timeout, 3600))
        # Hermes's own default is 300 seconds. Ending sooner or at the same
        # time is safe; a UI timeout is always a denial, never consent.
        self.approval_timeout = 300

    def open_channel(self, ws: WebSocket) -> HUDChannel:
        channel = HUDChannel(channel_id=secrets.token_urlsafe(24), ws=ws)
        with self._state_lock:
            self._channels[channel.channel_id] = channel
        return channel

    def _channel(self, channel_id: str) -> HUDChannel | None:
        with self._state_lock:
            channel = self._channels.get(channel_id)
            return channel if channel and not channel.closed else None

    async def _send(self, channel_id: str, payload: dict) -> bool:
        channel = self._channel(channel_id)
        if channel is None:
            return False
        try:
            await channel.ws.send_json(payload)
            return True
        except Exception:
            return False

    async def start(self, input_text: str, conversation: str, channel_id: str,
                    strict: bool = False) -> OperatorRun:
        if not isinstance(channel_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", channel_id):
            raise OperatorBridgeError("HUD approval channel is unavailable.", 409)
        channel = self._channel(channel_id)
        if channel is None:
            raise OperatorBridgeError("HUD approval channel is unavailable.", 409)
        if not isinstance(conversation, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", conversation):
            raise OperatorBridgeError("Invalid conversation identifier.", 400)
        with self._state_lock:
            active = [
                run_id for run_id in channel.runs
                if run_id in self._runs and not self._runs[run_id].terminal
            ]
            if active or channel.starting:
                starting_conflict = True
            else:
                starting_conflict = False
                channel.starting = True
                channel.cancel_start = False
        if starting_conflict:
            raise OperatorBridgeError("An operator run is already active for this HUD.", 409)

        try:
            # Retain the existing named Hermes session/memory scope while the
            # actual turn uses the approval-capable Runs API.
            session_id = await asyncio.to_thread(self.hermes.get_session_id, conversation)
            started = await asyncio.to_thread(
                self.hermes.start_run, input_text, session_id, strict
            )
        except Exception as exc:
            with self._state_lock:
                current = self._channels.get(channel_id)
                if current is channel:
                    current.starting = False
            print(f"Hermes operator run start failed: {type(exc).__name__}", flush=True)
            raise OperatorBridgeError("Hermes operator run could not be started.", 502) from exc

        run_id = started["run_id"]
        run = OperatorRun(
            run_id=run_id,
            session_id=session_id,
            channel_id=channel_id,
            done=asyncio.get_running_loop().create_future(),
            strict=strict,
        )
        with self._state_lock:
            channel = self._channels.get(channel_id)
            if channel is None or channel.closed or channel.cancel_start:
                channel_lost = channel is None or channel.closed
                start_cancelled = not channel_lost
                if channel is not None:
                    channel.starting = False
            else:
                channel_lost = False
                start_cancelled = False
                channel.starting = False
                channel.runs.add(run_id)
                self._runs[run_id] = run
        if channel_lost or start_cancelled:
            # A newly created run must never become unattended before the HUD
            # can observe it. Stop immediately rather than returning its id.
            try:
                await asyncio.to_thread(self.hermes.stop_run, run_id)
            except Exception:
                pass
            message = "HUD connection closed before the run could be observed." if channel_lost else "Operator run was stopped before it could be observed."
            raise OperatorBridgeError(message, 409)

        run.observer_task = asyncio.create_task(self._observe(run))
        run.timeout_task = asyncio.create_task(self._expire_run(run))
        if not await self._send(channel_id, {"type": "operator_run_started", "run_id": run_id}):
            await self._fail_closed(run, "Run start could not be delivered to the HUD.")
        return run

    async def wait_for_terminal(self, run: OperatorRun) -> dict:
        result = await run.done
        # The HTTP caller now owns the terminal result. Retain no raw output or
        # approval metadata after delivery; a late token becomes stale and is
        # rejected rather than being rebound to a future run.
        with self._state_lock:
            if self._runs.get(run.run_id) is run:
                self._runs.pop(run.run_id, None)
            for approval in run.approvals:
                self._pending.pop(approval.token, None)
        return result

    async def _observe(self, run: OperatorRun) -> None:
        events: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def consume() -> None:
            try:
                for event in self.hermes.run_events(run.run_id):
                    loop.call_soon_threadsafe(events.put_nowait, ("event", event))
            except Exception as exc:
                loop.call_soon_threadsafe(events.put_nowait, ("error", type(exc).__name__))
            finally:
                loop.call_soon_threadsafe(events.put_nowait, ("closed", None))

        worker = asyncio.create_task(asyncio.to_thread(consume))
        try:
            while True:
                kind, payload = await events.get()
                if kind == "event":
                    await self._handle_event(run, payload)
                    continue
                if kind == "error" and not run.terminal:
                    await self._fail_closed(run, "Hermes run event stream failed.")
                if kind == "closed":
                    if not run.terminal:
                        await self._fail_closed(run, "Hermes run event stream closed unexpectedly.")
                        await self._poll_terminal_after_stream_loss(run)
                    break
        except asyncio.CancelledError:
            if not run.terminal:
                await self._fail_closed(run, "Hermes run observer was cancelled.")
            raise
        finally:
            # The worker exits with the SSE stream. Do not cancel a live
            # requests thread: fail_closed() asks Hermes to stop it instead.
            if worker.done():
                try:
                    await worker
                except Exception:
                    pass

    async def _poll_terminal_after_stream_loss(self, run: OperatorRun) -> None:
        """Keep ownership until Hermes confirms that a stopped run is terminal."""
        while not run.terminal:
            try:
                status = await asyncio.to_thread(self.hermes.get_run_status, run.run_id)
            except Exception:
                await asyncio.sleep(1)
                continue
            state = status.get("status")
            if state == "completed":
                if run.failing_closed:
                    await self._finish(
                        run,
                        run.failure_state or "failed",
                        error=run.failure_reason or "Run completed after fail-closed stop.",
                    )
                else:
                    await self._finish(run, "completed", output=str(status.get("output") or ""))
                return
            if state == "failed":
                await self._finish(
                    run,
                    run.failure_state if run.failing_closed else "failed",
                    error=run.failure_reason or _safe_hud_text(status.get("error"), 800),
                )
                return
            if state == "cancelled":
                await self._finish(
                    run,
                    run.failure_state or ("timed_out" if run.timing_out else "cancelled"),
                    error=run.failure_reason or ("Run timed out." if run.timing_out else "Run cancelled."),
                )
                return
            await asyncio.sleep(1)

    async def _handle_event(self, run: OperatorRun, event: dict) -> None:
        if run.terminal:
            return
        event_name = event.get("event")
        if not isinstance(event_name, str) or event.get("run_id") != run.run_id:
            await self._fail_closed(run, "Uncorrelated Hermes run event.")
            return
        if event_name == "message.delta" or event_name == "reasoning.available":
            return
        if event_name == "tool.started":
            tool = _safe_hud_text(event.get("tool"), 80) or "Hermes tool"
            preview = _safe_hud_text(event.get("preview"), 240)
            run.last_tool = tool
            run.tools.append({"name": tool, "preview": preview})
            if not await self._send(run.channel_id, {
                "type": "agent_status", "state": "tool_use", "tool": tool, "preview": preview,
            }):
                await self._fail_closed(run, "Tool activity could not be delivered to the HUD.")
            return
        if event_name == "tool.completed":
            tool = _safe_hud_text(event.get("tool"), 80) or run.last_tool
            state = "tool_failed" if bool(event.get("error")) else "tool_completed"
            if not await self._send(run.channel_id, {"type": "agent_status", "state": state, "tool": tool}):
                await self._fail_closed(run, "Tool result could not be delivered to the HUD.")
            return
        if event_name == "approval.request":
            await self._offer_approval(run, event)
            return
        if event_name == "approval.responded":
            await self._approval_responded(run, event)
            return
        if event_name == "run.completed":
            if run.failing_closed:
                await self._finish(run, "failed", error=run.failure_reason or "Run completed after fail-closed stop.")
                return
            if run.pending is not None:
                await self._fail_closed(run, "Run completed while an approval was still unresolved.")
                return
            await self._finish(run, "completed", output=str(event.get("output") or ""))
            return
        if event_name == "run.failed":
            await self._finish(run, "failed", error=run.failure_reason or _safe_hud_text(event.get("error"), 800))
            return
        if event_name == "run.cancelled":
            await self._finish(
                run,
                run.failure_state or ("timed_out" if run.timing_out else "cancelled"),
                error=run.failure_reason or ("Run timed out." if run.timing_out else "Run cancelled."),
            )
            return
        if "approval" in event_name:
            await self._fail_closed(run, "Unknown Hermes approval lifecycle event.")

    @staticmethod
    def _approval_fingerprint(event: dict) -> str:
        data = {
            "command": event.get("command"),
            "description": event.get("description"),
            "pattern_key": event.get("pattern_key"),
            "pattern_keys": event.get("pattern_keys"),
            "choices": event.get("choices"),
        }
        encoded = json.dumps(data, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _approval_summary(self, run: OperatorRun, event: dict) -> dict | None:
        choices = event.get("choices")
        if not isinstance(choices, list) or not {"once", "deny"}.issubset({str(x) for x in choices}):
            return None
        action = _safe_hud_text(event.get("command"), 700)
        reason = _safe_hud_text(event.get("description"), 700)
        if not action and not reason:
            return None
        summary = {
            "tool": run.last_tool,
            "action": action or reason,
            "reason": reason or "Hermes requires explicit approval for this action.",
        }
        working_directory = _safe_hud_text(event.get("working_directory") or event.get("cwd"), 300)
        if working_directory:
            summary["working_directory"] = working_directory
        pattern_keys = event.get("pattern_keys")
        if isinstance(pattern_keys, list):
            risk = _safe_hud_text(", ".join(str(key) for key in pattern_keys[:4]), 240)
            if risk:
                summary["risk"] = risk
        return summary

    async def _offer_approval(self, run: OperatorRun, event: dict) -> None:
        summary = self._approval_summary(run, event)
        if summary is None:
            await self._fail_closed(run, "Hermes approval request could not be parsed safely.")
            return
        approval_id = str(event.get("approval_id") or "")
        tool_call_id = str(event.get("tool_call_id") or "")
        if run.strict and (
            not re.fullmatch(r"appr_[0-9a-f]{32}", approval_id) or not tool_call_id
        ):
            # Hermes refuses to emit a strict approval without both identifiers,
            # so one arriving without them cannot be correlated and is unsafe.
            await self._fail_closed(run, "Strict approval request lacked correlation identifiers.")
            return
        fingerprint = self._approval_fingerprint(event)
        with self._state_lock:
            if run.strict:
                already_seen = approval_id in run.seen_approval_ids
                repeat_of_pending = (
                    run.pending is not None and run.pending.approval_id == approval_id
                )
            else:
                already_seen = fingerprint in run.seen_approval_fingerprints
                repeat_of_pending = (
                    run.pending is not None and run.pending.fingerprint == fingerprint
                )
            if run.pending is not None:
                if repeat_of_pending:
                    return  # duplicate SSE delivery for the same pending item
                invalid = True
            elif already_seen:
                # Replaying a resolved request must never authorize a later
                # action. Strict keys on Hermes's per-approval id; non-strict has
                # only the action shape to go on.
                invalid = True
            else:
                invalid = False
                approval = PendingOperatorApproval(
                    token=secrets.token_urlsafe(32),
                    run_id=run.run_id,
                    channel_id=run.channel_id,
                    fingerprint=fingerprint,
                    summary=summary,
                    # Approve-once is offered only on a strict run: that is the
                    # only mode where Hermes issues a per-approval id, so it is
                    # the only mode where a permissive choice can be proven to
                    # target this exact action. A non-strict run stays
                    # deny-only, which is what stops a consequential action from
                    # ever executing outside the strict guarded path.
                    allow_once=run.strict,
                    approval_id=approval_id,
                    tool_call_id=tool_call_id,
                )
                run.pending = approval
                run.approvals.append(approval)
                run.seen_approval_fingerprints.add(fingerprint)
                if approval_id:
                    run.seen_approval_ids.add(approval_id)
                self._pending[approval.token] = approval
        if invalid:
            await self._fail_closed(run, "Multiple or replayed approvals cannot be correlated safely.")
            return
        if not await self._send(run.channel_id, {
            "type": "approval_request",
            "run_id": run.run_id,
            "approval_token": approval.token,
            "summary": summary,
            "can_approve_once": approval.allow_once,
            # Tells the HUD which guarantee is in force. Hermes's own
            # approval_id is deliberately never sent to the browser: the HUD
            # answers with its own opaque token, which the bridge maps back.
            "strict": run.strict,
        }):
            await self._fail_closed(run, "Approval request could not be delivered to the HUD.")
            return
        approval.expiry_task = asyncio.create_task(self._expire_approval(approval))

    async def decide(self, channel_id: str, approval_token: object, decision: object) -> dict:
        if not isinstance(approval_token, str) or len(approval_token) > 256:
            return {"accepted": False, "reason": "Invalid approval token."}
        if decision not in {"once", "deny"}:
            return {"accepted": False, "reason": "Invalid approval decision."}
        with self._state_lock:
            approval = self._pending.get(approval_token)
            if approval is None:
                return {"accepted": False, "reason": "Approval is stale or unknown."}
            if approval.channel_id != channel_id:
                return {"accepted": False, "reason": "Approval belongs to a different HUD connection."}
            if approval.state != "pending":
                return {"accepted": False, "reason": "Approval was already resolved."}
            if decision == "once" and not approval.allow_once:
                return {"accepted": False, "reason": "Hermes cannot safely correlate approve-once for this tool."}
            run = self._runs.get(approval.run_id)
            if run is None or run.terminal or run.pending is not approval:
                return {"accepted": False, "reason": "Approval is no longer active."}
            approval.state = "submitting"
            approval.decision = decision
        if approval.expiry_task:
            approval.expiry_task.cancel()
        return await self._submit_decision(approval)

    @staticmethod
    def _decision_confirmed(response: dict, approval: PendingOperatorApproval, choice: str) -> bool:
        data = response.get("data") or {}
        resolved = data.get("resolved")
        confirmed = (
            200 <= int(response.get("status_code") or 0) < 300
            and data.get("run_id") == approval.run_id
            and data.get("choice") == choice
            and isinstance(resolved, int)
            and not isinstance(resolved, bool)
            and resolved == 1
        )
        if approval.approval_id:
            # A strict decision is only confirmed when Hermes echoes back the
            # exact approval id we resolved, closing the correlation loop
            # between this HUD turn, the run, the action and the execution.
            confirmed = confirmed and data.get("approval_id") == approval.approval_id
        return confirmed

    async def _submit_decision(self, approval: PendingOperatorApproval) -> dict:
        with self._state_lock:
            run = self._runs.get(approval.run_id)
            choice = approval.decision
        if run is None or choice not in {"once", "deny"}:
            return {"accepted": False, "reason": "Approval is no longer active."}
        try:
            response = await asyncio.to_thread(
                self.hermes.post_run_approval, approval.run_id, choice, approval.approval_id
            )
        except Exception:
            response = {"status_code": 0, "data": {}}
        if not self._decision_confirmed(response, approval, choice):
            await self._fail_closed(run, "Hermes did not confirm the approval decision.")
            return {"accepted": False, "reason": "Hermes did not confirm the decision."}
        with self._state_lock:
            if approval.state == "submitting":
                approval.state = "submitted"
        if not await self._send(approval.channel_id, {
            "type": "approval_decision_sent",
            "approval_token": approval.token,
            "decision": choice,
        }):
            await self._fail_closed(run, "Approval outcome could not be delivered to the HUD.")
            return {"accepted": False, "reason": "HUD connection closed."}
        return {"accepted": True}

    async def _approval_responded(self, run: OperatorRun, event: dict) -> None:
        choice = event.get("choice")
        with self._state_lock:
            approval = run.pending
            valid = (
                approval is not None
                and approval.state in {"submitting", "submitted"}
                and approval.decision == choice
            )
            if valid:
                approval.state = "resolved"
                run.pending = None
        if not valid:
            await self._fail_closed(run, "Uncorrelated Hermes approval response.")
            return
        if approval.expiry_task:
            approval.expiry_task.cancel()
        state = "approved_once" if choice == "once" else "denied"
        if not await self._send(approval.channel_id, {
            "type": "approval_resolved",
            "approval_token": approval.token,
            "decision": choice,
            "state": state,
        }):
            await self._fail_closed(run, "Approval resolution could not be delivered to the HUD.")

    async def _expire_approval(self, approval: PendingOperatorApproval) -> None:
        try:
            await asyncio.sleep(self.approval_timeout)
        except asyncio.CancelledError:
            return
        with self._state_lock:
            run = self._runs.get(approval.run_id)
            if run is None or run.terminal or run.pending is not approval or approval.state != "pending":
                return
            approval.state = "submitting"
            approval.decision = "deny"
        await self._send(approval.channel_id, {
            "type": "approval_expired",
            "approval_token": approval.token,
            "message": "Approval timed out and was denied.",
        })
        await self._submit_decision(approval)

    async def _expire_run(self, run: OperatorRun) -> None:
        try:
            await asyncio.sleep(self.run_timeout)
        except asyncio.CancelledError:
            return
        if run.terminal:
            return
        run.timing_out = True
        await self._fail_closed(run, "Operator run timed out.", terminal_state="timed_out")

    async def _fail_closed(self, run: OperatorRun, reason: str, terminal_state: str = "failed") -> None:
        """Deny a known pending action and stop the run on every unsafe edge."""
        with self._state_lock:
            if run.terminal or run.failing_closed:
                return
            run.failing_closed = True
            run.failure_state = terminal_state
            run.failure_reason = reason
            approval = run.pending
            # Never issue a second permissive decision. A concurrent once may
            # already be in flight, so stop the run rather than retrying it.
            may_deny = approval is not None and approval.decision != "once"
            if may_deny and approval is not None:
                approval.state = "submitting"
                approval.decision = "deny"
        denial_confirmed = False
        if may_deny and approval is not None:
            try:
                response = await asyncio.to_thread(self.hermes.post_run_approval, run.run_id, "deny")
                denial_confirmed = self._decision_confirmed(response, approval, "deny")
            except Exception:
                denial_confirmed = False
        try:
            stop = await asyncio.to_thread(self.hermes.stop_run, run.run_id)
            stop_confirmed = stop.get("status_code") == 200
        except Exception:
            stop_confirmed = False
        if approval is not None:
            approval.state = "failed_closed"
            if approval.expiry_task:
                approval.expiry_task.cancel()
            await self._send(approval.channel_id, {
                "type": "approval_expired",
                "approval_token": approval.token,
                "message": "Approval blocked: " + reason,
            })
        suffix = " Approval denial was confirmed." if denial_confirmed else " Run stop was requested."
        if not stop_confirmed:
            suffix += " Hermes has not confirmed that the run stopped."
        run.failure_reason = reason + suffix
        # /stop is asynchronous. Keep this run owned and its event observer
        # alive until Hermes reports a real terminal state; never claim that a
        # possibly still-running agent has safely stopped.
        await self._send(run.channel_id, {
            "type": "operator_run_stopping",
            "run_id": run.run_id,
            "state": terminal_state,
            "error": run.failure_reason,
        })

    async def _finish(self, run: OperatorRun, state: str, output: str = "", error: str = "") -> None:
        with self._state_lock:
            if run.terminal:
                return
            run.terminal = True
            run.terminal_state = state
            if run.timeout_task and run.timeout_task is not asyncio.current_task():
                run.timeout_task.cancel()
            pending = run.pending
            run.pending = None
            if pending is not None:
                pending.state = "closed"
                if pending.expiry_task and pending.expiry_task is not asyncio.current_task():
                    pending.expiry_task.cancel()
            channel = self._channels.get(run.channel_id)
            if channel:
                channel.runs.discard(run.run_id)
            approval_state = None
            if any(item.decision == "deny" and item.state == "resolved" for item in run.approvals):
                approval_state = "denied"
            elif any(item.decision == "once" and item.state == "resolved" for item in run.approvals):
                approval_state = "approved_once"
            result = {
                "run_id": run.run_id,
                "state": state,
                "text": output,
                "error": error,
                "tools": list(run.tools),
                "approval_state": approval_state,
            }
            if not run.done.done():
                run.done.set_result(result)
        await self._send(run.channel_id, {
            "type": "operator_run_terminal",
            "run_id": run.run_id,
            "state": state,
            "approval_state": approval_state,
            "error": error,
        })

    async def stop_channel_runs(self, channel_id: str) -> bool:
        with self._state_lock:
            channel = self._channels.get(channel_id)
            starting = bool(channel and channel.starting)
            if starting and channel is not None:
                channel.cancel_start = True
            runs = [self._runs[run_id] for run_id in (channel.runs if channel else set())
                    if run_id in self._runs and not self._runs[run_id].terminal]
        for run in runs:
            await self._fail_closed(run, "Run stopped by the HUD user.")
        return bool(runs) or starting

    async def close_channel(self, channel_id: str) -> None:
        with self._state_lock:
            channel = self._channels.pop(channel_id, None)
            if channel is None:
                return
            channel.closed = True
            runs = [self._runs[run_id] for run_id in channel.runs
                    if run_id in self._runs and not self._runs[run_id].terminal]
        for run in runs:
            await self._fail_closed(run, "HUD connection closed before the run finished.")


OPERATOR_BRIDGE = OperatorApprovalBridge(HERMES, CFG)
PIPELINE: VoicePipelineServer | None = None


_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> VoicePipelineServer:
    """Lock prevents the four uvicorn listeners' startup hooks from racing
    into concurrent recorder inits (which crashed three of the four lifespans
    and silently killed the TLS ports)."""
    global PIPELINE
    if PIPELINE is None:
        with _PIPELINE_LOCK:
            if PIPELINE is None:
                PIPELINE = VoicePipelineServer(CFG)
    return PIPELINE


app = FastAPI(title="Hermes Voice Pipeline")



@app.get("/health")
async def zod_health():
    cfg = load_config()
    hermes_cfg = cfg.get("hermes") or {}
    base = str(hermes_cfg.get("base_url") or "http://127.0.0.1:8642").rstrip("/")
    online = False
    try:
        response = requests.get(base + "/health", timeout=3)
        online = response.status_code == 200
    except Exception:
        online = False

    return {
        "status": "ok" if online else "degraded",
        "service": "zod-hud",
        "hermes": "online" if online else "offline",
        "model": hermes_cfg.get("model"),
        "provider": hermes_cfg.get("provider"),
    }
@app.on_event("startup")
async def warm_pipeline() -> None:
    """Warm the local Whisper fallback in the BACKGROUND, exactly once (this
    hook fires once per uvicorn listener — there are four), and never let a
    warm failure take a listener down."""
    if not (CFG.get("stt") or {}).get("warm_on_startup", True):
        print("STT startup warm disabled; lazy initialization enabled.", flush=True)
        return
    global _WARM_STARTED
    if _WARM_STARTED:
        return
    _WARM_STARTED = True

    async def warm() -> None:
        try:
            await asyncio.to_thread(get_pipeline)
            print("STT pipeline warmed.", flush=True)
        except Exception as exc:
            print(f"STT warm failed (remote STT still available): {exc}", flush=True)

    asyncio.get_running_loop().create_task(warm())


_WARM_STARTED = False


@app.on_event("shutdown")
async def shutdown_pipeline() -> None:
    pipeline = PIPELINE
    if pipeline is None or pipeline._recorder is None:
        return
    try:
        await asyncio.to_thread(pipeline.shutdown)
    except Exception as exc:
        print(f"STT recorder shutdown failed: {type(exc).__name__}: {exc}", flush=True)


# ------------------------------------------------------------------ Auth

ALLOWED_ORIGIN_HOSTS = {"jarvis.local", "jarvis", "localhost", "127.0.0.1"}
ALLOWED_ORIGIN_HOSTS |= set((CFG.get("security") or {}).get("extra_origin_hosts") or [])


def hud_token() -> str | None:
    env_name = (CFG.get("security") or {}).get("hud_token_env", "JARVIS_HUD_TOKEN")
    return os.environ.get(env_name) or None


def _request_authed(request: Request) -> bool:
    token = hud_token()
    if not token:
        return True
    supplied = request.headers.get("x-jarvis-token") or request.cookies.get("jarvis_token")
    return supplied == token


@app.middleware("http")
async def api_auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/") and not _request_authed(request):
        return Response(status_code=401, content="jarvis auth required")
    return await call_next(request)


def _ws_allowed(ws: WebSocket) -> bool:
    """Browsers send Origin (+cookie); native clients (PTT, tests) send neither."""
    origin = ws.headers.get("origin")
    if not origin:
        return True  # non-browser client on the LAN (Python PTT, e2e tests)
    from urllib.parse import urlparse
    host = (urlparse(origin).hostname or "").lower()
    if host not in ALLOWED_ORIGIN_HOSTS:
        return False
    token = hud_token()
    if not token:
        return True
    return ws.cookies.get("jarvis_token") == token or ws.query_params.get("token") == token


# --------------------------------------------------------------- HUD + proxy

HUD_DIR = ROOT / "hud"
ALLOWED_GET_PATHS = {
    "/health", "/health/detailed", "/v1/capabilities",
    "/v1/skills", "/v1/toolsets", "/api/jobs", "/api/sessions",
}


def _proxy_allowed(method: str, path: str) -> bool:
    if method == "GET":
        return path in ALLOWED_GET_PATHS or (
            path.startswith("/api/sessions/") and path.endswith("/messages")
        )
    if method == "POST":
        return path == "/v1/responses"
    return False


@app.api_route("/api/hermes/{path:path}", methods=["GET", "POST"])
async def hermes_proxy(path: str, request: Request) -> Response:
    target = "/" + path
    if not _proxy_allowed(request.method, target):
        return Response(status_code=403, content="path not allowed")
    hermes = HERMES
    body = await request.body()
    params = dict(request.query_params)

    def do_request() -> requests.Response:
        return requests.request(
            request.method, hermes.base + target, params=params,
            headers=hermes.headers(), data=body if body else None, timeout=300,
        )

    resp = await asyncio.to_thread(do_request)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("Content-Type", "application/json"))


# Consequential-intent routing for the hybrid strict architecture. This is a
# routing hint, never the safety boundary: a non-strict run is deny-only
# (allow_once is False unless the run is strict), so a misroute can never let an
# unapproved state-changing action execute — at worst it costs the user a retry.
# Over-triggering is cheap too: a strict run still has the terminal tool, which
# is what read-only inspection uses anyway; it only gives up the wider toolsets.
_STRICT_INTENT_RE = re.compile(
    r"\b(?:"
    r"write|writes|writing|create|creates|creating|mkdir|touch|append|save|saves|"
    r"overwrite|overwrites|truncate|edit|edits|editing|modify|modifies|patch|"
    r"delete|deletes|deleting|remove|removes|removing|rm|erase|wipe|purge|"
    r"rename|renames|move|moves|moving|mv|chmod|chown|sudo|"
    r"install|installs|installing|uninstall|upgrade|pip|npm|brew|"
    r"commit|commits|push|pushes|merge|rebase|checkout|stash|"
    r"restart|restarts|reboot|kill|kills|launchctl|systemctl|"
    r"deploy|deploys|publish|publishes"
    r")\b",
    re.IGNORECASE,
)
# Shell constructs that write or destroy regardless of the surrounding words.
# Discard/stream devices and fd duplications are NOT writes: routing a
# read-only command that merely silences stderr (`2>/dev/null`, `2>&1`) into a
# strict run would cost it the wider toolset for no safety gain. Mirrors the
# same exclusion in Hermes's strict write gate.
_STRICT_SHELL_RE = re.compile(
    r"(?:"
    r"\d?>>?\s*(?!&)(?!/dev/(?:null|stdout|stderr|tty|fd/\d+)\b)[\"']?[^\s;&|<>\"']"
    r"|\brm\s+-|\btee\b|\bdd\s+if="
    r")"
)


def _requires_strict_run(text: str) -> bool:
    """True when a turn's stated intent could change state."""
    return bool(_STRICT_INTENT_RE.search(text) or _STRICT_SHELL_RE.search(text))


@app.post("/api/chat")
async def hud_chat(request: Request) -> JSONResponse:
    """Typed HUD operator turn via Hermes's approval-capable Runs API."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    raw_text = body.get("input")
    if not isinstance(raw_text, str):
        return JSONResponse({"error": "input must be text"}, status_code=400)
    text = raw_text.strip()
    conversation = body.get("conversation") or (CFG.get("hermes") or {}).get("conversation", "jarvis-main")
    channel_id = body.get("channel_id")
    if not text:
        return JSONResponse({"error": "empty input"}, status_code=400)
    # Hybrid strict routing: a caller may pin the mode explicitly, otherwise the
    # stated intent decides. Consequential turns execute inside a strict run so
    # the approval is correlated and the atomic stop-before-spawn guard is armed
    # before any consequential tool runs; conversation and read-only inspection
    # stay non-strict and keep the full toolset.
    requested_strict = body.get("strict")
    strict = bool(requested_strict) if isinstance(requested_strict, bool) else _requires_strict_run(text)
    try:
        run = await OPERATOR_BRIDGE.start(text, conversation, channel_id, strict)
        result = await OPERATOR_BRIDGE.wait_for_terminal(run)
        result["strict"] = strict
        return JSONResponse(result)
    except OperatorBridgeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=exc.status_code)


@app.get("/api/usage")
async def usage() -> JSONResponse:
    """Local token and local TTS character usage."""
    u = read_usage()
    cost_cfg = CFG.get("usage") or {}
    cin = float(cost_cfg.get("llm_cost_per_mtok_input", 0) or 0)
    cout = float(cost_cfg.get("llm_cost_per_mtok_output", 0) or 0)

    def est(b: dict) -> float | None:
        if not (cin or cout):
            return None
        return round(b.get("llm_in", 0) / 1e6 * cin + b.get("llm_out", 0) / 1e6 * cout, 4)

    out = {
        "llm": {
            "today": u["today"], "total": u["total"],
            "today_cost": est(u["today"]), "total_cost": est(u["total"]),
        },
        "tts": {
            "today_chars": int(u["today"].get("tts_chars") or 0),
            "total_chars": int(u["total"].get("tts_chars") or 0),
        },
    }
    return JSONResponse(out)


@app.get("/api/loadout")
async def loadout() -> JSONResponse:
    """Safe effective runtime values for the HUD Models Loadout."""
    hermes_cfg = CFG.get("hermes") or {}
    stt_cfg = CFG.get("stt") or {}
    voice_cfg = CFG.get("voice") or {}
    model = str(hermes_cfg.get("model") or "Unavailable")
    provider = str(hermes_cfg.get("provider") or "Unavailable")
    fallback = hermes_cfg.get("fallback_provider")
    voice_provider = str(voice_cfg.get("provider") or "").strip().lower()
    with _RUNTIME_STATUS_LOCK:
        runtime = dict(_LAST_HERMES_RUNTIME)
    lock_confirmed = (
        runtime.get("model_lock") == "confirmed"
        and runtime.get("model") == model
        and runtime.get("provider") == provider
    )
    return JSONResponse({
        "brain": model,
        "provider": provider,
        "stt": str(stt_cfg.get("model") or "Unavailable"),
        "tts": "macOS Local" if voice_provider in ("macos", "macos-say", "say", "local") else "Unavailable",
        "fallback": "None" if fallback is None else str(fallback),
        "model_lock": "confirmed" if lock_confirmed else "requested",
    })


WS_CLIENTS: set = set()


@app.post("/api/summon")
async def summon(request: Request) -> JSONResponse:
    """Broadcast a holographic media panel to all connected HUD clients.

    Body: {"media": "video"|"iframe"|"image", "src": "...", "title": "...",
           "position": "center"|"left"|"right"}  or  {"action": "dismiss"}
    Hermes can call this (curl with X-Jarvis-Token) to display media on the HUD.
    """
    body = await request.json()
    if body.get("action") == "dismiss":
        payload = {"type": "dismiss_panels"}
    else:
        payload = {"type": "summon_panel",
                   "media": body.get("media") or body.get("type") or "iframe",
                   "src": body.get("src", ""),
                   "title": body.get("title", "INCOMING FEED"),
                   "position": body.get("position", "center")}
    sent = 0
    for client in list(WS_CLIENTS):
        try:
            await client.send_json(payload)
            sent += 1
        except Exception:
            WS_CLIENTS.discard(client)
    return JSONResponse({"sent_to": sent})


_WORKER_CACHE: dict = {"ts": 0.0, "data": [], "refreshing": False}


@app.get("/api/machines")
async def machines() -> JSONResponse:
    """Local (Mac) stats + configured remote workers.

    Worker polls can take seconds when a worker is offline, so they run in a
    background refresh; the endpoint always answers instantly from cache.
    """
    result: list[dict] = []
    mac: dict = {"name": "MAC MINI · HERMES", "online": True}
    if psutil:
        mac.update({
            "cpu": psutil.cpu_percent(interval=0.1),
            "mem": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage(str(ROOT)).percent,
        })
    result.append(mac)

    def poll_worker(w: dict) -> dict:
        info = {"name": w.get("name", w.get("host", "worker")), "online": False}
        url = w.get("stats_url")
        if url:
            try:
                r = requests.get(url, timeout=2)
                if r.ok:
                    info.update(r.json())
                    info["online"] = True
                    return info
            except Exception:
                pass
        import socket
        try:
            with socket.create_connection((w.get("host"), int(w.get("ping_port", 445))), timeout=1.5):
                info["online"] = True
                info["note"] = "online (no stats agent)"
        except Exception:
            pass
        return info

    workers = CFG.get("machines") or []
    now = time.time()
    if workers and now - _WORKER_CACHE["ts"] > 10 and not _WORKER_CACHE["refreshing"]:
        _WORKER_CACHE["refreshing"] = True

        async def refresh() -> None:
            try:
                data = [await asyncio.to_thread(poll_worker, w) for w in workers]
                _WORKER_CACHE.update(ts=time.time(), data=data)
            finally:
                _WORKER_CACHE["refreshing"] = False

        asyncio.get_running_loop().create_task(refresh())
    result.extend(_WORKER_CACHE["data"] or
                  [{"name": w.get("name", "worker"), "online": False, "note": "checking..."} for w in workers])
    return JSONResponse({"machines": result})


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/hud/")


if HUD_DIR.exists():
    app.mount("/hud", StaticFiles(directory=str(HUD_DIR), html=True), name="hud")


# ------------------------------------------------------------ Zod's Universe
# Strictly additive. /hud/ stays the accepted Operator and recovery interface
# and "/" still redirects there. The Universe owns no conversation or voice
# backend of its own: it drives the same /ws channel and the same /api/chat
# operator turn the HUD does, so there is exactly one proven runtime.
UNIVERSE_V2_DIR = ROOT / "universe-v2"
try:
    import universe_api

    universe_api.configure(CFG)
    app.include_router(universe_api.router)
    if UNIVERSE_V2_DIR.exists():
        app.mount("/universe-v2", StaticFiles(directory=str(UNIVERSE_V2_DIR), html=True),
                  name="universe-v2")
    print("Zod's Universe (v2) available on /universe-v2/", flush=True)
except Exception as exc:  # the Universe must never take the Operator down
    print(f"Universe interface unavailable: {type(exc).__name__}: {exc}", flush=True)


# ----------------------------------------------- Hermes dashboard TLS proxy
# The HUD (https) cannot iframe the plain-http dashboard (mixed content), so
# this second app reverse-proxies the entire dashboard over TLS, stripping
# frame-blocking headers. Served on its own port (see server.dashboard_proxy).

dash_app = FastAPI(title="Hermes Dashboard TLS Proxy")
_STRIP_HEADERS = {"x-frame-options", "content-security-policy", "content-length",
                  "transfer-encoding", "connection", "content-encoding"}


@dash_app.middleware("http")
async def dash_auth_middleware(request: Request, call_next):
    if not _request_authed(request):
        return Response(status_code=401, content="jarvis auth required")
    return await call_next(request)


def _dash_target() -> str:
    return ((CFG.get("server") or {}).get("dashboard_proxy") or {}).get(
        "target", "http://127.0.0.1:9119").rstrip("/")


@dash_app.websocket("/{path:path}")
async def dash_ws_proxy(ws: WebSocket, path: str) -> None:
    import websockets as wslib
    token = hud_token()
    if token and ws.cookies.get("jarvis_token") != token:
        await ws.close(code=4401)
        return
    await ws.accept()
    target = _dash_target().replace("http://", "ws://").replace("https://", "wss://")
    uri = f"{target}/{path}" + (f"?{ws.url.query}" if ws.url.query else "")
    try:
        async with wslib.connect(uri, max_size=None) as backend:
            async def client_to_backend() -> None:
                while True:
                    m = await ws.receive()
                    if m.get("text") is not None:
                        await backend.send(m["text"])
                    elif m.get("bytes") is not None:
                        await backend.send(m["bytes"])
                    elif m.get("type") == "websocket.disconnect":
                        break

            async def backend_to_client() -> None:
                async for m in backend:
                    if isinstance(m, str):
                        await ws.send_text(m)
                    else:
                        await ws.send_bytes(m)

            done, pending_t = await asyncio.wait(
                [asyncio.create_task(client_to_backend()),
                 asyncio.create_task(backend_to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending_t:
                t.cancel()
    except Exception:
        pass


@dash_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def dash_http_proxy(path: str, request: Request) -> Response:
    body = await request.body()
    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "accept-encoding", "connection")}

    def do_request() -> requests.Response:
        return requests.request(
            request.method, f"{_dash_target()}/{path}",
            params=dict(request.query_params), headers=fwd_headers,
            data=body if body else None, timeout=60, allow_redirects=False,
        )

    resp = await asyncio.to_thread(do_request)
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _STRIP_HEADERS}
    return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)


# ------------------------------------------------------------------ WebSocket


@dataclass
class ConnState:
    audio_chunks: list = field(default_factory=list)
    recording: bool = False
    timing: TurnTiming | None = None
    turn_task: asyncio.Task | None = None
    current_run_id: str | None = None
    conversation: str = "jarvis-main"
    spoken_sentences: list = field(default_factory=list)
    interrupt_note: str | None = None
    partial_task: asyncio.Task | None = None
    last_partial_bytes: int = 0


async def _run_turn(ws: WebSocket, pipeline: VoicePipelineServer, conn: ConnState) -> None:
    timing = conn.timing
    assert timing is not None
    audio = b"".join(conn.audio_chunks)
    conn.audio_chunks = []
    try:
        transcript = await pipeline.transcribe(audio, timing)
        timing.transcript = transcript
        await ws.send_json({"type": "transcript", "text": transcript})
        if not transcript:
            await ws.send_json({"type": "error", "message": "No transcript detected."})
        else:
            if conn.interrupt_note:
                transcript_sent = (
                    f"[note: your previous spoken reply was cut off by the user after you said: "
                    f"\"{conn.interrupt_note}\"]\n{transcript}"
                )
                conn.interrupt_note = None
            else:
                transcript_sent = transcript
            conn.spoken_sentences = []
            await pipeline.stream_response_audio(ws, transcript_sent, timing, conn)
            timing.total_done_monotonic = time.perf_counter()
            await ws.send_json({"type": "done", "turn_id": timing.turn_id, "timing": timing.summary()})
    except asyncio.CancelledError:
        timing.errors.append("turn cancelled (barge-in or stop)")
        raise
    except Exception as exc:
        timing.errors.append(f"{type(exc).__name__}: {exc}")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        timing.total_done_monotonic = timing.total_done_monotonic or time.perf_counter()
        pipeline.log_turn(timing)
        conn.timing = None
        conn.current_run_id = None


async def _cancel_active_turn(ws: WebSocket, pipeline: VoicePipelineServer, conn: ConnState,
                              stop_remote: bool = True) -> None:
    run_id = conn.current_run_id  # capture BEFORE cancel: turn cleanup clears it
    turn_was_active = conn.turn_task is not None and not conn.turn_task.done()
    if turn_was_active:
        if conn.spoken_sentences:
            conn.interrupt_note = conn.spoken_sentences[-1]
        conn.turn_task.cancel()
        try:
            await conn.turn_task
        except (asyncio.CancelledError, Exception):
            pass
    if stop_remote and run_id and turn_was_active:
        conn.current_run_id = None
        try:
            res = await asyncio.to_thread(pipeline.hermes.stop_run, run_id)
            # 404 = session runs not in the runs registry on this Hermes build;
            # dropping the SSE stream (above) still cuts the turn off.
            msg = "Run halted." if res["status_code"] in (200, 202, 404) else f"Stop returned {res['status_code']}."
            await ws.send_json({"type": "status", "message": msg})
        except Exception as exc:
            await ws.send_json({"type": "status", "message": f"Stop failed: {exc}"})


def _maybe_schedule_partial(ws: WebSocket, pipeline: VoicePipelineServer, conn: ConnState) -> None:
    stt_cfg = CFG.get("stt") or {}
    if not stt_cfg.get("partials", True) or not conn.recording:
        return
    if conn.partial_task and not conn.partial_task.done():
        return
    buf = b"".join(conn.audio_chunks)
    min_new = int(16000 * 2 * float(stt_cfg.get("partial_interval", 1.2)))
    if len(buf) < 16000 or len(buf) - conn.last_partial_bytes < min_new or len(buf) > 16000 * 2 * 30:
        return
    conn.last_partial_bytes = len(buf)

    async def run() -> None:
        try:
            text = await pipeline.transcribe(buf)
            if text and conn.recording:
                await ws.send_json({"type": "partial_transcript", "text": text})
        except Exception:
            pass

    conn.partial_task = asyncio.create_task(run())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    if not _ws_allowed(ws):
        await ws.close(code=4401)
        return
    await ws.accept()
    WS_CLIENTS.add(ws)
    channel = OPERATOR_BRIDGE.open_channel(ws)
    pipeline = get_pipeline()
    conn = ConnState(conversation=(CFG.get("hermes") or {}).get("conversation", "jarvis-main"))
    await ws.send_json({"type": "status", "message": "Hermes voice server connected."})
    await ws.send_json({"type": "operator_ready", "channel_id": channel.channel_id})
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if "text" in message and message["text"] is not None:
                try:
                    event = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid WebSocket JSON."})
                    continue
                if not isinstance(event, dict):
                    await ws.send_json({"type": "error", "message": "Invalid WebSocket event."})
                    continue
                etype = event.get("type")
                if etype == "start":
                    await _cancel_active_turn(ws, pipeline, conn)  # barge-in
                    if event.get("conversation"):
                        conn.conversation = str(event["conversation"])
                    conn.audio_chunks = []
                    conn.last_partial_bytes = 0
                    conn.recording = True
                    conn.timing = TurnTiming(turn_id=pipeline.next_turn_id())
                    conn.timing.audio_start_monotonic = time.perf_counter()
                    conn.timing.stt_model = CFG["stt"]["model"]
                    await ws.send_json({"type": "status", "message": f"Turn {conn.timing.turn_id} recording started."})
                elif etype == "stop":
                    if conn.timing is None:
                        await ws.send_json({"type": "error", "message": "Received stop before start."})
                        continue
                    conn.recording = False
                    conn.timing.end_of_speech_monotonic = time.perf_counter()
                    conn.turn_task = asyncio.create_task(_run_turn(ws, pipeline, conn))
                elif etype == "stop_run":
                    await _cancel_active_turn(ws, pipeline, conn)
                    await ws.send_json({"type": "agent_status", "state": "stopped"})
                elif etype == "stop_operator_run":
                    if await OPERATOR_BRIDGE.stop_channel_runs(channel.channel_id):
                        await ws.send_json({"type": "agent_status", "state": "stopped"})
                elif etype == "approval_decision":
                    result = await OPERATOR_BRIDGE.decide(
                        channel.channel_id, event.get("approval_token"), event.get("decision"),
                    )
                    if not result["accepted"]:
                        token = event.get("approval_token")
                        await ws.send_json({
                            "type": "approval_rejected",
                            "approval_token": token if isinstance(token, str) else "",
                            "message": result["reason"],
                        })
                else:
                    await ws.send_json({"type": "error", "message": f"Unknown event type: {etype}"})
            elif "bytes" in message and message["bytes"] is not None:
                if conn.recording:
                    conn.audio_chunks.append(message["bytes"])
                    _maybe_schedule_partial(ws, pipeline, conn)
    except WebSocketDisconnect:
        if conn.turn_task and not conn.turn_task.done():
            conn.turn_task.cancel()
        print("Client disconnected", flush=True)
    finally:
        await OPERATOR_BRIDGE.close_channel(channel.channel_id)
        WS_CLIENTS.discard(ws)


def main() -> int:
    server = CFG["server"]
    host = server.get("host", "0.0.0.0")
    port = int(server.get("port", 8765))
    # Fix localhost specifically to be 127.0.0.1
    if host == "localhost":
        host = "127.0.0.1"
    tls_ports = server.get("tls_ports") or ([server["tls_port"]] if server.get("tls_port") else [])
    cert = server.get("tls_cert")
    key = server.get("tls_key")
    print(f"Starting Hermes voice server on ws://{host}:{port}/ws", flush=True)
    if tls_ports and cert and key and (ROOT / cert).exists() and (ROOT / key).exists():
        servers = [uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))]
        for tp in tls_ports:
            print(f"HUD available on https://{host}:{tp}/hud/", flush=True)
            servers.append(uvicorn.Server(uvicorn.Config(
                app, host=host, port=int(tp), log_level="info",
                ssl_certfile=str(ROOT / cert), ssl_keyfile=str(ROOT / key),
            )))

        dp = server.get("dashboard_proxy") or {}
        if dp.get("port"):
            print(f"Dashboard proxy on https://{host}:{dp['port']}/", flush=True)
            servers.append(uvicorn.Server(uvicorn.Config(
                dash_app, host=host, port=int(dp["port"]), log_level="warning",
                ssl_certfile=str(ROOT / cert), ssl_keyfile=str(ROOT / key),
            )))

        async def serve_all() -> None:
            await asyncio.gather(*[s.serve() for s in servers])

        asyncio.run(serve_all())
    else:
        uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
