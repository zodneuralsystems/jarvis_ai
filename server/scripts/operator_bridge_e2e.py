#!/usr/bin/env python3
"""Runtime proof for the fail-closed HUD/Hermes operator approval bridge.

Run with JARVIS_HUD_TOKEN exported. It never prints that token and only creates
disposable SQLite fixtures below this repository.
"""

import asyncio
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import requests
import websockets


API_URL = "http://127.0.0.1:8765/api/chat"
WS_URL = "ws://127.0.0.1:8765/ws"
CONVERSATION = "zod-operator-bridge-e2e"
TIMEOUT = 300
ROOT = Path(__file__).resolve().parents[2]


def _post_chat(text: str, channel_id: str) -> dict:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("JARVIS_HUD_TOKEN")
    if not token:
        raise RuntimeError("JARVIS_HUD_TOKEN is required for the authenticated HUD API test")
    headers["X-Jarvis-Token"] = token
    response = requests.post(
        API_URL,
        headers=headers,
        json={"input": text, "conversation": CONVERSATION, "channel_id": channel_id},
        timeout=TIMEOUT + 30,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"error": f"HTTP {response.status_code} returned non-JSON"}
    if response.status_code != 200:
        raise RuntimeError(body.get("error") or f"HUD API HTTP {response.status_code}")
    return body


async def _ready(ws) -> str:
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=20)
        if isinstance(raw, bytes):
            continue
        event = json.loads(raw)
        if event.get("type") == "operator_ready" and isinstance(event.get("channel_id"), str):
            return event["channel_id"]


def _fixture_db() -> Path:
    path = ROOT / f".operator-approval-fixture-{uuid.uuid4().hex}.db"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        database.execute("INSERT INTO items (value) VALUES ('safe test row')")
    return path


def _row_count(path: Path) -> int:
    with sqlite3.connect(path) as database:
        return int(database.execute("SELECT COUNT(*) FROM items").fetchone()[0])


def _cleanup_fixture(path: Path) -> None:
    if path.parent != ROOT or not path.name.startswith(".operator-approval-fixture-"):
        raise RuntimeError("refusing to clean a non-test fixture")
    if path.exists():
        path.unlink()


async def _run(ws, channel_id: str, prompt: str, approval_handler=None) -> tuple[dict, dict]:
    post_task = asyncio.create_task(asyncio.to_thread(_post_chat, prompt, channel_id))
    saw_tool = False
    tool_names = []
    approval_event = None
    terminal_event = None
    rejected = False
    while terminal_event is None:
        raw = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT)
        if isinstance(raw, bytes):
            continue
        event = json.loads(raw)
        event_type = event.get("type")
        if event_type == "agent_status" and event.get("state") == "tool_use":
            saw_tool = True
            tool_names.append(str(event.get("tool") or "tool"))
        elif event_type == "approval_request":
            if approval_event is not None:
                raise RuntimeError("received more than one pending approval for one test action")
            approval_event = event
            if approval_handler is not None:
                await approval_handler(ws, event)
        elif event_type == "approval_rejected":
            rejected = True
        elif event_type == "operator_run_terminal":
            terminal_event = event
    result = await post_task
    result["_saw_tool"] = saw_tool
    result["_tool_names"] = tool_names
    result["_approval_event"] = approval_event
    result["_approval_rejected"] = rejected
    return result, terminal_event


def _terminal_prompt(command: str) -> str:
    return (
        "Use the terminal tool exactly once. Run exactly this command and no other command: "
        f"{command}\n"
        "This is a disposable Zod HUD operator approval test fixture. "
        "Wait for the tool policy if required, then report the result."
    )


async def _normal_tool(ws, channel_id: str) -> None:
    result, terminal = await _run(
        ws,
        channel_id,
        "Use the terminal tool exactly once. Run exactly: pwd. Report only the command output.",
    )
    if not result["_saw_tool"] or result["_approval_event"] is not None:
        raise RuntimeError("pwd did not produce one normal tool path without approval")
    if result.get("state") != "completed" or not result.get("text"):
        raise RuntimeError("pwd run did not complete with output")
    if terminal.get("state") != "completed":
        raise RuntimeError("HUD did not report pwd completion")
    print("NORMAL_TOOL=PASS")


async def _typed_chat(ws, channel_id: str) -> None:
    result, terminal = await _run(
        ws,
        channel_id,
        "Reply with exactly: HUD CHAT OK. Do not use tools.",
    )
    if result["_saw_tool"] or result.get("state") != "completed" or not result.get("text"):
        raise RuntimeError("plain typed chat did not complete normally")
    if terminal.get("state") != "completed":
        raise RuntimeError("HUD did not report typed chat completion")
    print("HUD_TYPED_CHAT=PASS")


async def _approval_case(ws, channel_id: str, decision: str, stale_first: bool = False) -> None:
    fixture = _fixture_db()
    try:
        command = f'sqlite3 {fixture} "DELETE FROM items"'

        async def decide(ws_conn, event: dict) -> None:
            if _row_count(fixture) != 1:
                raise RuntimeError("fixture changed before an approval decision")
            summary = event.get("summary") or {}
            if "DELETE FROM items" not in str(summary.get("action") or ""):
                raise RuntimeError("approval summary did not describe the requested terminal action")
            if decision == "once" and event.get("can_approve_once") is not True:
                raise RuntimeError("terminal approval was not eligible for approve-once")
            if stale_first:
                await ws_conn.send(json.dumps({
                    "type": "approval_decision",
                    "approval_token": "stale-" + event["approval_token"],
                    "decision": "once",
                }))
                # The real prompt remains pending. The next valid answer must
                # still be a denial so this disposable data survives.
                await asyncio.sleep(0.2)
                if _row_count(fixture) != 1:
                    raise RuntimeError("stale approval changed the fixture")
                await ws_conn.send(json.dumps({
                    "type": "approval_decision",
                    "approval_token": event["approval_token"],
                    "decision": "deny",
                }))
            else:
                await ws_conn.send(json.dumps({
                    "type": "approval_decision",
                    "approval_token": event["approval_token"],
                    "decision": decision,
                }))

        result, terminal = await _run(ws, channel_id, _terminal_prompt(command), decide)
        if result["_approval_event"] is None:
            raise RuntimeError(
                "Hermes did not generate an approval request "
                f"(state={result.get('state')}, tools={result['_tool_names']}, "
                f"approval_state={result.get('approval_state')}, rows={_row_count(fixture)})"
            )
        if terminal.get("state") != "completed":
            raise RuntimeError("Hermes run did not reach a real terminal completion")
        if stale_first:
            if not result["_approval_rejected"] or _row_count(fixture) != 1:
                raise RuntimeError("stale approval was not rejected fail-closed")
            if result.get("approval_state") != "denied":
                raise RuntimeError("real denial was not reported after stale rejection")
            print("FAIL_CLOSED=PASS")
        elif decision == "deny":
            if _row_count(fixture) != 1 or result.get("approval_state") != "denied":
                raise RuntimeError("denied action executed or denial was not reported")
            print("APPROVAL_GENERATED=PASS")
            print("DENY=PASS")
        else:
            if _row_count(fixture) != 0 or result.get("approval_state") != "approved_once":
                raise RuntimeError("approve-once did not execute exactly the disposable action")
            print("APPROVE_ONCE=PASS")
    finally:
        _cleanup_fixture(fixture)


async def main() -> None:
    async with websockets.connect(WS_URL, max_size=None) as ws:
        channel_id = await _ready(ws)
        await _typed_chat(ws, channel_id)
        await _normal_tool(ws, channel_id)
        await _approval_case(ws, channel_id, "deny")
        await _approval_case(ws, channel_id, "once")
        await _approval_case(ws, channel_id, "deny", stale_first=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"OPERATOR_BRIDGE_E2E=FAIL: {exc}")
        raise SystemExit(1)
