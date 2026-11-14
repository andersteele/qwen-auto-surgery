import json
import sys
import threading
import time
from pathlib import Path

from surgery.agentd import Daemon
from surgery.monitor import Run

FAKE_PI = [sys.executable, str(Path(__file__).parent / "fake_pi.py")]


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_daemon_relays_inbox_to_pi_and_tracks_state(tmp_path):
    (tmp_path / "inbox.jsonl").write_text(json.dumps({"type": "prompt", "message": "stale, from an earlier daemon"}) + "\n")
    d = Daemon(tmp_path, FAKE_PI)
    t = threading.Thread(target=d.serve, args=("long",))
    t.start()
    watcher = Run(tmp_path)
    try:
        assert watcher.agent_alive()
        assert wait_for(lambda: d.running), "the task prompt should start a run"

        assert watcher.send({"type": "prompt", "message": "also this"})  # mid-run: must be queued as steering
        assert wait_for(lambda: len([r for r in rows(tmp_path / "agent.jsonl") if r["t"] == "sent"]) == 2)
        assert watcher.send({"type": "abort"})
        assert wait_for(lambda: not d.running)
    finally:
        d.stop.set()
        t.join()
        d.close()

    sent = [json.loads(l)["cmd"] for l in (tmp_path / "events.jsonl").read_text().splitlines() if '"echo"' in l]
    assert [(c["type"], c.get("message"), c.get("streamingBehavior")) for c in sent] == [
        ("prompt", "long", None),
        ("prompt", "also this", "steer"),
        ("abort", None, None),
    ]
    log = rows(tmp_path / "agent.jsonl")
    assert [r["t"] for r in log] == ["daemon_start", "sent", "state", "sent", "error", "abort", "error", "state", "exit"]
    assert log[1]["source"] == "task" and log[3]["steer"] is True and log[7]["aborted"] is True
    assert log[6]["error"] == "404 generation thread died"  # pi's own failures are passed on
    assert not (tmp_path / "agent.pid").exists() and not watcher.agent_alive()
    assert not watcher.send({"type": "prompt", "message": "nobody home"})

    lines, _ = watcher.poll()
    text = "\n".join(l.plain for l in lines)
    assert "task prompt → model (sent)" in text and "you → model (queued" in text and "turn interrupted" in text
    assert "pi: 404 generation thread died" in text
    assert watcher.state.agent == "exited"
