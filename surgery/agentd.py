"""Run pi headless for one run and let other terminals talk to it.

    python -m surgery.agentd <run_dir> [--task FILE] -- pi --mode rpc ...

pi runs in RPC mode as a child process. This daemon

- saves pi's event stream to <run>/events.jsonl,
- forwards commands that bin/watch appends to <run>/inbox.jsonl:
      {"type": "prompt", "message": "..."}   sent as steering if the model is mid-run
      {"type": "abort"}                      stop the current turn
- reports what it did in <run>/agent.jsonl, and its pid in <run>/agent.pid.

It keeps the session open after the model finishes, so you can go on talking
to the (possibly self-edited) model. Stop it with Ctrl-C or SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .monitor import Tail


class Daemon:
    def __init__(self, run_dir: Path, pi_cmd: list[str]):
        self.run = run_dir
        self.running = False  # is the agent mid-run?
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.n = 0
        self.log = (run_dir / "agent.jsonl").open("a", buffering=1)
        self.events = (run_dir / "events.jsonl").open("ab", buffering=0)
        (run_dir / "workspace").mkdir(exist_ok=True)
        self.pi = subprocess.Popen(
            pi_cmd,
            cwd=run_dir / "workspace",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(run_dir / "pi.stderr.log").open("ab"),
        )
        (run_dir / "agent.pid").write_text(str(os.getpid()))
        self.emit({"t": "daemon_start", "pid": os.getpid(), "pi_pid": self.pi.pid})

    def emit(self, row: dict):
        with self.lock:
            self.log.write(json.dumps({"time": round(time.time(), 3), **row}, ensure_ascii=False) + "\n")

    def send(self, cmd: dict) -> str:
        self.n += 1
        cmd = {"id": f"d{self.n}", **cmd}
        try:
            self.pi.stdin.write(json.dumps(cmd, ensure_ascii=False).encode() + b"\n")
            self.pi.stdin.flush()
        except (BrokenPipeError, OSError):
            self.emit({"t": "error", "error": "pi is not running"})
        return cmd["id"]

    def prompt(self, message: str, source: str = "user"):
        message = message.strip()
        if not message:
            return
        steer = self.running
        cmd = {"type": "prompt", "message": message}
        if steer:
            cmd["streamingBehavior"] = "steer"
        self.emit({"t": "sent", "message": message, "steer": steer, "source": source, "id": self.send(cmd)})

    def read_pi(self):
        """pi's stdout: save every record, track whether the agent is running."""
        for line in self.pi.stdout:  # binary: split on LF only, as the protocol requires
            self.events.write(line)
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = r.get("type")
            if t == "agent_start" and not self.running:
                self.running = True
                self.emit({"t": "state", "running": True})
            elif t == "agent_settled":
                self.running = False
                self.emit({"t": "state", "running": False, "aborted": r.get("aborted", False)})
            elif t == "response" and not r.get("success", True):
                self.emit({"t": "error", "command": r.get("command"), "error": r.get("error")})
            elif t == "message_end" and (r.get("message") or {}).get("stopReason") == "error":
                self.emit({"t": "error", "error": r["message"].get("errorMessage") or "the model request failed"})
            elif t == "auto_retry_start":
                self.emit({"t": "retry", "attempt": r.get("attempt")})
            elif t == "extension_error":
                self.emit({"t": "error", "error": str(r.get("error") or r)[:500]})
        self.stop.set()

    def serve(self, task: str | None):
        threading.Thread(target=self.read_pi, daemon=True).start()
        inbox = Tail(self.run / "inbox.jsonl")
        try:
            inbox.pos = inbox.path.stat().st_size  # ignore messages from before this daemon
        except OSError:
            pass
        if task:
            self.prompt(task, source="task")
        while not self.stop.wait(0.2):
            for cmd in inbox.read():
                if cmd.get("type") == "prompt":
                    self.prompt(str(cmd.get("message", "")))
                elif cmd.get("type") == "abort":
                    self.emit({"t": "abort", "id": self.send({"type": "abort"})})

    def close(self):
        try:
            self.pi.stdin.close()  # asks pi for an orderly shutdown
        except OSError:
            pass
        try:
            code = self.pi.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.pi.terminate()
            code = self.pi.wait()
        self.emit({"t": "exit", "code": code})
        try:
            (self.run / "agent.pid").unlink()
        except OSError:
            pass


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: python -m surgery.agentd <run_dir> [--task FILE] -- <pi command>")
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--task", type=Path, help="send this file as the first message")
    a = ap.parse_args(argv[:split])

    d = Daemon(a.run_dir.resolve(), argv[split + 1:])
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: d.stop.set())
    print(f"pi is running headless (pid {d.pi.pid}). Watch and talk to it with: bin/watch", file=sys.stderr)
    try:
        d.serve(a.task.read_text() if a.task else None)
    finally:
        d.close()


if __name__ == "__main__":
    main()
