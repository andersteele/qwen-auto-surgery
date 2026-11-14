"""Put a recorded conversation to the weights the server has now.

    python -m surgery.replay <run> --cut "words of a user message" [-n 10] [--ask "another message"]

A reply inside a long working conversation depends on the weights and on the conversation.
This holds the conversation fixed: it takes <run>'s pi session, cuts it off just before the
user message that contains the --cut text, and sends everything before it, followed by that
message (or --ask), to the model server that is up now, n times. Run it once on the
original weights and once with the edits in place and compare.

The request is built by pi itself, from a copy of the session, against a stand-in server
that only records it. pi fills in today's system prompt and tool definitions, so those two
are then put back to what the session recorded at the time. When nothing but the weights
differs, the prompt has the same number of tokens as the original request; both counts
are printed.

Tool calls in a sampled reply are not carried out. Each sample is one reply and no more.
Samples are appended to runs/latest/replay.jsonl, and the server logs each generation in
its own run directory as usual.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def text_of(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    return "".join(p.get("text") or "" for p in c or [] if isinstance(p, dict))


def cut_index(rows: list[dict], needle: str) -> int:
    """Index of the one user message that contains `needle`."""
    hits = [
        i for i, r in enumerate(rows)
        if r.get("type") == "message" and (r.get("message") or {}).get("role") == "user" and needle in text_of(r["message"])
    ]
    if len(hits) != 1:
        found = "; ".join(repr(text_of(rows[i]["message"])[:60]) for i in hits) or "none"
        raise SystemExit(f"--cut must match exactly one user message; {len(hits)} match ({found})")
    return hits[0]


def recorded_state(rows: list[dict]) -> tuple[dict, list[dict]]:
    """The system prompt sections and tool definitions in force after `rows`.

    pi writes a system entry when either changes: its sections replace the earlier ones by
    name, the tools it removes are dropped, and the tools it adds go to the end of the list.
    """
    sections: dict = {}
    tools: list[dict] = []
    for r in rows:
        m = r.get("message") or {}
        if r.get("type") != "message" or m.get("role") != "system":
            continue
        sections.update(m.get("sections") or {})
        gone = {t["name"] for t in m.get("toolsRemoved") or []} | {t["name"] for t in m.get("toolsAdded") or []}
        tools = [t for t in tools if t["name"] not in gone] + list(m.get("toolsAdded") or [])
    return sections, tools


def assemble(sections: dict) -> str:
    return "\n\n".join(sections.values())


def restore(body: dict, before: list[dict], after: list[dict]) -> tuple[dict, str]:
    """`body` as pi built it today, with the system prompt and tools the session recorded.

    `before` is the session up to the cut; `after` adds whatever pi appended while building
    the request. The second value says what was done, for the log.
    """
    then_sections, then_tools = recorded_state(before)
    now_sections, now_tools = recorded_state(after)
    wire = {t["function"]["name"]: t for t in body.get("tools") or []}
    system = body["messages"][0]
    understood = (
        system.get("role") == "system"
        and isinstance(system.get("content"), str)
        and assemble(now_sections) == system["content"]
        and [t["name"] for t in now_tools] == list(wire)
        and all(wire[t["name"]]["function"].get("description") == t.get("description") for t in now_tools)
    )
    if not understood:
        return body, "could not match pi's request to its session record; sent as pi built it (today's system prompt and tools)"
    out = copy.deepcopy(body)
    out["messages"][0]["content"] = assemble(then_sections)
    out["tools"] = []
    for t in then_tools:
        w = copy.deepcopy(wire.get(t["name"]) or {"type": "function", "function": {"name": t["name"]}})
        w["function"]["description"] = t.get("description")
        w["function"]["parameters"] = t.get("parameters")
        out["tools"].append(w)
    same = out["messages"][0]["content"] == system["content"] and out["tools"] == body.get("tools")
    return out, "system prompt and tools are unchanged since the run" if same else "system prompt and tools put back to the ones the run recorded"


class _Recorder(BaseHTTPRequestHandler):
    """Stands in for the model server while pi builds its request: keeps the body, answers with a stub."""

    bodies: list[bytes] = []

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._json({"ok": True, "text": "", "object": "list", "data": []})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        if not self.path.endswith("/chat/completions"):
            return self._json({"ok": True, "text": ""})
        self.bodies.append(body)
        req = json.loads(body)
        base = {"id": "stub", "object": "chat.completion.chunk", "created": int(time.time()), "model": req.get("model")}
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if not req.get("stream"):
            return self._json({**base, "object": "chat.completion", "usage": usage,
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": "."}, "finish_reason": "stop"}]})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for chunk in (
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "."}, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
        ):
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def build_request(run: str, needle: str, ask: str | None) -> tuple[dict, dict]:
    """The request for <run>'s conversation up to the cut, and facts about it for the log."""
    sessions = sorted((ROOT / "runs" / run / "sessions").glob("*.jsonl"))
    if not sessions:
        raise SystemExit(f"no pi session in runs/{run}/sessions")
    lines = [ln for ln in sessions[-1].read_text().split("\n") if ln]
    rows = [json.loads(ln) for ln in lines]
    cut = cut_index(rows, needle)
    message = ask if ask is not None else text_of(rows[cut]["message"])
    usage = ((rows[cut + 1].get("message") or {}).get("usage") or {}) if cut + 1 < len(rows) else {}
    original_tokens = (usage.get("input") or 0) + (usage.get("cacheRead") or 0) or None

    tmp = Path(tempfile.mkdtemp(prefix="surgery-replay-")).resolve()  # pi matches a session to its real working directory
    try:
        (tmp / "workspace").mkdir()
        (tmp / "sessions").mkdir()
        shutil.copytree(ROOT / "pi" / "agent", tmp / "agent")
        head = {**rows[0], "cwd": str(tmp / "workspace")}  # pi continues the session that belongs to its working directory
        copy_path = tmp / "sessions" / sessions[-1].name
        copy_path.write_text("\n".join([json.dumps(head, ensure_ascii=False)] + lines[1:cut]) + "\n")

        _Recorder.bodies = []
        stub = HTTPServer(("127.0.0.1", 0), _Recorder)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{stub.server_address[1]}"
        settings = json.loads((tmp / "agent" / "settings.json").read_text())
        models = json.loads((tmp / "agent" / "models.json").read_text())
        models["providers"][settings["defaultProvider"]]["baseUrl"] = f"{url}/v1"
        (tmp / "agent" / "models.json").write_text(json.dumps(models))
        env = {**os.environ, "PI_CODING_AGENT_DIR": str(tmp / "agent"), "PI_OFFLINE": "1",
               "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0", "SURGERY_URL": url}
        pi = [str(ROOT / "node_modules" / ".bin" / "pi"), "--model", f"{settings['defaultProvider']}/{settings['defaultModel']}",
              "--no-builtin-tools", "--no-context-files", "--no-skills", "--no-prompt-templates",
              "--session-dir", str(tmp / "sessions"), "--continue", "--mode", "json", message]
        done = subprocess.run(pi, cwd=tmp / "workspace", env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
        stub.shutdown()
        if not _Recorder.bodies:
            raise SystemExit(f"pi sent no request (exit {done.returncode}): {done.stderr.decode(errors='replace')[-400:]}")
        body = json.loads(_Recorder.bodies[0])
        after = [json.loads(ln) for ln in copy_path.read_text().split("\n") if ln]
        body, note = restore(body, after[:cut], after)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if text_of(body["messages"][-1]) != message or len(after) <= cut:
        raise SystemExit("the request pi built does not continue the recorded conversation; not sending it")
    if sum(m.get("role") != "system" for m in body["messages"]) < 2 <= sum(r.get("type") == "message" for r in rows[:cut]):
        raise SystemExit("pi started a new conversation instead of continuing the recorded one; not sending it")
    facts = {"run": run, "session": sessions[-1].name, "cut_entry": cut, "messages": len(body["messages"]), "message": message,
             "original_prompt_tokens": None if ask is not None else original_tokens, "note": note}
    return body, facts


def sample(body: dict, port: int) -> dict:
    """One reply from the server that is up now."""
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps({**body, "stream": True, "stream_options": {"include_usage": True}}).encode(),
                                 headers={"content-type": "application/json"})
    reasoning, content, calls, usage, finish = [], [], [], None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for c in chunk.get("choices") or []:
                d = c.get("delta") or {}
                reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
                content.append(d.get("content") or "")
                calls += [(tc.get("function") or {}).get("name") for tc in d.get("tool_calls") or []]
                finish = c.get("finish_reason") or finish
    return {"seconds": round(time.time() - t0, 1), "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"), "finish": finish,
            "tool_calls": [c for c in calls if c], "reasoning": "".join(reasoning), "content": "".join(content)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bin/replay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run whose conversation to replay")
    ap.add_argument("--cut", required=True, metavar="TEXT", help="words of the user message to cut the conversation at (it must be the only one containing them)")
    ap.add_argument("--ask", metavar="TEXT", help="send this in place of that message")
    ap.add_argument("-n", type=int, default=5, help="how many samples (default 5)")
    ap.add_argument("--temperature", type=float, help="override the temperature pi sends (0 is deterministic)")
    ap.add_argument("--max-tokens", type=int, default=2500, help="cap on each reply (default 2500)")
    ap.add_argument("--label", default="", help="a name for this condition, kept with each sample")
    ap.add_argument("--out", type=Path, help="where to append the samples (default runs/latest/replay.jsonl)")
    ap.add_argument("--save-request", type=Path, metavar="FILE", help="write the request to FILE and send nothing")
    args = ap.parse_args(argv)

    body, facts = build_request(args.run, args.cut, args.ask)
    body["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature
    print(f"{facts['run']}: {facts['messages']} messages up to and including {facts['message'][:70]!r}", file=sys.stderr)
    print(f"  {facts['note']}", file=sys.stderr)
    if args.save_request:
        args.save_request.write_text(json.dumps(body, ensure_ascii=False))
        print(f"  request written to {args.save_request}; nothing sent", file=sys.stderr)
        return 0

    port = int(os.environ.get("SURGERY_PORT", "8642"))
    out = args.out or ROOT / "runs" / "latest" / "replay.jsonl"
    for k in range(args.n):
        try:
            s = sample(body, port)
        except OSError as e:
            print(f"no answer from the model server on port {port} (is a run up?): {e}", file=sys.stderr)
            return 1
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": args.label, "k": k, **facts,
               "temperature": body.get("temperature"), **s}
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if k == 0 and facts["original_prompt_tokens"]:
            same = "the same" if s["prompt_tokens"] == facts["original_prompt_tokens"] else "NOT the same"
            print(f"  prompt tokens: {s['prompt_tokens']} now, {facts['original_prompt_tokens']} in the original request ({same})", file=sys.stderr)
        calls = f" [tool calls, not carried out: {', '.join(s['tool_calls'])}]" if s["tool_calls"] else ""
        print(f"#{k} {s['seconds']}s{calls} {' '.join(s['content'].split())[:160]}")
    print(f"  samples appended to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
