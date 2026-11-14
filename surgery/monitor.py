"""Watch a run from another terminal.

    bin/watch                 follow runs/latest (switches when a new run starts)
    bin/watch <run-name>      follow one run
    bin/watch --plain         no TUI: print the transcript as it happens
    bin/watch <run> --dump    print the whole transcript so far, with full tool results, and exit

Watching is passive: it tails the files the surgery server writes into the run
directory (stream.jsonl, ledger.jsonl) and never talks to the model server.

Talking works when pi runs headless (bin/run, or bin/agent --daemon): what you
type is appended to the run's inbox.jsonl, and the agent daemon hands it to pi.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.text import Text

from .atlas import Atlas, fmt_neuron, parse_neuron

ROOT = Path(__file__).resolve().parent.parent
TAGS = {"<think>": "think", "</think>": "text", "<tool_call>": "call", "</tool_call>": "text"}
STYLE = {"think": "italic grey58", "text": "default", "call": "cyan"}
RESULT_LINES = 14


class Tail:
    """Incrementally read complete JSON lines from a file that is being appended to."""

    def __init__(self, path: Path):
        self.path, self.pos, self.partial = path, 0, b""

    def read(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.pos:  # truncated or replaced
            self.pos, self.partial = 0, b""
        if size == self.pos:
            return []
        with self.path.open("rb") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos += len(data)
        *lines, self.partial = (self.partial + data).split(b"\n")
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows


@dataclass
class Edit:
    seq: int
    op: str
    target: str
    detail: dict
    active: bool = True


@dataclass
class State:
    phase: str = "waiting for the server"
    turn: int = 0
    context: int = 0
    cached: int = 0
    reading: tuple[int, int] | None = None
    gen_tokens: int = 0
    total_tokens: int = 0
    gen_started: float = 0.0
    rate: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    budget_gb: float = 0.0
    started: float = 0.0
    last_event: float = 0.0  # time of the newest record from the server: its last sign of life
    edits: list[Edit] = field(default_factory=list)
    agent: str = "none"  # none | idle | working | exited
    tool: str = ""  # tool running right now
    tool_started: float = 0.0
    crashes: int = 0
    dead: str = ""  # set when the model thread has given up
    pi_error: str = ""

    @property
    def active_edits(self) -> int:
        return sum(e.active for e in self.edits)


class Run:
    """Turns a run directory's logs into transcript entries and a State."""

    def __init__(self, run_dir: Path, show_thinking: bool = True, full_results: bool = False):
        self.dir = run_dir
        self.stream, self.ledger = Tail(run_dir / "stream.jsonl"), Tail(run_dir / "ledger.jsonl")
        self.agent_log = Tail(run_dir / "agent.jsonl")
        self.state = State()
        self.show_thinking, self.full_results = show_thinking, full_results
        self.entries: list[tuple[str, Text]] = []  # (kind, line): everything, for re-rendering
        self.line = Text()  # the line being written right now
        self.line_kind = "text"
        self.mode = "text"
        self.pending = ""
        self._hidden_chars = 0
        self._last_user = None

    # -- transcript ---------------------------------------------------------

    def visible(self, kind: str) -> bool:
        return self.show_thinking or kind != "think"

    def _push(self, kind: str, line: Text, out: list):
        self.entries.append((kind, line))
        if self.visible(kind):
            out.append(line)

    def _flush(self, out: list):
        if self.line.plain or self.line_kind != "think":
            self._push(self.line_kind, self.line, out)
        self.line, self.line_kind = Text(), self.mode

    def _text(self, s: str, out: list):
        if self.mode == "call":  # keep a tool call's markup on one line
            s = s.replace("\n", " ")
            if self.line.plain.endswith(" "):
                s = s.lstrip(" ")
        parts = s.split("\n")
        for i, part in enumerate(parts):
            if part:
                if not self.line.plain:
                    self.line_kind = self.mode
                self.line.append(part, style=STYLE[self.mode])
            if i < len(parts) - 1:
                self._flush(out)

    def _feed(self, s: str, out: list):
        self.pending += s
        while self.pending:
            hits = [(self.pending.find(t), t) for t in TAGS if t in self.pending]
            if not hits:
                keep = max((k for t in TAGS for k in range(1, len(t)) if self.pending.endswith(t[:k])), default=0)
                cut = len(self.pending) - keep
                self._text(self.pending[:cut], out)
                self.pending = self.pending[cut:]
                return
            idx, tag = min(hits)
            self._text(self.pending[:idx], out)
            self.pending = self.pending[idx + len(tag):]
            if self.line.plain:
                self._flush(out)
            self.mode = TAGS[tag]
            self.line_kind = self.mode
            if tag == "<tool_call>":
                self.line.append("→ calls ", style="bold cyan")

    def _rule(self, text: str, style: str, out: list, kind: str = "meta"):
        self._push(kind, Text(text, style=style), out)

    def _handle(self, r: dict, out: list):
        st, t = self.state, r.get("t")
        st.last_event = r.get("time", time.time())
        if t == "server_start":
            edits = st.edits
            self.state = st = State(edits=edits, started=r.get("time", time.time()), last_event=st.last_event, agent=st.agent)
            st.phase = "loading the model"
            self._last_user = None
            self._rule("━━ server started: original weights ━━", "bold green", out)
        elif t == "ready":
            st.phase = "idle"
        elif t == "beat":
            pass  # only moves last_event
        elif t == "replayed":
            self._rule(f"━━ {r.get('text')} ━━", "bold green", out)
        elif t == "crash":
            st.crashes += 1
            st.phase = "idle"
            self._rule(f"━━ generation crashed: {r.get('error')}", "bold red", out)
            self._rule(f"   recovered: the model and its {r.get('active_edits', 0)} edit(s) are still loaded; "
                       "the request in flight failed and pi will retry it ━━", "bold red", out)
        elif t == "dead":
            st.dead = r.get("error") or "unknown error"
            self._rule(f"━━ the model thread has stopped for good: {st.dead} ━━", "bold red", out)
        elif t == "prompt":
            st.reading = (r["done"], r["total"])
            st.phase = "reading context"
        elif t == "tool_start":
            st.phase, st.tool, st.tool_started = "running tool", r.get("tool", "?"), st.last_event
        elif t == "start":
            st.turn, st.context, st.cached = r["id"], r.get("prompt_tokens", 0), r.get("cached_tokens", 0)
            st.gen_tokens, st.gen_started, st.phase = 0, r.get("time", time.time()), "reading context"
            st.reading, st.pi_error = (0, max(0, st.context - st.cached)), ""
            user = r.get("user")
            if user and user != self._last_user:
                self._last_user = user
                self._rule("", "default", out)
                for ln in user.splitlines() or [""]:
                    self._push("user", Text("▌ " + ln, style="bold yellow"), out)
            fresh = st.context - st.cached
            note = "all re-read by the current weights" if st.cached == 0 and st.turn > 1 else f"{fresh:,} new"
            self._rule(f"── turn {st.turn} · context {st.context:,} tokens ({note}) ──", "grey50", out)
            self.mode = "think" if r.get("state") == "reasoning" else "text"
            self.line, self.line_kind, self.pending = Text(), self.mode, ""
        elif t == "tok":
            if st.gen_tokens == 0:
                st.gen_started = st.last_event  # the wait before this was reading the prompt
            st.gen_tokens += 1
            st.total_tokens += 1
            st.phase = "generating"
            dt = st.last_event - st.gen_started
            if dt > 0.5:
                st.rate = st.gen_tokens / dt
            self._feed(r.get("s", ""), out)
        elif t == "end":
            self._feed("", out)
            if self.pending:
                self._text(self.pending, out)
                self.pending = ""
            if self.line.plain:
                self._flush(out)
            secs = r.get("seconds") or 0
            if secs > 0 and r.get("tokens"):
                st.rate = r["tokens"] / secs
            st.phase = "idle" if r.get("finish") != "tool_calls" else "between steps"
            reading = r.get("reading_seconds") or 0
            took = f"read context in {reading:.1f}s · " if reading >= 1 else ""
            self._rule(f"   {took}{r.get('tokens', 0):,} tokens in {secs:.1f}s · {st.rate:.1f} tok/s", "grey42", out)
        elif t == "tool":
            mine = r.get("source") == "operator"  # run from outside (bin/score), not by the model
            if not mine:
                st.tool_calls += 1
                st.tool_errors += not r.get("ok", True)
            st.budget_gb = r.get("budget_gb", st.budget_gb)
            st.phase, st.tool = ("idle" if mine and st.agent != "working" else "between steps"), ""
            args = json.dumps(r.get("args", {}), ensure_ascii=False)
            if len(args) > 400 and not self.full_results:
                args = args[:400] + "…"
            head = Text("⚙ ", style="bold magenta")
            if mine:
                head.append("operator ran ", style="bold yellow")
            head.append(r.get("tool", "?"), style="bold magenta")
            head.append(f" {args}", style="magenta")
            head.append(f"  {r.get('seconds', 0):.1f}s", style="grey50")
            self._push("tool", head, out)
            lines = (r.get("result") or "").splitlines() or [""]
            shown = lines if self.full_results else lines[:RESULT_LINES]
            style = "green" if r.get("ok", True) else "bold red"
            for ln in shown:
                self._push("result", Text("  │ " + ln, style=style), out)
            if len(lines) > len(shown):
                self._push("result", Text(f"  │ … {len(lines) - len(shown)} more lines (press r)", style="grey50"), out)

    def _ledger(self, r: dict):
        op = r.get("op")
        if op == "server_start":  # a new server process starts from the original weights
            for e in self.state.edits:
                e.active = False
        elif op == "undo":
            for e in self.state.edits:
                if e.seq == r.get("undid_seq"):
                    e.active = False
        elif op == "reset":
            for e in self.state.edits:
                e.active = False
        elif "seq" in r:
            detail = {k: v for k, v in r.items() if k not in ("seq", "op", "target", "time")}
            self.state.edits.append(Edit(r["seq"], op or "?", r.get("target", "?"), detail))

    def _agent(self, r: dict, out: list):
        st, t = self.state, r.get("t")
        if t == "daemon_start":
            st.agent = "idle"
        elif t == "state":
            st.agent = "working" if r.get("running") else "idle"
            if r.get("aborted"):
                self._rule("   turn interrupted", "bold red", out)
        elif t == "sent":
            how = "queued; the model sees it after its current step" if r.get("steer") else "sent"
            who = "task prompt" if r.get("source") == "task" else "you"
            self._rule(f"   {who} → model ({how})", "yellow", out)
        elif t == "abort":
            self._rule("   interrupt requested", "bold red", out)
        elif t == "error":
            st.pi_error = str(r.get("error"))
            self._rule(f"   pi: {st.pi_error}", "bold red", out)
        elif t == "retry":
            self._rule(f"   pi retries (attempt {r.get('attempt')})", "yellow", out)
        elif t == "exit":
            st.agent = "exited"
            self._rule(f"━━ pi exited (code {r.get('code')}) ━━", "bold red", out)

    def _pid_alive(self, name: str) -> bool:
        try:
            os.kill(int((self.dir / name).read_text()), 0)
            return True
        except (OSError, ValueError):
            return False

    def agent_alive(self) -> bool:
        """Is a headless pi daemon attached to this run (so that typing reaches the model)?"""
        return self._pid_alive("agent.pid")

    def server_alive(self) -> bool:
        return self._pid_alive("server.pid")

    def send(self, cmd: dict) -> bool:
        if not self.agent_alive():
            return False
        with (self.dir / "inbox.jsonl").open("a") as f:
            f.write(json.dumps({"time": round(time.time(), 3), **cmd}, ensure_ascii=False) + "\n")
        return True

    def poll(self) -> tuple[list[Text], bool]:
        """Read whatever is new. Returns (new visible lines, whether the ledger changed)."""
        out: list[Text] = []
        rows = [(r.get("time", 0), 0, r) for r in self.agent_log.read()]
        rows += [(r.get("time", 0), 1, r) for r in self.stream.read()]
        for _, source, r in sorted(rows, key=lambda x: x[:2]):  # two logs, one timeline
            try:
                (self._handle if source else self._agent)(r, out)
            except (KeyError, TypeError, ValueError):
                continue
        rows = self.ledger.read()
        for r in rows:
            self._ledger(r)
        return out, bool(rows)

    def rerender(self) -> list[Text]:
        return [line for kind, line in self.entries if self.visible(kind)]

    def live(self) -> Text:
        if self.visible(self.line_kind):
            return self.line
        return Text("… thinking (hidden; press t)", style="grey50")


# -- side panels --------------------------------------------------------------


SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def health(run: Run, now: float | None = None) -> tuple[str, str, str]:
    """(symbol, sentence, style): is the model working, waiting, or gone?

    The spinner only turns while records keep arriving from the server, so a
    frozen symbol means the server has gone quiet, whatever the sentence says.
    """
    st, now = run.state, now or time.time()
    quiet = now - st.last_event if st.last_event else 0.0
    spin = SPINNER[int(now * 8) % len(SPINNER)] if quiet < 8 else "◌"
    if not run.server_alive():
        if not st.started:
            return "◌", "waiting for the model server to start (bin/run)", "yellow"
        return "✖", f"the model server is not running. Restart with: bin/run {run.dir.name} --resume", "bold red"
    if st.dead:
        return "✖", f"the model thread is dead ({st.dead}). Restart with: bin/run {run.dir.name} --resume", "bold red"
    if st.phase == "loading the model":
        return spin, f"loading the model · {now - st.started:.0f}s", "yellow"
    if st.phase == "running tool":
        return "⚙", f"running {st.tool} · started {now - st.tool_started:.0f}s ago", "bold magenta"
    if quiet > 30:
        doing = {"generating": "while generating", "reading context": "while reading the context"}.get(st.phase, "")
        return "!", f"no sign of life from the model for {quiet:.0f}s {doing}".rstrip() + " (it may be hung)", "bold red"
    if st.phase == "reading context":
        done, total = st.reading or (0, 0)
        return spin, f"reading context · {done:,}/{total:,} tokens", "bold yellow"
    if st.phase == "generating":
        return spin, f"generating · {st.rate:.1f} tok/s", "bold green"
    if st.phase == "between steps" and st.agent == "working":
        return spin, "between steps · pi is preparing the next request", "green"
    if st.pi_error and st.agent != "working":
        return "!", f"stopped on an error: {st.pi_error}", "bold red"
    alive = f"model alive, heartbeat {quiet:.0f}s ago"
    if st.agent == "none" or not run.agent_alive():
        return "●", f"idle · {alive}", "grey70"
    return "●", f"idle · waiting for your message · {alive}", "grey70"


def health_text(run: Run) -> Text:
    symbol, sentence, style = health(run)
    return Text(f"{symbol} {sentence}", style=style)


def status_text(run: Run, name: str) -> Text:
    st = run.state
    t = Text()

    def row(label: str, value: str, style: str = "default"):
        t.append(f"{label:<10}", style="grey58")
        t.append(value + "\n", style=style)

    _, sentence, style = health(run)
    row("run", name, "bold")
    row("model", sentence.split(" · ")[0].split(". ")[0], style)
    agent = {"none": "not headless (watch only)", "idle": "waiting for a message", "working": "working",
             "exited": "exited"}[st.agent if run.agent_alive() or st.agent == "exited" else "none"]
    row("pi", agent, "bold green" if agent == "working" else "grey70")
    row("turn", str(st.turn))
    row("context", f"{st.context:,} tokens")
    row("generated", f"{st.total_tokens:,} tokens")
    row("tools", f"{st.tool_calls} calls" + (f", {st.tool_errors} errors" if st.tool_errors else ""))
    row("edits", f"{st.active_edits} active / {len(st.edits)} made", "bold cyan" if st.active_edits else "default")
    if st.crashes:
        row("crashes", f"{st.crashes} (recovered)", "bold red")
    if st.started:
        m, sec = divmod(int(time.time() - st.started), 60)
        row("uptime", f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m{sec:02d}s")
    return t


def edits_text(run: Run, atlas: Atlas) -> Text:
    t = Text()
    if not run.state.edits:
        t.append("no weight edits yet", style="grey50")
        return t
    for e in reversed(run.state.edits):
        base = "default" if e.active else "strike grey42"
        t.append(f"#{e.seq} ", style="bold cyan" if e.active else base)
        t.append(f"{e.op} ", style=base)
        t.append(e.target, style="bold" if e.active else base)
        d = e.detail
        if "factor" in d:
            t.append(f"  ×{d['factor']:g}", style="bold yellow" if e.active else base)
        for k in ("out_norm", "gate_norm", "up_norm"):
            if k in d:
                t.append(f"  {k[:-5]} {d[k][0]:.3g}→{d[k][1]:.3g}", style="grey58" if e.active else base)
        if "changed_values" in d:
            t.append(f"  {d['changed_values']:,} values, Δ {d.get('delta_norm', 0):.3g}", style="grey58" if e.active else base)
        t.append("\n")
        if not e.active:
            continue
        try:
            L, E, N = parse_neuron(e.target)
            for pole in ("+", "-"):
                label = atlas.label(L, E, N, pole)
                if label:
                    t.append(f"   {pole} {label[:150]}\n", style="green" if pole == "+" else "red")
        except ValueError:
            pass
        if d.get("note"):
            t.append(f"   “{d['note']}”\n", style="italic grey70")
    return t


# -- entry points ---------------------------------------------------------------


def resolve(arg: str | None) -> tuple[Path, bool]:
    """Return (run directory, whether to keep following runs/latest)."""
    runs = ROOT / "runs"
    if arg:
        p = Path(arg)
        return (p if p.is_dir() else runs / arg), False
    return runs / "latest", True


def plain(run_dir: Path, follow_latest: bool):
    console = Console(highlight=False)
    target = run_dir.resolve()
    run = Run(target)
    said = ""
    try:
        while True:
            if follow_latest and run_dir.resolve() != target:
                target = run_dir.resolve()
                run = Run(target)
                console.rule(f"now following {target.name}")
            lines, _ = run.poll()
            for line in lines:
                console.print(line, soft_wrap=True)
            symbol, sentence, style = health(run)
            if symbol in "✖!" and sentence != said:
                said = sentence
                console.print(Text(f"{symbol} {sentence}", style=style))
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass


def dump(run_dir: Path):
    """The whole run so far as plain text: every turn, tool call and result, then the edits."""
    console = Console(highlight=False, width=160)
    run = Run(run_dir.resolve(), show_thinking=True, full_results=True)
    lines, _ = run.poll()
    console.print(Text(f"run {run.dir.name} · transcript written {time.strftime('%Y-%m-%d %H:%M:%S')}\n"))
    for line in lines:
        console.print(line, soft_wrap=True)
    if run.line.plain:
        console.print(run.line, soft_wrap=True)
    console.print(Text("\n━━ weight edits, newest first (struck through = undone) ━━"))
    console.print(edits_text(run, Atlas()), soft_wrap=True)
    symbol, sentence, _ = health(run)
    console.print(Text(f"\n{symbol} {sentence}"))


def make_app(run_dir: Path, follow_latest: bool):
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.binding import Binding
    from textual.widgets import Footer, Input, RichLog, Static

    class Monitor(App):
        TITLE = "auto_surgery"
        CSS = """
        #main { width: 2fr; }
        #side { width: 1fr; min-width: 34; max-width: 60; border-left: solid $primary-darken-2; }
        #log { height: 1fr; padding: 0 1; }
        #live { height: auto; max-height: 8; padding: 0 1; border-top: dashed $primary-darken-2; }
        #health { height: 1; padding: 0 1; }
        #input { height: 3; }
        #status { height: auto; padding: 0 1; border-bottom: solid $primary-darken-2; }
        #edits-title { height: 1; padding: 0 1; color: $text-muted; }
        #edits { padding: 0 1; }
        """
        BINDINGS = [
            ("i", "write", "Write to the model"),
            Binding("ctrl+t", "task", "Send task prompt", priority=True),
            Binding("ctrl+x", "interrupt", "Interrupt", priority=True),
            ("f", "follow", "Follow on/off"),
            ("t", "thinking", "Thinking on/off"),
            ("r", "results", "Full tool results"),
            ("q", "quit", "Quit"),
            Binding("escape", "leave_input", "Back", show=False),
        ]

        def compose(self) -> ComposeResult:
            with Horizontal():
                with Vertical(id="main"):
                    yield RichLog(id="log", wrap=True, markup=False, highlight=False, auto_scroll=True)
                    yield Static(id="live")
                    yield Static(id="health")
                    yield Input(id="input")
                with Vertical(id="side"):
                    yield Static(id="status")
                    yield Static("weight edits (newest first)", id="edits-title")
                    with VerticalScroll():
                        yield Static(id="edits")
            yield Footer()

        def on_mount(self):
            self.atlas = Atlas()
            self.target = run_dir.resolve()
            self.run_state = Run(self.target)
            self.set_interval(0.15, self.tick)
            self.tick(force=True)
            self.query_one("#log", RichLog).focus()

        def tick(self, force: bool = False):
            if follow_latest and run_dir.resolve() != self.target:
                self.target = run_dir.resolve()
                self.run_state = Run(self.target, self.run_state.show_thinking, self.run_state.full_results)
                self.query_one("#log", RichLog).clear()
                force = True
            run = self.run_state
            lines, ledger_changed = run.poll()
            log = self.query_one("#log", RichLog)
            for line in lines:
                log.write(line)
            self.query_one("#live", Static).update(run.live())
            self.query_one("#health", Static).update(health_text(run))
            self.query_one("#status", Static).update(status_text(run, self.target.name))
            if ledger_changed or force:
                self.query_one("#edits", Static).update(edits_text(run, self.atlas))
            box = self.query_one("#input", Input)
            alive = run.agent_alive()
            box.disabled = not alive
            box.placeholder = (
                "press i to write to the model · enter sends · esc goes back"
                if alive else "watch only: pi is not running headless here (start it with bin/run)"
            )

        def _send(self, cmd: dict, what: str):
            if not self.run_state.send(cmd):
                self.notify("No headless pi is attached to this run, so there is nobody to send to.", severity="warning")
            else:
                self.notify(what)

        def on_input_submitted(self, event: Input.Submitted):
            text = event.value.strip()
            if text:
                self._send({"type": "prompt", "message": text}, "sent to the model")
            event.input.value = ""

        def action_write(self):
            box = self.query_one("#input", Input)
            if not box.disabled:
                box.focus()
            else:
                self.notify("Watch only: pi is not running headless for this run.", severity="warning")

        def action_leave_input(self):
            self.query_one("#log", RichLog).focus()

        def action_task(self):
            task_file = self.target / "task.md"  # the prompt this run was started with, if any
            if not task_file.is_file():
                task_file = ROOT / "prompts" / "task.md"
            try:
                task = task_file.read_text().strip()
            except OSError:
                self.notify(f"cannot read {task_file}", severity="error")
                return
            self._send({"type": "prompt", "message": task}, "task prompt sent")

        def action_interrupt(self):
            self._send({"type": "abort"}, "asked pi to stop the current turn")

        def _redraw(self):
            log = self.query_one("#log", RichLog)
            log.clear()
            for line in self.run_state.rerender():
                log.write(line)

        def action_follow(self):
            log = self.query_one("#log", RichLog)
            log.auto_scroll = not log.auto_scroll
            if log.auto_scroll:
                log.scroll_end(animate=False)
            self.notify("following the newest output" if log.auto_scroll else "scroll freely; press f to follow again")

        def action_thinking(self):
            self.run_state.show_thinking = not self.run_state.show_thinking
            self._redraw()

        def action_results(self):
            self.notify("tool results from now on are shown " + ("in full" if not self.run_state.full_results else "shortened"))
            self.run_state.full_results = not self.run_state.full_results

    return Monitor()


def tui(run_dir: Path, follow_latest: bool):
    make_app(run_dir, follow_latest).run()


def main():
    ap = argparse.ArgumentParser(description="Watch an auto_surgery run.")
    ap.add_argument("run", nargs="?", help="run name or directory (default: runs/latest)")
    ap.add_argument("--plain", action="store_true", help="print the transcript instead of opening the TUI")
    ap.add_argument("--dump", action="store_true", help="print the whole transcript so far and exit")
    a = ap.parse_args()
    run_dir, follow = resolve(a.run)
    if not follow and not run_dir.is_dir():
        sys.exit(f"no such run: {run_dir}")
    if a.dump:
        return dump(run_dir)
    (plain if a.plain or not sys.stdout.isatty() else tui)(run_dir, follow)


if __name__ == "__main__":
    main()
