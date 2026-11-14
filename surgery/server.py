"""mlx_lm's OpenAI-compatible server plus /surgery/* tools on the same model.

    python -m surgery.server --run-dir runs/x --model <repo-or-path> [mlx_lm.server args]

Chat completions are served exactly as mlx_lm.server serves them. Surgery
tools are queued on the same request queue and executed by the generation
thread between requests, so reads, probes and weight edits never race with
token generation. After any edit the prompt cache is dropped, so the whole
conversation is re-read by the edited weights before the next token.

If generation fails (most likely by running out of GPU memory) the request in
flight gets an error, memory is released, and the generation loop starts
again with the model and its edits still in place.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import itertools
import json
import logging
import os
import sys
import threading
import time
import traceback
import weakref
from pathlib import Path
from queue import Empty, Queue

import mlx.core as mx
from mlx_lm import server as mls

from . import score
from .atlas import Atlas
from .core import Surgery, SurgeryError, py_help

TOOLS = {
    "atlas_search": lambda s, a: s.t_atlas_search(a["query"], a.get("layer"), a.get("limit", 20)),
    "inspect_neuron": lambda s, a: s.t_inspect_neuron(a["neuron"]),
    "probe": lambda s, a: s.t_probe(a["text"], a["neurons"], a.get("chat", False), a.get("reply") or ""),
    "contrast": lambda s, a: s.t_contrast(
        a["texts_a"], a["texts_b"], a.get("layers"), a.get("top_k", 12),
        a.get("chat", False), a.get("reply_a") or "", a.get("reply_b") or "",
    ),
    "scale_neuron": lambda s, a: s.t_scale_neuron(a["neuron"], a["factor"], a.get("note", "")),
    "scale_neurons": lambda s, a: s.t_scale_neurons(a["edits"], a.get("note", "")),
    "score": score.tool,
    "python": lambda s, a: s.t_python(a["code"]),
    "sample_self": lambda s, a: s.t_sample_self(
        a["prompt"], a.get("system"), a.get("max_tokens", 300), a.get("thinking", False), a.get("temperature", 0.7)
    ),
    "ledger": lambda s, a: s.t_ledger(a.get("action", "list"), a.get("n", 1)),
    "python_help": lambda s, a: py_help(s),
    "status": lambda s, a: json.dumps(s.status()),
}


class Job:
    """One tool call. ``source`` is "model" (pi, on the model's behalf) or "operator" (bin/score and the like)."""

    def __init__(self, name: str, args: dict, source: str = "model", quiet: bool = False):
        self.name, self.args, self.source, self.done = name, args, source, Queue()
        self.quiet = quiet and source == "operator"  # leave no record of the call in the run directory


class PinnedModelProvider(mls.ModelProvider):
    """Serve the one loaded model whatever name a request asks for.

    A reload would silently discard every weight edit, so it must never happen.
    """

    def load(self, model_path=None, adapter_path=None, draft_model_path=None):
        if self.model is None:
            self._load(self._model_map["default_model"], self._adapter_map["default_model"], None)
        return self.model, self.tokenizer


class Generator(mls.ResponseGenerator):
    def __init__(self, model_provider, prompt_cache, run_dir: Path, mem_budget_gb: float, allow_python: bool,
                 keep_cache: bool = False, replay: bool = False, private: dict | None = None):
        self.surgery: Surgery | None = None
        self.keep_cache = keep_cache
        self.replay = replay
        self.private = private  # the operator's start-up settings for this run; see bin/serve
        self._epoch = 0  # bumped when generation crashes, so requests in flight fail instead of waiting
        self._queue_epoch: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._crashes: list[float] = []
        self._last_error = ""
        self._last_beat = 0.0
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.mem_budget_gb = mem_budget_gb
        self.allow_python = allow_python
        self._stream_lock = threading.Lock()
        self._stream_file = (self.run_dir / "stream.jsonl").open("a", buffering=1)
        self._request_ids = itertools.count(1)
        self._emit({"t": "server_start"})
        super().__init__(model_provider, prompt_cache)  # starts the generation thread

    # -- a live record of everything the model reads and writes, for bin/watch --

    def _emit(self, row: dict):
        row = {"time": round(time.time(), 3), **row}
        with self._stream_lock:
            try:
                self._stream_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            except (OSError, ValueError):
                pass

    @staticmethod
    def _text_of(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return ""

    def generate(self, request, generation_args, progress_callback=None):
        rid = next(self._request_ids)
        messages = getattr(request, "messages", None) or []
        start = {"t": "start", "id": rid, "messages": len(messages)}
        if messages and messages[-1].get("role") == "user":
            start["user"] = self._text_of(messages[-1].get("content"))

        def progress(done, total):
            self._emit({"t": "prompt", "id": rid, "done": done, "total": total})
            if progress_callback is not None:
                progress_callback(done, total)

        ctx, responses = super().generate(request, generation_args, progress)
        start.update(prompt_tokens=len(ctx.prompt), cached_tokens=max(0, ctx.prompt_cache_count), state=ctx.initial_state)
        self._emit(start)

        def tee():
            n, t0, first, finish = 0, time.time(), None, None
            try:
                for r in responses:
                    n += 1
                    first = first or time.time()  # everything before this was reading the prompt
                    finish = r.finish_reason or finish
                    if r.text:
                        self._emit({"t": "tok", "id": rid, "s": r.text})
                    yield r
            finally:
                now = time.time()
                self._emit({
                    "t": "end", "id": rid, "tokens": n, "finish": finish,
                    "seconds": round(now - (first or now), 2), "reading_seconds": round((first or now) - t0, 2),
                })

        return ctx, tee()

    # -- staying alive ------------------------------------------------------------

    def _run_generate(self):
        """Run the generation loop; if it crashes, clean up and run it again."""
        while not self._stop:
            try:
                self._generate()
                return
            except Exception as e:  # noqa: BLE001
                self._last_error = f"{type(e).__name__}: {e}"
                logging.exception("generation loop crashed: %s", e)
                now = time.time()
                self._crashes = [t for t in self._crashes if now - t < 300] + [now]
                self._epoch += 1  # every request that was in flight now fails
                if len(self._crashes) >= 4:
                    self._generation_failed = True
                    self._emit({"t": "dead", "error": self._last_error})
                    return
                try:
                    self.prompt_cache.trim_to(n_sequences=0)
                    gc.collect()
                    mx.clear_cache()
                except Exception:  # noqa: BLE001
                    logging.exception("cleanup after the crash failed")
                self._emit({
                    "t": "crash", "error": self._last_error,
                    "active_edits": len(self.surgery.edits) if self.surgery else 0,
                })

    def _await_response(self, response_queue):
        epoch = self._queue_epoch.setdefault(response_queue, self._epoch)
        while True:
            try:
                return response_queue.get(timeout=1.0)
            except Empty:
                if not self.generation_available():
                    raise RuntimeError(f"the model thread is dead ({self._last_error})") from None
                if self._epoch != epoch:
                    raise RuntimeError(
                        f"generation failed ({self._last_error}); the model recovered with its weight edits "
                        "intact, so the request can simply be retried"
                    ) from None

    def _init_surgery(self):
        model, tokenizer = self.model_provider.load()
        self.surgery = Surgery(model, tokenizer, self.run_dir, Atlas(), self.mem_budget_gb)
        if self.private:
            # Nothing about this is written to the run's logs: the model can read its own run directory.
            if self.private.get("baseline"):
                self.surgery.bake(self.private["baseline"])
                self.surgery.dirty = False
            score.PRIVATE.update(self.private.get("metrics") or {})
            score.HIDE = bool(self.private.get("hide_metrics"))
            self.private = None
        if self.replay:
            msg = self.surgery.replay()
            self.surgery.dirty = False
            logging.info(msg)
            self._emit({"t": "replayed", "text": msg, "active_edits": len(self.surgery.edits)})
        self._emit({"t": "ready"})

    def _next_request(self, timeout=None):
        if self.surgery is None:
            self._init_surgery()
        if time.time() - self._last_beat > 5:  # this loop turning over is the sign of life bin/watch looks for
            self._last_beat = time.time()
            self._emit({"t": "beat"})
        req = super()._next_request(timeout)
        while isinstance(req, Job):
            if req.name == "_crash" and os.environ.get("SURGERY_TEST_CRASH"):  # lets a test kill the loop
                req.done.put((True, "crashing"))
                raise RuntimeError("injected crash")
            self._run_job(req)
            req = super()._next_request(None)
        return req

    def _run_job(self, job: Job):
        t0 = time.time()
        ok, text = False, ""
        logged = job.name != "status" and not job.quiet
        if logged:
            self._emit({"t": "tool_start", "tool": job.name, "source": job.source})
        try:
            fn = TOOLS.get(job.name)
            if fn is None:
                raise SurgeryError(f"unknown tool {job.name!r}; tools: {', '.join(TOOLS)}")
            if job.name == "python" and not self.allow_python:
                raise SurgeryError("the python tool is disabled for this run")
            args = self.surgery.read_safe(job.args)  # ‹...› written by the model stands for the control token
            args = {**args, "_source": job.source, "_quiet": job.quiet} if job.name == "score" else args
            text, ok = fn(self.surgery, args), True
        except SurgeryError as e:
            text = str(e)
        except KeyError as e:
            text = f"missing argument: {e}"
        except Exception as e:  # noqa: BLE001
            logging.error("surgery tool %s failed:\n%s", job.name, traceback.format_exc())
            text = f"{type(e).__name__}: {e}"
        finally:
            s = self.surgery
            if s is not None and s.dirty:
                if not self.keep_cache:
                    self.prompt_cache.trim_to(n_sequences=0)
                s.dirty = False
        if self.surgery is not None:  # a control token in a result would act on the model's reading of it
            text = self.surgery.show_safe(text)
        row = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ts": round(time.time(), 3),
            "tool": job.name,
            "source": job.source,
            "args": job.args,
            "ok": ok,
            "seconds": round(time.time() - t0, 2),
            "result": text,
            "active_edits": len(self.surgery.edits) if self.surgery else 0,
            "budget_gb": round(self.surgery._budget_used() / 1e9, 3) if self.surgery else 0,
        }
        if logged:
            try:
                with (self.run_dir / "tools.jsonl").open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                logging.exception("could not write tools.jsonl")
        if logged:
            self._emit({
                "t": "tool", "tool": job.name, "source": job.source, "args": job.args, "ok": ok,
                "seconds": row["seconds"],
                "result": text if len(text) <= 6000 else text[:6000] + "…",
                "active_edits": row["active_edits"], "budget_gb": row["budget_gb"],
            })
        if not job.quiet:
            logging.info("surgery %s ok=%s %.1fs", job.name, ok, time.time() - t0)
        job.done.put((ok, text))


class Handler(mls.APIHandler):
    def _surgery(self, name: str, args: dict):
        source = "operator" if (self.headers.get("X-Surgery-Source") or "").lower() == "operator" else "model"
        job = Job(name, args, source, quiet=(self.headers.get("X-Surgery-Log") or "").lower() == "off")
        gen = self.response_generator
        gen.requests.put(job)
        while True:
            try:
                ok, text = job.done.get(timeout=1.0)
                break
            except Empty:
                if not gen.generation_available():
                    ok, text = False, f"the model thread is dead ({gen._last_error}); the server must be restarted"
                    break
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps({"ok": ok, "text": text}, ensure_ascii=False).encode())
        self.wfile.flush()

    def do_POST(self):
        if not self.path.startswith("/surgery/"):
            return super().do_POST()
        try:
            n = int(self.headers.get("Content-Length") or 0)
            args = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(args, dict):
                raise ValueError("body must be a JSON object")
        except ValueError as e:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "text": f"bad request: {e}"}).encode())
            return
        self._surgery(self.path[len("/surgery/"):].strip("/"), args)

    def do_GET(self):
        if self.path.rstrip("/") == "/surgery/status":
            return self._surgery("status", {})
        return super().do_GET()

    def log_message(self, fmt, *args):  # quieter access log
        logging.debug("http " + fmt, *args)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mem-budget-gb", type=float, default=6.0,
                   help="memory allowed for whole-tensor replacements (single-neuron edits are free)")
    p.add_argument("--cache-gb", type=float, default=3.0, help="cap on the prompt cache")
    p.add_argument("--replay", action="store_true",
                   help="re-apply the edits that were active when the previous server on this run stopped")
    p.add_argument("--no-python", action="store_true", help="disable the python tool")
    p.add_argument("--keep-cache", action="store_true",
                   help="after an edit, keep the cached state of the conversation instead of re-reading it "
                        "with the edited weights (faster, but earlier context was then read by the old weights)")
    p.add_argument("--private-fd", type=int, default=None,
                   help="an open file descriptor holding the operator's start-up settings as JSON (see bin/serve)")
    ours, rest = p.parse_known_args()
    private = None
    if ours.private_fd is not None:
        with os.fdopen(ours.private_fd) as f:
            private = json.load(f)

    def run(host, port, model_provider, **_):
        # Freed buffers go back to the system instead of piling up in MLX's pool: with a
        # 38 GB model on a 64 GB machine there is little room for a high-water mark.
        mx.set_cache_limit(2 * 1024**3)
        run_dir = Path(ours.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        pid_file = run_dir / "server.pid"
        pid_file.write_text(str(os.getpid()))
        atexit.register(lambda: pid_file.unlink(missing_ok=True))
        cache = mls.LRUPromptCache(model_provider.cli_args.prompt_cache_size, int(ours.cache_gb * 1024**3))
        gen = Generator(model_provider, cache, run_dir, ours.mem_budget_gb, not ours.no_python,
                        ours.keep_cache, ours.replay, private)
        mls._run_http_server(host, port, gen, handler_class=Handler)

    mls.run = run
    mls.ModelProvider = PinnedModelProvider
    sys.argv = [sys.argv[0], *rest]
    mls.main()


if __name__ == "__main__":
    main()
