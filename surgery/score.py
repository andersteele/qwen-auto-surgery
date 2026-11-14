"""A fixed measurement to repeat before and after weight edits.

A metric is a JSON file in metrics/ holding groups of questions. Each question is put to
the current weights in a fresh chat (thinking off) and two things are recorded:

  opening  the first ``max_tokens`` tokens of the reply at temperature 0. The same weights
           always give the same opening. It is *flagged* when it matches one of the group's
           patterns (a disclaimer, a refusal, the right answer: whatever the group looks for).
  margin   log P(the reply opens with text A) - log P(it opens with text B), in nats, computed
           exactly by teacher forcing. It moves continuously, so it shows a shift long before
           the opening itself changes.

Groups marked "target" hold the behaviour under study; groups marked "control" hold behaviour
that should stay as it is. Items marked "heldout" are left out of the model's own `score`
tool and can be run by the operator (bin/score --heldout), to tell a change in behaviour from
edits fitted to the questions that were used to find them.

    {"description": "...", "max_tokens": 48, "system": null,
     "groups": [{"name": "...", "role": "target" | "control", "looks_for": "what a flag means",
                 "flag": ["regex", ...], "a": "opening A", "b": "opening B",
                 "items": [{"prompt": "...", "heldout": false, "flag": [...], "a": "...", "b": "..."}]}]}

Every run of a metric is appended to <run>/scores.jsonl.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .core import Surgery, SurgeryError

METRICS = Path(__file__).resolve().parent.parent / "metrics"
MAX_ITEMS = 60
PRIVATE: dict[str, dict] = {}  # metrics only the operator can run in this run, by name (see bin/serve)
HIDE = False  # this run gives the model no ready-made metrics: it scores with its own prompts


def names() -> list[str]:
    try:
        return sorted(p.stem for p in METRICS.glob("*.json"))
    except OSError:
        return []


def describe(source: str = "model") -> str:
    found = [] if HIDE and source != "operator" else names()
    if not found:
        return "There are no ready-made metrics in this run. Pass your own `prompts` (and `flag`, or `a` and `b`)."
    lines = ["Metrics (pass one as `metric`), or pass your own `prompts` instead:"]
    for n in found:
        try:
            spec = json.loads((METRICS / f"{n}.json").read_text())
        except (OSError, ValueError):
            continue
        groups = "; ".join(
            f"{g.get('name')} ({g.get('role', 'target')}, {sum(not i.get('heldout') for i in g.get('items', []))} questions)"
            for g in spec.get("groups", [])
        )
        lines.append(f"  {n}: {spec.get('description', '')}\n      groups: {groups}")
    return "\n".join(lines)


def load(name: str, source: str = "model") -> dict:
    if source == "operator" and name in PRIVATE:
        return {"name": name, **PRIVATE[name]}
    known = False
    if re.fullmatch(r"[A-Za-z0-9_-]+", name or "") and (not HIDE or source == "operator"):
        try:
            known = (METRICS / f"{name}.json").is_file()
        except OSError:  # the run's sandbox may keep the metrics directory from this process
            pass
    if not known:
        raise SurgeryError(f"no metric named {name!r}. " + describe(source))
    try:
        spec = json.loads((METRICS / f"{name}.json").read_text())
    except ValueError as e:
        raise SurgeryError(f"metric {name!r} is not valid JSON: {e}") from None
    spec["name"] = name
    return spec


def custom(prompts, flag=None, a=None, b=None) -> dict:
    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        raise SurgeryError("pass a metric name, or a non-empty list of prompts")
    group = {"name": "your prompts", "role": "custom", "items": [{"prompt": str(p)} for p in prompts]}
    if flag:
        group["flag"] = [flag] if isinstance(flag, str) else list(flag)
        group["looks_for"] = "a match for your patterns"
    if a and b:
        group["a"], group["b"] = str(a), str(b)
    return {"name": "custom", "description": "", "groups": [group]}


def _patterns(pats) -> list[re.Pattern]:
    out = []
    for p in pats or ():
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            raise SurgeryError(f"bad pattern {p!r}: {e}") from None
    return out


def measure(s: Surgery, spec: dict, heldout: bool = False, max_tokens: int | None = None) -> dict:
    """Run every question of ``spec`` (the held-out ones instead, if asked) through the current weights."""
    max_tokens = max(1, min(int(max_tokens or spec.get("max_tokens") or 48), 200))
    todo = [(g, it) for g in spec.get("groups", []) for it in g.get("items", []) if bool(it.get("heldout")) == heldout]
    if not todo:
        raise SurgeryError(f"metric {spec.get('name')!r} has no {'held-out ' if heldout else ''}questions")
    if len(todo) > MAX_ITEMS:
        raise SurgeryError(f"at most {MAX_ITEMS} questions per score, got {len(todo)}")
    groups: dict[str, dict] = {}
    for g, it in todo:
        pats = _patterns(it.get("flag", g.get("flag")))
        toks, _, _ = s.chat_tokens(it["prompt"], system=spec.get("system"))
        opening = s.tok.decode(s.greedy(toks, max_tokens)).strip()
        a, b = it.get("a", g.get("a")), it.get("b", g.get("b"))
        row = {
            "prompt": it["prompt"],
            "opening": opening,
            "flagged": any(p.search(opening) for p in pats) if pats else None,
            "margin": round(sum(s.logprob(toks, a)) - sum(s.logprob(toks, b)), 3) if a and b else None,
        }
        if row["margin"] is not None:
            row["a"], row["b"] = a, b
        out = groups.setdefault(g.get("name", "?"), {
            "name": g.get("name", "?"), "role": g.get("role", "target"), "looks_for": g.get("looks_for", ""), "items": [],
        })
        out["items"].append(row)
    for g in groups.values():
        flags = [i["flagged"] for i in g["items"] if i["flagged"] is not None]
        margins = [i["margin"] for i in g["items"] if i["margin"] is not None]
        g["n"] = len(g["items"])
        g["flagged"] = sum(flags) if flags else None
        g["judged"] = len(flags)
        g["mean_margin"] = round(sum(margins) / len(margins), 3) if margins else None
    return {
        "metric": spec.get("name"), "heldout": heldout, "max_tokens": max_tokens,
        "active_edits": len(s.edits), "last_edit": s.edits[-1].seq if s.edits else 0,
        "groups": list(groups.values()),
    }


def render(res: dict, brief: bool = False) -> str:
    n = sum(g["n"] for g in res["groups"])
    lines = [
        f"score {res['metric']!r}: {n} {'held-out ' if res['heldout'] else ''}question(s), each asked in a fresh chat "
        f"with thinking off · {res['active_edits']} edit(s) active",
        f"opening = your first {res['max_tokens']} tokens at temperature 0 (same weights, same opening); "
        "[x] = the opening matches what the group looks for, [ ] = it does not.",
    ]
    if any(g["mean_margin"] is not None for g in res["groups"]):
        lines.append(
            "margin = log P(reply opens with A) - log P(reply opens with B), in nats: above 0 favours A. "
            "It moves before the opening does."
        )
    for g in res["groups"]:
        head = f"\n{g['name']} ({g['role']})"
        if g["flagged"] is not None:
            head += f": {g['flagged']}/{g['judged']} flagged" + (f" [{g['looks_for']}]" if g["looks_for"] else "")
        if g["mean_margin"] is not None:
            ab = next(i for i in g["items"] if i["margin"] is not None)
            same = all(i.get("a") == ab["a"] and i.get("b") == ab["b"] for i in g["items"] if i["margin"] is not None)
            head += f" · mean margin {g['mean_margin']:+.2f}" + (f"  (A = {ab['a']!r}, B = {ab['b']!r})" if same else "")
        lines.append(head)
        for i in g["items"]:
            mark = "   " if i["flagged"] is None else ("[x]" if i["flagged"] else "[ ]")
            margin = f" {i['margin']:+7.2f} " if i["margin"] is not None else " "
            lines.append(f"  {mark}{margin}{i['prompt']}")
            if not brief:
                lines.append("          → " + json.dumps(i["opening"], ensure_ascii=False))
    return "\n".join(lines)


def tool(s: Surgery, args: dict) -> str:
    """The `score` tool. ``args['_source']`` says who asked ("operator" may run held-out questions)."""
    source = args.get("_source") or "model"
    name = args.get("metric")
    if name == "list" or (not name and not args.get("prompts")):
        return describe(source)
    spec = load(name, source) if name else custom(args.get("prompts"), args.get("flag"), args.get("a"), args.get("b"))
    heldout = bool(args.get("heldout")) and source == "operator"
    res = measure(s, spec, heldout=heldout, max_tokens=args.get("max_tokens"))
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "ts": round(time.time(), 3), "source": source, **res}
    try:
        if not args.get("_quiet"):
            with (s.run_dir / "scores.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return render(res, brief=bool(args.get("brief")))
