"""Local view of Goodfire's public Qwen3.6 Neuron Atlas.

The atlas labels a random sample of MoE expert neurons. A neuron is addressed
as layer / expert / neuron (``L14/E78/N441``) and has two poles: ``+`` (what
the neuron fires positively on) and ``-`` (what it fires negatively on).

``build_index()`` condenses the mirrored release (atlas/data, see fetch.py)
into one small JSON-lines file; ``Atlas`` searches and renders it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "atlas"
DATA = ROOT / "data"
INDEX = ROOT / "index.jsonl"

NEURON_RE = re.compile(r"L?(\d+)\s*[/_:,\s]\s*E?(\d+)\s*[/_:,\s]\s*N?(\d+)", re.I)


def parse_neuron(s) -> tuple[int, int, int]:
    """'L14/E78/N441', 'L14_E78_N441__pos', '14/78/441', [14, 78, 441] -> (14, 78, 441)."""
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return int(s[0]), int(s[1]), int(s[2])
    m = NEURON_RE.search(str(s))
    if not m:
        raise ValueError(f"cannot parse neuron id {s!r}; use the form L14/E78/N441")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def fmt_neuron(L: int, E: int, N: int) -> str:
    return f"L{L}/E{E}/N{N}"


def _snippet(tokens, focal: int | None, radius: int = 14) -> str:
    """Render an atlas example as text with the peak token wrapped in «»."""
    if not tokens:
        return ""
    if focal is None or not (0 <= focal < len(tokens)):
        focal = max(range(len(tokens)), key=lambda i: abs(tokens[i][1]))
    lo, hi = max(0, focal - radius), min(len(tokens), focal + radius // 2 + 1)
    parts = []
    for i in range(lo, hi):
        t = tokens[i][0]
        parts.append(f"«{t}»" if i == focal else t)
    return ("…" if lo > 0 else "") + "".join(parts).replace("\n", "⏎") + ("…" if hi < len(tokens) else "")


def build_index() -> int:
    """Condense atlas/data into atlas/index.jsonl (one line per pole)."""
    idx = json.loads((DATA / "public_atlas_data/index.json").read_text())
    modes = {}
    p = DATA / "subcircuit_modes_index.json"
    if p.exists():
        modes = json.loads(p.read_text()).get("poles", {})
    sim = {}
    p = DATA / "ns11_pages/data/index.json"
    if p.exists():
        sim = json.loads(p.read_text()).get("sim", {})

    n = 0
    with INDEX.open("w") as out:
        for neuron in idx["neurons"]:
            for sign, pole in neuron["poles"].items():
                pid = pole["pid"]
                row = {
                    "pid": pid,
                    "L": neuron["L"],
                    "E": neuron["E"],
                    "N": neuron["N"],
                    "pole": sign,
                    "cat": pole.get("cat"),
                    "desc": pole.get("desc") or "",
                    "sim": sim.get(pid),
                    "modes": [],
                    "islands": [],
                }
                m = modes.get(pid)
                if m:
                    row["verdict"] = m.get("verdict")
                    row["modes"] = [
                        {"label": x.get("label") or "", "share": x.get("share")} for x in m.get("modes") or []
                    ]
                detail = DATA / f"public_atlas_data/islands/poles/{pid}.json"
                if detail.exists():
                    try:
                        d = json.loads(detail.read_text())
                    except json.JSONDecodeError:
                        d = {}
                    for isl in d.get("islands") or []:
                        exs = [e for e in isl.get("examples") or [] if e.get("label") == "top"][:2]
                        row["islands"].append(
                            {
                                "name": isl.get("name") or "",
                                "n": isl.get("n"),
                                "description": isl.get("description") or "",
                                "examples": [
                                    {"act": e.get("act"), "text": _snippet(e.get("tokens"), e.get("focal_idx"))}
                                    for e in exs
                                ],
                            }
                        )
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n


class Atlas:
    def __init__(self, index: Path = INDEX):
        self.rows: list[dict] = []
        self.by_neuron: dict[tuple[int, int, int], dict[str, dict]] = {}
        if index.exists():
            with index.open() as f:
                for line in f:
                    r = json.loads(line)
                    self.rows.append(r)
                    self.by_neuron.setdefault((r["L"], r["E"], r["N"]), {})[r["pole"]] = r
        for r in self.rows:
            r["_hay_desc"] = r["desc"].lower()
            r["_hay_more"] = " ".join(
                [m["label"] for m in r["modes"]] + [i["name"] + " " + i["description"] for i in r["islands"]]
            ).lower()
        self.layers = sorted({r["L"] for r in self.rows})

    def __len__(self) -> int:
        return len(self.rows)

    def label(self, L: int, E: int, N: int, pole: str) -> str | None:
        r = self.by_neuron.get((L, E, N), {}).get(pole)
        return r["desc"] if r else None

    def search(self, query: str, layer: int | None = None, limit: int = 20) -> list[tuple[float, dict]]:
        """Rank poles by how many query terms their description matches.

        Terms are whitespace/comma separated and matched as case-insensitive
        substrings ("happ" matches happy/happiness). A term in the one-line
        description counts 1.0; a term found only in the detailed mode/island
        text counts 0.4.
        """
        terms = [t for t in re.split(r"[\s,|]+", query.lower()) if t]
        if not terms:
            return []
        phrase = " ".join(terms)
        scored = []
        for r in self.rows:
            if layer is not None and r["L"] != layer:
                continue
            s = 0.0
            for t in terms:
                if t in r["_hay_desc"]:
                    s += 1.0
                elif t in r["_hay_more"]:
                    s += 0.4
            if s == 0:
                continue
            if len(terms) > 1 and phrase in r["_hay_desc"]:
                s += 1.0
            scored.append((s, r))
        scored.sort(key=lambda x: (-x[0], -(x[1].get("sim") or 0), x[1]["L"]))
        return scored[:limit]

    @staticmethod
    def line(r: dict) -> str:
        sim = f" sim={r['sim']:.2f}" if r.get("sim") is not None else ""
        return f"{fmt_neuron(r['L'], r['E'], r['N'])} [{r['pole']}] ({r.get('cat') or '?'}{sim}) {r['desc']}"

    def describe(self, L: int, E: int, N: int, max_islands: int = 5) -> str:
        poles = self.by_neuron.get((L, E, N))
        if not poles:
            return (
                f"{fmt_neuron(L, E, N)} is not in the atlas (the atlas labels a random sample of "
                f"{len(self.by_neuron)} neurons in layers {self.layers})."
            )
        out = []
        for sign in ("+", "-"):
            r = poles.get(sign)
            if not r:
                continue
            sim = f", simulatability {r['sim']:.2f}" if r.get("sim") is not None else ""
            out.append(f"pole {sign} ({r.get('cat') or '?'}{sim}): {r['desc']}")
            if len(r["modes"]) > 1:
                for m in r["modes"][:4]:
                    share = f"{100 * m['share']:.0f}%" if m.get("share") is not None else "?"
                    out.append(f"    mode ({share}): {m['label']}")
            for isl in r["islands"][:max_islands]:
                out.append(f"    cluster (n={isl['n']}): {isl['name']}")
                for ex in isl["examples"][:1]:
                    act = f"{ex['act']:+.2f}" if ex.get("act") is not None else "?"
                    out.append(f"        act {act}: {ex['text']}")
        return "\n".join(out)


if __name__ == "__main__":
    print(f"wrote {build_index()} poles to {INDEX}")
