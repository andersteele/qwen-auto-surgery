"""Mirror Goodfire's public Qwen3.6 Neuron Atlas into atlas/data/.

Source: https://www.goodfire.com/research/public_neuron_atlas
The explorer is a static site under an immutable release prefix; we copy the
index files and the per-pole detail files so the agent can use the atlas offline.

    uv run python atlas/fetch.py            # resume-able; skips files already present
"""

from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

RELEASE = "https://static.goodfire.ai/neuron-atlas/2026-11-09-082204"
OUT = Path(__file__).parent / "data"
UA = "neuron-atlas-mirror/0.1 (python-urllib)"
INDEXES = [
    "public_atlas_data/index.json",
    "subcircuit_modes_index.json",
    "ns11_pages/data/index.json",
    "neuron_landscape_data/atlas_index.json",
]


def get(rel: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(
        f"{RELEASE}/{rel}",
        headers={"User-Agent": UA, "Accept-Encoding": "gzip"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_to(rel: str) -> tuple[str, bool, str]:
    dest = OUT / rel
    if dest.exists() and dest.stat().st_size > 0:
        return rel, True, "cached"
    try:
        body = get(rel)
        json.loads(body)  # refuse to store anything that is not JSON
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.rename(dest)
        return rel, True, "fetched"
    except Exception as e:  # noqa: BLE001
        return rel, False, repr(e)


def main() -> int:
    for rel in INDEXES:
        print(fetch_to(rel), flush=True)
    index = json.loads((OUT / "public_atlas_data/index.json").read_text())
    pids = [p["pid"] for n in index["neurons"] for p in n["poles"].values()]
    rels = [f"public_atlas_data/islands/poles/{pid}.json" for pid in pids]
    print(f"{len(rels)} pole files", flush=True)
    ok = bad = 0
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for i, (rel, good, msg) in enumerate(ex.map(fetch_to, rels), 1):
            ok += good
            bad += not good
            if not good:
                print("FAIL", rel, msg, flush=True)
            if i % 500 == 0:
                print(f"{i}/{len(rels)} ok={ok} bad={bad}", flush=True)
    print(f"done ok={ok} bad={bad}", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
