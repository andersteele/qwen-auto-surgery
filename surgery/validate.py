"""Check the real model against the atlas and the surgery invariants.

    uv run python -m surgery.validate [--model REPO_OR_PATH] [--poles 40]

1. Addressing: for a sample of atlas poles, replay the atlas's own top example
   and compare our activation at its focal token with the value the atlas
   recorded. If layer/expert/neuron or the sign convention were off, these
   would not line up.
2. An edit that rewrites a neuron with its own values changes no logit.
3. A real edit changes the logits; undo restores them exactly.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from .atlas import DATA, Atlas
from .core import Surgery

PROMPT = "The weather was lovely that morning, and for the first time in weeks she felt"


def atlas_examples(n: int, seed: int = 0):
    files = sorted((DATA / "public_atlas_data/islands/poles").glob("*.json"))
    random.Random(seed).shuffle(files)
    for f in files:
        d = json.loads(f.read_text())
        best = None
        for isl in d.get("islands") or []:
            for ex in isl.get("examples") or []:
                if ex.get("label") == "top" and ex.get("focal_idx") is not None and ex.get("act"):
                    if best is None or abs(ex["act"]) > abs(best["act"]):
                        best = ex
        if best:
            yield d, best
            n -= 1
            if n == 0:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-8bit")
    ap.add_argument("--poles", type=int, default=40)
    a = ap.parse_args()

    t0 = time.time()
    model, tok = load(a.model)
    s = Surgery(model, tok, Path(tempfile.mkdtemp()), Atlas())
    print(f"loaded in {time.time() - t0:.0f}s: {json.dumps(s.status())}")

    # 1. addressing
    ours, theirs, routed, skipped = [], [], 0, 0
    for d, ex in atlas_examples(a.poles):
        pieces = [t[0] for t in ex["tokens"]]
        ids = s.tokens("".join(pieces))
        if len(ids) != len(pieces):  # the excerpt does not re-tokenize to the same tokens
            skipped += 1
            continue
        L, E, N = d["layer"], d["expert"], d["neuron"]
        _, rec = s.forward(ids, neurons=[(L, E, N)])
        act = np.array(rec.acts[(L, E, N)])
        f = ex["focal_idx"]
        routed += bool(np.array(rec.route[(L, E)])[f] > 0)
        ours.append(float(act[f]))
        theirs.append(float(ex["act"]))  # already signed: negative on a "-" pole
    ours, theirs = np.array(ours), np.array(theirs)
    same_sign = float(np.mean(np.sign(ours) == np.sign(theirs)))
    corr = float(np.corrcoef(ours, theirs)[0, 1]) if len(ours) > 2 else float("nan")
    ratio = float(np.median(ours[theirs != 0] / theirs[theirs != 0]))
    print(
        f"\n1. atlas replay over {len(ours)} poles ({skipped} skipped): expert routed at the focal token on "
        f"{routed}/{len(ours)}; sign agrees on {100 * same_sign:.0f}%; correlation {corr:.3f}; "
        f"median ours/atlas {ratio:.2f}"
    )
    for o, t in list(zip(ours, theirs))[:8]:
        print(f"     ours {o:+8.3f}   atlas {t:+8.3f}")

    # 2. an edit that changes no value must change no logit
    ids = s.tokens(PROMPT)
    t0 = time.time()
    base = np.array(s.forward(ids)[0])
    dt = time.time() - t0
    _, rec = s.forward(ids, agg_layers=[20])
    pos = np.array(rec.agg[20]["pos"])
    E, N = divmod(int(pos.argmax()), pos.shape[1])
    s.set_neuron((20, E, N), **{w: s.neuron_vec(20, E, N, w) for w in ("gate", "up", "out")})
    same = np.array(s.forward(ids)[0])
    s.undo()
    top = lambda x: s.pieces(np.argsort(-x)[:5])  # noqa: E731
    print(
        f"\n2. rewriting L20/E{E}/N{N} with its own values: max |logit change| {np.abs(same - base).max():.6f} "
        f"(must be 0); forward pass over {len(ids)} tokens took {dt:.2f}s; top tokens {top(base)}"
    )

    # 3. edit / undo
    s.scale_neuron((20, E, N), 30.0)
    edited = np.array(s.forward(ids)[0])
    s.undo()
    restored = np.array(s.forward(ids)[0])
    print(
        f"\n3. scaling the most active layer-20 neuron on the prompt (L20/E{E}/N{N}) by 30: max |logit change| "
        f"{np.abs(edited - base).max():.4f}, top tokens {top(edited)}; after undo "
        f"{np.abs(restored - base).max():.6f} (must be 0)"
    )
    ok = same_sign > 0.9 and corr > 0.8 and np.abs(restored - base).max() == 0 and np.abs(same - base).max() == 0
    print("\nPASS" if ok else "\nCHECK THE NUMBERS ABOVE")


if __name__ == "__main__":
    main()
