"""Candidate directions from the saved residual streams: per layer and position, the mean over
censored questions minus the mean over controls. Run outside the server (numpy only).

    .venv/bin/python lab/prc/directions.py runs/<run>/lab
"""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
R = {k: np.load(out / f"resid_off_{k}.npy") for k in ("censored", "elsewhere", "china", "everyday")}  # (prompts, layers, 6, d)
base = json.loads((out / "baseline_off.json").read_text())
fit = {k: np.arange(len(v)) % 2 == 0 for k, v in R.items()}  # even-numbered questions fit the direction, odd ones test it

refused = np.array(["As an AI assistant, I must emphasi" in r["opening"] for r in base["censored"]])
print("stock refusal on", int(refused.sum()), "of", len(refused), "censored questions")


def mean(k, mask=None):
    m = fit[k] if mask is None else mask
    return R[k][m].mean(0)


cands = {
    "cens-else": mean("censored") - mean("elsewhere"),
    "cens-both": mean("censored") - 0.5 * (mean("elsewhere") + mean("china")),
    "cens-china": mean("censored") - mean("china"),
    "refused-else": mean("censored", refused & fit["censored"]) - mean("elsewhere"),
    "refused-cens": mean("censored", refused & fit["censored"]) - mean("censored", ~refused & fit["censored"]),
}
np.savez(out / "candidates.npz", **cands)

pos = -1  # the position that chooses the first token of the reply
norm = np.linalg.norm(R["elsewhere"][:, :, pos], axis=-1).mean(0)  # typical size of the stream, per layer
print("\nlayer | stream norm | |r| for each candidate (last position) | held-out separation d' of cens-else, cens-both, refused-cens")
for L in range(0, 40, 2):
    row = [np.linalg.norm(c[L, pos]) for c in cands.values()]
    seps = []
    for name, a, b in (("cens-else", "censored", "elsewhere"), ("cens-both", "censored", "china")):
        u = cands[name][L, pos] / np.linalg.norm(cands[name][L, pos])
        pa, pb = R[a][~fit[a], L, pos] @ u, R[b][~fit[b], L, pos] @ u
        seps.append((pa.mean() - pb.mean()) / np.sqrt(0.5 * (pa.var() + pb.var())))
    u = cands["refused-cens"][L, pos] / np.linalg.norm(cands["refused-cens"][L, pos])
    pa, pb = R["censored"][refused & ~fit["censored"], L, pos] @ u, R["censored"][~refused & ~fit["censored"], L, pos] @ u
    seps.append((pa.mean() - pb.mean()) / np.sqrt(0.5 * (pa.var() + pb.var()) + 1e-9))
    print(f"{L:5d} | {norm[L]:8.1f} | " + " ".join(f"{v:7.2f}" for v in row) + " | " + " ".join(f"{v:6.1f}" for v in seps))

print("\ncosine between candidates at layers 12, 20, 28 (last position)")
names = list(cands)
for L in (12, 20, 28):
    U = np.stack([cands[n][L, pos] / np.linalg.norm(cands[n][L, pos]) for n in names])
    print(L, np.round(U @ U.T, 2).tolist())
print("\ncosine of cens-else between layers (last position), rows 8, 12, 16, 20, 24, 28, 32, 36")
Ls = [8, 12, 16, 20, 24, 28, 32, 36]
U = np.stack([cands["cens-else"][L, pos] / np.linalg.norm(cands["cens-else"][L, pos]) for L in Ls])
print(np.round(U @ U.T, 2))
print("\ncosine of cens-else between the six positions at layer 20:")
U = np.stack([cands["cens-else"][20, p] / np.linalg.norm(cands["cens-else"][20, p]) for p in range(6)])
print(np.round(U @ U.T, 2))
