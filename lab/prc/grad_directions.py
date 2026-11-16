"""Per layer, the direction the refusal score reads at the two positions that carry it
(</think> and the last one): the mean over refused questions of the unit gradient.

    .venv/bin/python lab/prc/grad_directions.py runs/<run>/lab
"""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
z = np.load(out / "tian_grad.npz")
stock = z["stock"]
fit = stock & (np.arange(len(stock)) % 2 == 0)      # even-numbered refused questions fit; the rest test
g = z["g"][fit]                                      # (prompts, 2, layers, 6, d)
with np.errstate(invalid="ignore"):  # layer 39 has no gradient at the earlier positions
    unit = g / np.linalg.norm(g, axis=-1, keepdims=True)
mean = unit.mean(0)                                  # (2, layers, 6, d)
with np.errstate(invalid="ignore"):
    dirs = mean / np.linalg.norm(mean, axis=-1, keepdims=True)
np.savez(out / "grad_dirs.npz", attn=dirs[0], mlp=dirs[1], agreement=np.linalg.norm(mean, axis=-1), fit=fit)
print("fitted on", int(fit.sum()), "refused questions")
print("cosine between the </think> and last-position directions (mlp site), and between neighbouring layers at </think>:")
for L in range(16, 40, 2):
    a, b = dirs[1, L, 4], dirs[1, L, 5]
    print(f"  L{L}: think.last {a @ b:+.2f} | L.L+1 at think {dirs[1, L, 4] @ dirs[1, min(L + 1, 39), 4]:+.2f} | mid.after at think {dirs[0, L, 4] @ dirs[1, L, 4]:+.2f}")
