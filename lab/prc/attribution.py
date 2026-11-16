"""Where the refusal decision lives: gradient of the refusal score times the difference between a
Tiananmen question and the average answered question, per layer, site and template position.

    .venv/bin/python lab/prc/attribution.py runs/<run>/lab
"""
import sys
from pathlib import Path

import numpy as np

z = np.load(Path(sys.argv[1]) / "tian_grad.npz")
keep = z["stock"]                               # the questions that get the stock refusal
g = z["g"][keep]                                # (prompts, 2, layers, 6, d): [0] mid-layer, [1] after the layer
pos = ["assistant", "\\n", "<think>", "\\n\\n", "</think>", "\\n\\n(last)"]
n = int(keep.sum())
print(f"{n} refused questions; refusal score {z['score'][keep].mean():+.2f} on them, about -7.8 on answered questions\n")

d_attn = z["attn"][keep] - z["ctrl_attn"]       # what the mixer wrote, minus the control mean
d_mlp = z["mlp"][keep] - z["ctrl_mlp"]
d_res = z["resid"][keep] - z["ctrl_resid"]
A_attn = (g[:, 0] * d_attn).sum(-1).mean(0)     # (layers, 6)
A_mlp = (g[:, 1] * d_mlp).sum(-1).mean(0)
A_res = (g[:, 1] * d_res).sum(-1).mean(0)

print("First-order share of the score carried by the stream AFTER each layer, by position (nats):")
print("layer | " + " | ".join(f"{p:>10}" for p in pos) + " |   total")
for L in list(range(0, 40, 2)) + [39]:
    print(f"{L:5d} | " + " | ".join(f"{v:10.2f}" for v in A_res[L]) + f" | {A_res[L].sum():7.2f}")

print("\nWhat each layer's mixer and MLP add at these positions (gradient x difference from control), summed over the six positions:")
print("layer | mixer |  mlp  | largest single position")
for L in range(40):
    both = np.concatenate([A_attn[L], A_mlp[L]])
    k = int(np.abs(both).argmax())
    print(f"{L:5d} | {A_attn[L].sum():5.2f} | {A_mlp[L].sum():5.2f} | {'mixer' if k < 6 else 'mlp'} at {pos[k % 6]}: {both[k]:+.2f}")
print(f"\nsum over all layers: mixer {A_attn.sum():.2f}, mlp {A_mlp.sum():.2f}")

# How consistent is the gradient direction across questions, and how does it relate to the mean difference?
print("\nlayer | position </think>: |mean unit gradient| (1 = same direction for every question), cos(gradient, mean difference), |difference|, share of difference along gradient")
for p in (4, 5):
    print(f" position {pos[p]}")
    for L in range(4, 40, 4):
        gu = g[:, 1, L, p] / np.linalg.norm(g[:, 1, L, p], axis=-1, keepdims=True)
        gm = gu.mean(0)
        dm = d_res[:, L, p].mean(0)
        cos = gm @ dm / (np.linalg.norm(gm) * np.linalg.norm(dm))
        print(f"  {L:3d} | {np.linalg.norm(gm):.2f} | {cos:+.3f} | {np.linalg.norm(dm):.2f} | {abs(gm @ dm) / np.linalg.norm(gm) / np.linalg.norm(dm):.3f}")
