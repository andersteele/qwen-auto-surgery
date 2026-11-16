# bin/lab lab/prc/write.py     (after fit.py, or after session.py if runs/<run>/lab/fit.npz exists)
# Measure the original weights, write the fitted update into the weights as ordinary edits
# (ledger rows, undoable, replayed by `bin/run <run> --resume`), check that the written weights
# compute what the hook did, and measure again. Thinking on is included in both measurements.
assert not S._s.edits, "start from the original weights"
S._s.model.eval()
z = np.load(OUT / "fit.npz")
FIT = {"layers": [int(L) for L in z["layers"]], "A": z["A"], "B": z["B"], "step": int(z["step"])}
print(summary(full_eval("original weights", thinking=True)))

probe = [chat(q) for q in (tian_odd[0], val_c[5], val_o[0])]
with T.lora(FIT["layers"], FIT["A"], FIT["B"]):
    hooked = [T.logits(p) for p in probe]
for path, delta in T.shared_update(FIT["layers"], FIT["A"], FIT["B"]).items():
    W = np.array(S._s.get_weight(path))
    S._s.set_weight(path, W + delta, note=f"prc-lab: rank-{FIT['A'].shape[1]} update fitted to the in-conversation answers (fit.npz, step {FIT['step']})")
written = [T.logits(p) for p in probe]
print("edits written:", len(S._s.edits), "| largest difference between hooked and written logits:",
      [round(float(np.abs(a - b).max()), 3) for a, b in zip(hooked, written)], "| logit scale", round(float(np.abs(written[0]).max()), 1))
print(summary(full_eval("fitted update, written into the weights", thinking=True)))
