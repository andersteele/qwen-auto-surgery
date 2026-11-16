# bin/lab lab/prc/baseline.py     (after session.py, on the original weights)
# The opening of each fresh-chat reply, and the residual stream after every layer at the last six
# positions of the prompt, for every question set. Then, outside the server:
#     .venv/bin/python lab/prc/directions.py runs/<run>/lab
t0 = time.time()
BASE, RESID = {}, {}
for name, qs in Q.items():
    rows, acts = [], []
    for q in qs:
        toks = chat(q)
        got = T.read(toks, last=6, kinds=("resid",))
        text = T.greedy(toks, 60)
        top = [int(i) for i in np.argsort(-got["logits"])[:5]]
        rows.append({"q": q, "opening": text, "kind": kind_of(text), "top": S.pieces(top), "n_tok": len(toks)})
        acts.append(got["resid"])
    BASE[name] = rows
    RESID[name] = np.stack(acts)           # (questions, layers, 6, d)
    np.save(OUT / f"resid_off_{name}.npy", RESID[name])
(OUT / "baseline_off.json").write_text(json.dumps(BASE, ensure_ascii=False, indent=1))
print("seconds", round(time.time() - t0))
for name, rows in BASE.items():
    print(name, kinds([r["opening"] for r in rows]))
for r in BASE["censored"]:
    print(f"{r['kind'][:3]} | {r['q'][:44]:44} | {r['opening'][:95]!r}")
