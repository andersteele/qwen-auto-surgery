# bin/lab lab/prc/gradients.py     (after session.py, on the original weights)
# Where the refusal is decided. For each Tiananmen question: the gradient of the refusal score
# with respect to the stream at every layer, what each layer's mixer and MLP wrote at the last six
# positions, and what each routed expert added to the score. Then, outside the server:
#     .venv/bin/python lab/prc/attribution.py runs/<run>/lab
#     .venv/bin/python lab/prc/grad_directions.py runs/<run>/lab
N = 6
t0 = time.time()
ctrl = {k: [] for k in ("attn", "mlp", "resid")}
for q in Q["elsewhere"]:                              # answered questions: the reference
    got = T.read(chat(q), last=N, kinds=("attn", "mlp", "resid"))
    for k in ctrl: ctrl[k].append(got[k])
CTRL = {k: np.stack(v).mean(0) for k, v in ctrl.items()}            # (layers, N, d)
del ctrl
TIAN = []
for q in Q["tiananmen"] + [Q["censored"][i] for i in (0, 1, 3, 4, 47)]:
    toks = chat(q)
    text = T.greedy(toks, 40)
    got = T.read(toks, last=N, kinds=("attn", "mlp", "resid"))
    val, g = T.grad(toks, refusal_score)
    TIAN.append({"q": q, "opening": text, "kind": kind_of(text), "score": val, "g": g[:, :, -N:].astype(np.float32),
                 "attn": got["attn"], "mlp": got["mlp"], "resid": got["resid"]})
print("seconds", round(time.time() - t0), "|", len(TIAN), "Tiananmen questions:", kinds([r["opening"] for r in TIAN]),
      "| mean refusal score", round(float(np.mean([r["score"] for r in TIAN])), 2))
np.savez(OUT / "tian_grad.npz", g=np.stack([r["g"] for r in TIAN]), attn=np.stack([r["attn"] for r in TIAN]),
         mlp=np.stack([r["mlp"] for r in TIAN]), resid=np.stack([r["resid"] for r in TIAN]),
         score=np.array([r["score"] for r in TIAN]), stock=np.array([STOCK in r["opening"] for r in TIAN]),
         ctrl_attn=CTRL["attn"], ctrl_mlp=CTRL["mlp"], ctrl_resid=CTRL["resid"])
(OUT / "tian_openings.json").write_text(json.dumps([{k: r[k] for k in ("q", "opening", "kind", "score")} for r in TIAN], ensure_ascii=False, indent=1))
cs = [float(refusal_score(mx.array(T.logits(chat(q)))[None]).item()) for q in Q["elsewhere"][:16]]
print("refusal score on 16 answered questions: mean", round(float(np.mean(cs)), 2), "max", round(max(cs), 2))

# Experts. (what a routed expert wrote) . (gradient there) = the nats it adds to the score, to first order.
LAYERS = list(range(12, 40))


def expert_attr(q):
    toks = chat(q)
    val, g = T.grad(toks, refusal_score)
    ex = T.experts(toks, layers=LAYERS, last=2)
    rows = []
    for L in LAYERS:
        d = ex[L]
        for pi, pos in enumerate((-2, -1)):                       # </think>, and the last position
            a = d["wrote"][pi] @ g[1, L, pos]
            rows += [(L, int(e), pi, float(v)) for e, v in zip(d["experts"][pi].tolist(), a.tolist())]
            rows.append((L, -1, pi, float(d["shared"][pi] @ g[1, L, pos])))   # -1: the shared expert
    return rows


EXA = {"refused": [expert_attr(r["q"]) for r in TIAN if STOCK in r["opening"]],
       "control": [expert_attr(q) for q in Q["elsewhere"][:24]]}
agg = {}
for grp, runs in EXA.items():
    for rows in runs:
        for L, e, pi, v in rows:
            agg.setdefault((L, e, pi), {"refused": [], "control": []})[grp].append(v)
nr, nc = len(EXA["refused"]), len(EXA["control"])
table = [{"L": L, "E": e, "pos": ("think", "last")[pi], "ref_n": len(a["refused"]), "ref_sum": sum(a["refused"]) / nr,
          "ctl_n": len(a["control"]), "ctl_sum": sum(a["control"]) / nc} for (L, e, pi), a in agg.items()]
(OUT / "expert_attr.json").write_text(json.dumps({"n_refused": nr, "n_control": nc, "table": table}))
table.sort(key=lambda r: -(r["ref_sum"] - r["ctl_sum"]))
print(f"\ntop 30 experts by (refused - control) nats per question   [E -1 = the shared expert]\n L   E   pos   | refused: routed/{nr}, nats | control: routed/{nc}, nats")
for r in table[:30]:
    print(f"{r['L']:3d} {r['E']:4d} {r['pos']:5} | {r['ref_n']:3d} {r['ref_sum']:+6.2f} | {r['ctl_n']:3d} {r['ctl_sum']:+6.2f}")
