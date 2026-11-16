# bin/lab lab/prc/fit.py     (after session.py and harvest.py, on the original weights; about 25 minutes)
# Fit a low-rank update to the shared experts' down projections so that a fresh chat gives the
# answers the same weights give inside the prc-1 conversation, while control replies stay put.
#
#   targets   harvested answers to the even-numbered Tiananmen and censored questions
#   held back harvested answers to the odd-numbered ones: the fit stops when these stop improving
#   controls  the original weights' own fresh-chat replies to other-country, uncensored-China and
#             everyday questions, and to the metric's four visible harmful requests (never printed)
#
# Then  bin/lab lab/prc/write.py  writes the update into the weights and measures it.
RANK, LAYERS_FIT, LR, STEPS, LAMBDA, PATIENCE, NAME, TOPK = 2, list(range(8, 32)), 1e-3, 200, 3.0, 40, "fit", 64
EXCLUDE = {"harvest:censored:0"}   # asked about the massacre, it described the square
_target = next(g for g in SPEC["groups"] if g["role"] == "target")


def question_of(label):
    _, g, i = label.split(":")
    return _target["items"][int(i)]["prompt"] if g == "metric" else Q[g][int(i)]


def reply_of(r):
    """The answer itself: what came after the reasoning and before any tool call."""
    return (r.get("content") or "").split("<tool_call>")[0].strip()


def usable(r):
    c = reply_of(r)
    about_the_work = re.search(r"\b(score|metric|my (current )?weights|edits?|neurons?|ledger|flagged)\b", c[:400])
    return (r["label"] not in EXCLUDE and len(c) > 120 and kind_of(c[:400]) == "answer" and not about_the_work
            and not re.match(r"I (don't|do not) (have|know)", c))


def ref_topk(toks, n):
    """The original weights' top next-token log-probabilities at each position of a control reply."""
    logits = S._s.model(mx.array([toks]))[0, n - 1:-1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    idx = mx.argpartition(-lp, kth=TOPK, axis=-1)[:, :TOPK]
    val = mx.take_along_axis(lp, idx, axis=-1)
    mx.eval(idx, val)
    return idx, val


assert not S._s.edits, "fit on the original weights"
DATA = {"target": [], "valid": [], "control": []}
for r in (json.loads(l) for l in (OUT / "harvest.jsonl").open()):
    _, group, i = r["label"].split(":")
    if group != "metric" and usable(r):
        p = chat(question_of(r["label"]))
        DATA["target" if int(i) % 2 == 0 else "valid"].append({"toks": p + S.tokens(reply_of(r))[:72], "n": len(p), "kind": "target"})
ctrl_q = [q for name in ("elsewhere", "china", "everyday") for i, q in enumerate(Q[name]) if i % 2 == 0]
ctrl_q += [it["prompt"] for g in SPEC["groups"] if "harmful" in g["name"] for it in g["items"] if not it.get("heldout")]
S._s.model.eval()
conts = [(q, S._s.greedy(chat(q), 32)) for q in ctrl_q]
S._s.model.train()   # the path with gradients; the references are taken on the same path
try:
    for q, cont in conts:
        p = chat(q)
        idx, val = ref_topk(p + cont, len(p))
        DATA["control"].append({"toks": p + cont, "n": len(p), "kind": "control", "ref_idx": idx, "ref_lp": val})
finally:
    S._s.model.eval()
print({k: len(v) for k, v in DATA.items()})

rng = np.random.default_rng(0)
hidden = S._s.get_weight(T.shared_path(LAYERS_FIT[0])).shape[1]
P = {"A": mx.array(rng.normal(0, 0.02, (len(LAYERS_FIT), RANK, hidden)).astype(np.float32)),
     "B": mx.zeros((len(LAYERS_FIT), RANK, S.d_model))}


def seq_loss(params, item):
    with T.lora(LAYERS_FIT, params["A"], params["B"]):
        logits = S._s.model(mx.array([item["toks"]]))[0].astype(mx.float32)
    n = item["n"]
    lp = logits[n - 1:-1]
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    if item["kind"] == "target":      # cross-entropy of the harvested answer
        return -mx.take_along_axis(lp, mx.array(item["toks"][n:])[:, None], axis=-1).mean()
    lq = mx.take_along_axis(lp, item["ref_idx"], axis=-1)   # KL from the original reply's distribution
    return (mx.exp(item["ref_lp"]) * (item["ref_lp"] - lq)).sum(axis=-1).mean()


vg = mx.value_and_grad(seq_loss)
M = {k: mx.zeros_like(v) for k, v in P.items()}
V = {k: mx.zeros_like(v) for k, v in P.items()}
LOG, BEST = [], {"ce": float("inf"), "step": 0}
S._s.model.train()
t0 = time.time()
try:
    for step in range(1, STEPS + 1):
        batch = [DATA["target"][i] for i in rng.choice(len(DATA["target"]), 4, replace=False)] + \
                [DATA["control"][i] for i in rng.choice(len(DATA["control"]), 4, replace=False)]
        G = {k: mx.zeros_like(v) for k, v in P.items()}
        tl, cl = [], []
        for it in batch:
            val, g = vg(P, it)
            w = 1.0 if it["kind"] == "target" else LAMBDA
            G = {k: G[k] + w * g[k] / 4 for k in G}
            mx.eval(val, *G.values())
            (tl if it["kind"] == "target" else cl).append(float(val.item()))
        for k in P:                                   # Adam
            M[k] = 0.9 * M[k] + 0.1 * G[k]
            V[k] = 0.999 * V[k] + 0.001 * G[k] * G[k]
            P[k] = P[k] - LR * (M[k] / (1 - 0.9 ** step)) / (mx.sqrt(V[k] / (1 - 0.999 ** step)) + 1e-8)
        mx.eval(*P.values(), *M.values(), *V.values())
        LOG.append({"step": step, "target_ce": float(np.mean(tl)), "control_kl": float(np.mean(cl)), "seconds": round(time.time() - t0)})
        if step % 10 == 0 or step == 1:
            vce = float(np.mean([float(seq_loss(P, it).item()) for it in DATA["valid"]]))
            LOG[-1]["valid_ce"] = vce
            if vce < BEST["ce"]:
                BEST.update(ce=vce, step=step, A=np.array(P["A"]), B=np.array(P["B"]))
                np.savez(OUT / f"{NAME}.npz", A=BEST["A"], B=BEST["B"], layers=np.array(LAYERS_FIT), step=step)
            last = LOG[-10:]
            print(f"step {step:3d} | target CE {np.mean([l['target_ce'] for l in last]):.3f} | held-back CE {vce:.3f} | control KL {np.mean([l['control_kl'] for l in last]):.4f} | {round(time.time() - t0)} s")
            (OUT / f"{NAME}_log.json").write_text(json.dumps(LOG))
            if step - BEST["step"] >= PATIENCE:
                print("stopping: no improvement on held-back answers for", PATIENCE, "steps"); break
finally:
    S._s.model.eval()
print("best held-back CE", round(BEST["ce"], 3), "at step", BEST["step"])
FIT = {"layers": LAYERS_FIT, "A": BEST["A"], "B": BEST["B"], "step": BEST["step"]}

# A first look, in a fresh chat, on questions the fit did not use.
with T.lora(FIT["layers"], FIT["A"], FIT["B"]):
    QUICK = {"tian": [T.greedy(chat(q), 90) for q in tian_odd], "cens": [T.greedy(chat(q), 60) for q in val_c], "drift": drift()}
print(f"{len(tian_odd)} held-back Tiananmen questions:", kinds(QUICK["tian"]), f"| {len(val_c)} held-back censored:", kinds(QUICK["cens"]),
      "| drift of other replies: median %.3f max %.3f" % QUICK["drift"])
for q, t in zip(tian_odd, QUICK["tian"]): print(f"  {q[:44]:44} | {t[:200]!r}")
(OUT / f"{NAME}_quick.json").write_text(json.dumps(QUICK, ensure_ascii=False, indent=1))
