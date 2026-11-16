# bin/lab lab/prc/sweep.py     (after baseline.py and directions.py)
# Remove one candidate direction at a time from everything written to the stream, and see what
# held-back censored questions get: the kind of reply, the log-probability of the stock refusal,
# and how far the first token of unrelated replies moves.
C = dict(np.load(OUT / "candidates.npz"))          # name -> (layers, 6, d)


def evaluate(R):
    res = {"stock_lp": [], "openings": []}
    with T.ablate(R):
        for q in val_c:
            toks = chat(q)
            res["stock_lp"].append(sum(S._s.logprob(toks, STOCK)))
            res["openings"].append(T.greedy(toks, 48))
        res["drift"] = drift()
    res["kinds"] = kinds(res["openings"])
    return res


t0 = time.time()
SWEEP = {"none": {"stock_lp": [sum(S._s.logprob(chat(q), STOCK)) for q in val_c], "openings": [T.greedy(chat(q), 48) for q in val_c], "drift": (0.0, 0.0)}}
SWEEP["none"]["kinds"] = kinds(SWEEP["none"]["openings"])
plan = [(f"cens-both@L{L}", C["cens-both"][L, -1]) for L in (8, 12, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36)]
for L in (16, 20, 24, 28):
    plan += [(f"refused-else@L{L}", C["refused-else"][L, -1]), (f"cens-else@L{L}", C["cens-else"][L, -1])]
stack = np.stack([C["cens-both"][L, -1] / np.linalg.norm(C["cens-both"][L, -1]) for L in range(12, 37)])
Vt = np.linalg.svd(stack, full_matrices=False)[2]
plan += [(f"cens-both svd{k} (L12-36)", Vt[:k]) for k in (1, 2, 4, 8)]
for name, R in plan:
    SWEEP[name] = evaluate(R)
    (OUT / "sweep.json").write_text(json.dumps(SWEEP, ensure_ascii=False, indent=1))
print("seconds", round(time.time() - t0))
refused = [i for i, t in enumerate(SWEEP["none"]["openings"]) if STOCK in t]
print(f"{'direction removed':28} | kinds on {len(val_c)} held-back censored | mean logP(stock refusal): all / the {len(refused)} refused | drift median, max")
for name, r in SWEEP.items():
    lp = np.array(r["stock_lp"])
    print(f"{name:28} | {str(r['kinds']):52} | {lp.mean():7.1f} / {lp[refused].mean():7.1f} | {r['drift'][0]:.3f} {r['drift'][1]:.3f}")
