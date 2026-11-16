# bin/lab lab/prc/steer.py     (after gradients.py and grad_directions.py, on the original weights)
# Three tries at acting on what the gradients found, read on Tiananmen questions the directions
# were not fitted on:
#   1. remove, from each layer's writers, the direction the refusal score reads there;
#   2. push against that direction at one layer (steering, which is not a weight edit);
#   3. the push together with removing the layer-16 censored-versus-control direction.
from contextlib import ExitStack

GD = dict(np.load(OUT / "grad_dirs.npz"))
C = dict(np.load(OUT / "candidates.npz"))
tq_all = Q["tiananmen"] + [Q["censored"][i] for i in (0, 1, 3, 4, 47)]
tian_test = [q for q, f in zip(tq_all, GD["fit"]) if not f]
RES = {}


def by_layers(layers, sites=("mlp",), positions=(4, 5)):         # 4 is </think>, 5 the last position
    by = {}
    for L in layers:
        for site in sites:
            vs = [GD[site][L, p] for p in positions if np.isfinite(GD[site][L, p]).all()]
            if vs: by[(site, L)] = np.stack(vs)
    return by


def show(name, ctx, n=8, tokens=80):
    with ctx:
        rows = [T.greedy(chat(q), tokens) for q in tian_test[:n]]
        d = drift()
    RES[name] = {"replies": rows, "drift": d}
    print(f"\n== {name}: {kinds(rows)} | drift of other replies: median {d[0]:.3f}, max {d[1]:.3f}")
    for q, t in zip(tian_test, rows): print(f"   {q[:38]:38} | {t[:220]!r}")


def both(R, L, vec, at):
    st = ExitStack(); st.enter_context(T.ablate(R)); st.enter_context(T.add({L: vec}, at=at)); return st


show("1. gradient direction removed from the MLPs, L17-37", T.ablate_each(by_layers(range(17, 38), positions=(4,))))
show("1. both directions removed from MLPs and mixers, all layers", T.ablate_each(by_layers(range(40), sites=("mlp", "attn"))))
for L in (12, 20, 28):
    show(f"2. push of 1.5 against the gradient at layer {L}, at </think> only", T.add({L: -1.5 * GD["mlp"][L, 4]}, at=(-2,)))
show("2. push of 1.5 against the gradient at layer 20, every position", T.add({20: -1.5 * GD["mlp"][20, 4]}))
r16 = C["cens-both"][16, -1]
show("3. layer-16 direction removed", T.ablate(r16))
show("3. layer-16 direction removed, and the push at layer 20", both(r16, 20, -1.5 * GD["mlp"][20, 4], (-2,)))
(OUT / "steer.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1))
