"""Experiments the operator runs on the live model: read what each layer writes to the
residual stream, remove a direction from it during the forward pass, and work out which
weights write that direction.

Driven from the python tool, as the operator (``S`` is the tool's handle on the weights)::

    from surgery import lab
    T = lab.Tap(S)
    got = T.read(tokens, last=4)             # {"resid": (layers, 4, d), "logits": (vocab,), ...}
    with T.ablate(r):                        # r: (d,) or (n, d)
        text = T.greedy(tokens, 60)

Each layer computes ``h = x + mixer(norm(x))`` and ``out = h + mlp(norm(h))``, so three things
write to the stream: the token embedding, each layer's mixer (attention or the linear kind)
and each layer's MLP. Removing a unit direction r from each of those outputs, ``y - (y.r) r``,
is the same computation as replacing every matrix W that writes to the stream by
``(I - r r^T) W``: the embedding table, each mixer's output projection, and each expert's
down projection. So an ablation tried here is a weight edit that has not been written yet,
and ``Tap.writers`` lists the tensors it would change.

Nothing here is offered to the model as a tool.
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx
import numpy as np
from mlx_lm.models import qwen3_5
from mlx_lm.models.activations import swiglu

SITES = ("embed", "attn", "mlp")  # everything that writes to the stream
_TAP: "Tap | None" = None

# surgery.core wraps DecoderLayer.__call__ once; this wraps that, and a reload of this module
# must wrap the same function again, not its own earlier wrapper.
if not hasattr(qwen3_5, "_lab_layer_call_below"):
    qwen3_5._lab_layer_call_below = qwen3_5.DecoderLayer.__call__
_BELOW = qwen3_5._lab_layer_call_below


def _layer_call(self, x, mask=None, cache=None):
    tap = _TAP
    L = tap.layer_of.get(id(self)) if tap is not None else None
    if L is None:
        return _BELOW(self, x, mask=mask, cache=cache)
    if L == 0:
        x = tap._site("embed", 0, x)
    mixer = self.linear_attn if self.is_linear else self.self_attn
    r = tap._site("attn", L, mixer(self.input_layernorm(x), mask, cache))
    h = x + r
    m = tap._site("mlp", L, tap._mlp(L, self.mlp, self.post_attention_layernorm(h)))
    return tap._site("resid", L, h + m)


qwen3_5.DecoderLayer.__call__ = _layer_call


def unit_rows(R) -> np.ndarray:
    """(n, d) orthonormal rows spanning the given direction(s)."""
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))
    q, _ = np.linalg.qr(R.T)
    return np.ascontiguousarray(q.T[: R.shape[0]], dtype=np.float32)


class Tap:
    def __init__(self, S):
        self.s = getattr(S, "_s", S)
        self.layer_of = {id(l): i for i, l in enumerate(self.s.layers)}
        self.n_layers, self.d = self.s.n_layers, self.s.d_model
        self._ablate: dict | None = None
        self._add: dict | None = None
        self._rec: dict | None = None
        self._detail: dict | None = None
        self._delta = None  # (2, layers, T, d), added to the stream while a gradient is taken with respect to it
        self._lora: dict | None = None  # a low-rank update to the shared experts' down projections, being tried
        self._depth = 0

    # ------------------------------------------------------------ hooks --

    def _site(self, kind: str, L: int, y):
        a = self._ablate
        if a is not None:
            Q = a["by"].get((kind, L)) if "by" in a else (
                a["R"] if kind in a["sites"] and (a["layers"] is None or L in a["layers"]) else None)
            if Q is not None:
                y32 = y.astype(mx.float32)
                y = (y32 - (y32 @ Q.T) @ Q).astype(y.dtype)
        add = self._add
        if add is not None and kind == "resid" and L in add["v"]:
            v = add["v"][L].astype(y.dtype)
            if add["at"] is None:
                y = y + v
            elif y.shape[1] >= add["need"]:  # the pass that holds the whole prompt, not a later one-token step
                mask = np.zeros((1, y.shape[1], 1), dtype=np.float32)
                mask[0, list(add["at"]), 0] = 1.0
                y = y + mx.array(mask).astype(y.dtype) * v
        if self._delta is not None and kind in ("attn", "resid"):
            y = y + self._delta[0 if kind == "attn" else 1, L].astype(y.dtype)[None]
        rec = self._rec
        if rec is not None and kind in rec["kinds"]:
            rec["out"][(kind, L)] = y[0, -rec["last"]:].astype(mx.float32)
        return y

    def _mlp(self, L: int, moe, x):
        lo = self._lora
        if lo is not None and L in lo["index"]:
            # shared_expert.down_proj.weight + B^T A, written as what that adds to the block's output
            i = lo["index"][L]
            se = moe.shared_expert
            hid = swiglu(se.gate_proj(x), se.up_proj(x)).astype(mx.float32)  # (B, T, hidden)
            extra = (hid @ lo["A"][i].T) @ lo["B"][i]  # (B, T, d)
            gate = mx.sigmoid(moe.shared_expert_gate(x)).astype(mx.float32)
            return moe(x) + (gate * extra).astype(x.dtype)
        det = self._detail
        if det is None or L not in det["layers"]:
            return moe(x)
        # The expert block written out, so each routed expert's output can be read along `R`.
        gates = mx.softmax(moe.gate(x), axis=-1, precise=True)
        k = moe.top_k
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(moe, "norm_topk_prob", True):
            scores = scores / scores.sum(axis=-1, keepdims=True)
        sw = moe.switch_mlp
        xe = mx.expand_dims(x, (-2, -3))
        act = swiglu(sw.gate_proj(xe, inds), sw.up_proj(xe, inds))  # (B, T, k, 1, H)
        y = sw.down_proj(act, inds).squeeze(-2) * scores[..., None]  # (B, T, k, d)
        shared = mx.sigmoid(moe.shared_expert_gate(x)) * moe.shared_expert(x)
        n = det["last"]
        se = moe.shared_expert
        det["out"][L] = {
            "experts": inds[0, -n:],  # (n, k)
            "weights": scores[0, -n:].astype(mx.float32),
            "wrote": y[0, -n:].astype(mx.float32),  # (n, k, d): each routed expert's weighted output
            "shared": shared[0, -n:].astype(mx.float32),  # (n, d)
            "shared_hidden": swiglu(se.gate_proj(x), se.up_proj(x))[0, -n:].astype(mx.float32),  # (n, hidden)
            "shared_gate": mx.sigmoid(moe.shared_expert_gate(x))[0, -n:, 0].astype(mx.float32),  # (n,)
            "acts": act[0, -n:, :, 0].astype(mx.float32) if det["acts"] else None,  # (n, k, H)
        }
        return y.sum(axis=-2) + shared

    @contextmanager
    def _on(self):
        global _TAP
        self._depth += 1
        _TAP = self
        try:
            yield self
        finally:
            self._depth -= 1
            if self._depth == 0:
                _TAP = None

    # ----------------------------------------------------- interventions --

    @contextmanager
    def ablate(self, R, sites=SITES, layers=None):
        """Remove the direction(s) R from what the named sites write, in the given layers (default all)."""
        Q = mx.array(unit_rows(R))
        prev = self._ablate
        self._ablate = {"R": Q, "sites": set(sites), "layers": None if layers is None else set(layers)}
        try:
            with self._on():
                yield self
        finally:
            self._ablate = prev

    @contextmanager
    def ablate_each(self, by: dict):
        """Remove a different direction (or several) from each writer: by[(site, L)] is (d,) or (n, d),
        with site "attn" or "mlp" (or "embed", with L = 0). The weight edit this stands for is
        ``W <- (I - R^T R) W`` on that one writer."""
        prev = self._ablate
        self._ablate = {"by": {(k, int(L)): mx.array(unit_rows(R)) for (k, L), R in by.items()}}
        try:
            with self._on():
                yield self
        finally:
            self._ablate = prev

    @contextmanager
    def lora(self, layers, A, B):
        """Try the weight update ``layers.L.mlp.shared_expert.down_proj.weight += B[i].T @ A[i]`` for the
        i-th of ``layers``, without writing it. A is (n, r, hidden) and B is (n, r, d); both may be
        arrays a gradient is being taken with respect to. ``shared_update`` gives the matrices to write."""
        prev = self._lora
        self._lora = {"index": {int(L): i for i, L in enumerate(layers)},
                      "A": A if isinstance(A, mx.array) else mx.array(np.asarray(A, dtype=np.float32)),
                      "B": B if isinstance(B, mx.array) else mx.array(np.asarray(B, dtype=np.float32))}
        try:
            with self._on():
                yield self
        finally:
            self._lora = prev

    @staticmethod
    def shared_path(L: int) -> str:
        return f"layers.{int(L)}.mlp.shared_expert.down_proj.weight"

    @staticmethod
    def shared_update(layers, A, B) -> dict:
        """{path: delta (d, hidden)} for the update ``lora`` tries."""
        A, B = np.asarray(A, dtype=np.float32), np.asarray(B, dtype=np.float32)
        return {Tap.shared_path(L): B[i].T @ A[i] for i, L in enumerate(layers)}

    @contextmanager
    def add(self, vectors: dict, at=None):
        """Add vectors[L] to the stream after layer L: at every position, or only at the positions
        ``at`` (negative indices, counted from the end of the prompt)."""
        prev = self._add
        self._add = {"v": {int(L): mx.array(np.asarray(v, dtype=np.float32)) for L, v in vectors.items()},
                     "at": None if at is None else tuple(int(a) for a in at),
                     "need": 0 if at is None else max(abs(int(a)) for a in at) + 1}
        try:
            with self._on():
                yield self
        finally:
            self._add = prev

    # ------------------------------------------------------------ reading --

    def logits(self, tokens) -> np.ndarray:
        """Next-token logits after ``tokens``, with whatever intervention is in force."""
        out = self.s.model(mx.array([list(tokens)]))[0, -1].astype(mx.float32)
        mx.eval(out)
        return np.array(out)

    def read(self, tokens, last: int = 1, kinds=("resid",)) -> dict:
        """One forward pass. For each kind in ("embed", "attn", "mlp", "resid"): what that site wrote
        (for "resid", the stream after the layer) at the last ``last`` positions, as (layers, last, d)."""
        prev = self._rec
        self._rec = {"kinds": set(kinds), "last": int(last), "out": {}}
        try:
            with self._on():
                lg = self.s.model(mx.array([list(tokens)]))[0, -1].astype(mx.float32)
                got = self._rec["out"]
                mx.eval(lg, *got.values())
        finally:
            rec, self._rec = self._rec, prev
        res = {"logits": np.array(lg)}
        for kind in kinds:
            layers = [0] if kind == "embed" else range(self.n_layers)
            res[kind] = np.stack([np.array(rec["out"][(kind, L)]) for L in layers])
        return res

    def grad(self, tokens, score) -> tuple[float, np.ndarray]:
        """score(logits) for one prompt, and its gradient with respect to the residual stream at every
        position, as (2, layers, T, d): [0] is the stream in the middle of each layer (after the mixer
        has written, before the MLP reads) and [1] the stream after the layer. So [0, L] is also the
        gradient with respect to what layer L's mixer writes, and [1, L] to what its MLP writes.
        ``logits`` is (T, vocab), float32.

        The linear-attention layers run their plain-operations path for this (their fast kernel has
        no gradient), which the model selects in training mode; nothing is trained.
        """
        toks = mx.array([list(tokens)])

        def f(delta):
            self._delta = delta
            with self._on():
                return score(self.s.model(toks)[0].astype(mx.float32))

        self.s.model.train()
        try:
            val, g = mx.value_and_grad(f)(mx.zeros((2, self.n_layers, len(tokens), self.d)))
            mx.eval(val, g)
        finally:
            self._delta = None
            self.s.model.eval()
        return float(val.item()), np.array(g)

    def experts(self, tokens, layers=None, last: int = 1, acts: bool = False) -> dict:
        """One forward pass that also reads, per layer, which experts were routed at the last positions,
        with what weight, and what each wrote to the stream (weight included): {L: {"experts": (n, k),
        "weights": (n, k), "wrote": (n, k, d), "shared": (n, d), "shared_hidden": (n, hidden),
        "shared_gate": (n,), "acts": (n, k, neurons) or None}}. The shared expert writes
        ``shared_gate * (down_proj @ shared_hidden)``."""
        prev = self._detail
        self._detail = {"layers": set(range(self.n_layers) if layers is None else layers), "last": int(last),
                        "acts": bool(acts), "out": {}}
        try:
            with self._on():
                lg = self.s.model(mx.array([list(tokens)]))[0, -1]
                got = self._detail["out"]
                mx.eval(lg, *[v for d in got.values() for v in d.values() if v is not None])
        finally:
            det, self._detail = self._detail, prev
        return {L: {k: (None if v is None else np.array(v)) for k, v in d.items()} for L, d in det["out"].items()}

    def greedy(self, tokens, max_tokens: int = 48) -> str:
        return self.s.tok.decode(self.s.greedy(list(tokens), max_tokens))

    # ------------------------------------------------------------ weights --

    def writers(self) -> list[tuple[str, str, tuple]]:
        """(site, path, shape) of every tensor that writes to the residual stream."""
        out = [("embed", "embed_tokens.weight", None)]
        for L, layer in enumerate(self.s.layers):
            mixer = "linear_attn.out_proj" if layer.is_linear else "self_attn.o_proj"
            out.append(("attn", f"layers.{L}.{mixer}.weight", None))
            out.append(("mlp", f"layers.{L}.mlp.switch_mlp.down_proj.weight", None))
            out.append(("mlp", f"layers.{L}.mlp.shared_expert.down_proj.weight", None))
        shapes = {name: shape for name, shape, _ in self.s.list_weights()}
        return [(site, path, shapes.get(self.s._full(path))) for site, path, _ in out]
