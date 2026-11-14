"""Read, probe and edit the weights of a live MLX Qwen3.5/3.6 MoE model.

Everything here runs on the thread that also runs generation, so an edit
lands between two forward passes and the next token the model produces is
computed by the edited weights.

Neuron addressing follows Goodfire's atlas: layer / expert / neuron.
For neuron (L, E, N), with ``sw = layers[L].mlp.switch_mlp``::

    gate = sw.gate_proj.weight[E, N, :]   # reads the residual stream (2048)
    up   = sw.up_proj.weight[E, N, :]     # reads the residual stream (2048)
    out  = sw.down_proj.weight[E, :, N]   # writes to the residual stream (2048)

    activation = silu(gate . x) * (up . x)        # signed; "+" and "-" poles
    contribution = routing_weight(E) * activation * out   (only if E is in the top-k)

The base model on disk is never modified, and the 8-bit weights in memory
stay as they are. An edited neuron vector is stored as base + delta, the delta
a float32 vector applied in the forward pass, so an edit changes exactly the
values it names, costs a few kilobytes, and can be undone exactly. Replacing a
whole tensor swaps the module for a float32 one instead.
"""

from __future__ import annotations

import ast
import ctypes
import gc
import json
import math
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.generate import stream_generate
from mlx_lm.models import qwen3_5, qwen3_next
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.sample_utils import make_sampler

from .atlas import Atlas, fmt_neuron, parse_neuron

PREFIX = "language_model.model."
PROJ = {"gate": "gate_proj", "up": "up_proj", "out": "down_proj"}
MAX_PROBE_TOKENS = 4096
PYTHON_TIMEOUT = 300.0  # seconds one python tool call may run
MAX_BATCH = 2000  # neurons one scale_neurons call may edit
PY_RESERVED = ("S", "mx", "np", "math", "print")  # names the python tool provides


def control_tokens(tok) -> list[str]:
    """The strings the tokenizer reads as one control token each (end of message, end of thinking, ...)."""
    try:
        vocab = tok.get_added_vocab()
    except Exception:  # noqa: BLE001 - a tokenizer without added tokens has none to guard
        return []
    return sorted((t for t in vocab if len(t) > 2 and t[0] == "<" and t[-1] == ">"), key=len, reverse=True)


class PythonTimeout(BaseException):
    """Raised inside the model's Python code when it runs too long."""


# The time limit has to hold even when the model's code swallows exceptions: a bare `except:`
# inside a loop caught the one interrupt a call got, and that call ran for 83 minutes. Once a
# call is out of time, every line of the model's code that starts to run raises again, so a
# handler that catches the interrupt is stopped on its own next line.
_LIMIT_TOOL = 4  # a sys.monitoring tool id; 0, 1, 2 and 5 are reserved for debuggers, coverage, profilers, optimizers
_limit = {"over": False}


def _limit_line(code, line):
    if code.co_filename != "<surgery>" or not _limit["over"]:
        return sys.monitoring.DISABLE  # do not report this line again until events are restarted
    raise PythonTimeout


if sys.monitoring.get_tool(_LIMIT_TOOL) is None:
    sys.monitoring.use_tool_id(_LIMIT_TOOL, "surgery python time limit")
sys.monitoring.register_callback(_LIMIT_TOOL, sys.monitoring.events.LINE, _limit_line)

# --------------------------------------------------------------------------
# Activation capture: class-level wrappers that are inert unless a Recorder is
# installed for the duration of one forward pass.
# --------------------------------------------------------------------------

_REC: "Recorder | None" = None
_ORIG_MOE_CALL = qwen3_next.Qwen3NextSparseMoeBlock.__call__
_ORIG_LAYER_CALL = qwen3_5.DecoderLayer.__call__


class Recorder:
    def __init__(self, targets=None, agg_layers=(), resid_layers=()):
        self.targets: dict[int, list[tuple[int, int]]] = targets or {}
        self.agg_layers = set(agg_layers)
        self.resid_layers = set(resid_layers)
        self.moe_layer: dict[int, int] = {}
        self.block_layer: dict[int, int] = {}
        self.acts: dict[tuple[int, int, int], mx.array] = {}
        self.route: dict[tuple[int, int], mx.array] = {}
        self.agg: dict[int, dict[str, mx.array]] = {}
        self.resid: dict[int, mx.array] = {}
        self.ntok = 0
        self.mask: mx.array | None = None  # (T,) 1.0 at the positions the aggregates should count

    def pending(self) -> list[mx.array]:
        return list(self.acts.values()) + list(self.route.values()) + list(self.resid.values())


def _moe_call(self, x):
    rec = _REC
    L = rec.moe_layer.get(id(self)) if rec is not None else None
    if L is None or (L not in rec.targets and L not in rec.agg_layers):
        return _ORIG_MOE_CALL(self, x)

    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if getattr(self, "norm_topk_prob", True):
        scores = scores / scores.sum(axis=-1, keepdims=True)

    sw = self.switch_mlp
    xe = mx.expand_dims(x, (-2, -3))
    act = swiglu(sw.gate_proj(xe, inds), sw.up_proj(xe, inds))  # (B, T, k, 1, H)
    y = sw.down_proj(act, inds).squeeze(-2)
    y = (y * scores[..., None]).sum(axis=-2)
    shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    act = act.squeeze(-2)  # (B, T, k, H)

    for E, N in rec.targets.get(L, ()):
        m = inds == E
        rec.acts[(L, E, N)] = (act[..., N] * m).sum(axis=-1)[0].astype(mx.float32)
        rec.route[(L, E)] = (scores * m).sum(axis=-1)[0].astype(mx.float32)

    if L in rec.agg_layers:
        H = act.shape[-1]
        w = (act * scores[..., None]).astype(mx.float32)
        count = mx.ones(inds.shape)
        if rec.mask is not None:
            keep = rec.mask.reshape(1, -1, 1)
            w, count = w * keep[..., None], count * keep
        w = w.reshape(-1, H)
        flat = inds.reshape(-1)
        nE = self.num_experts
        pos = mx.zeros((nE, H)).at[flat].add(mx.maximum(w, 0))
        neg = mx.zeros((nE, H)).at[flat].add(mx.maximum(-w, 0))
        routed = mx.zeros((nE,)).at[flat].add(count.reshape(-1))
        mx.eval(pos, neg, routed)
        a = rec.agg.get(L)
        if a is None:
            rec.agg[L] = {"pos": pos, "neg": neg, "routed": routed}
        else:
            a["pos"], a["neg"], a["routed"] = a["pos"] + pos, a["neg"] + neg, a["routed"] + routed
            mx.eval(a["pos"], a["neg"], a["routed"])

    return y + shared


def _layer_call(self, x, mask=None, cache=None):
    out = _ORIG_LAYER_CALL(self, x, mask=mask, cache=cache)
    rec = _REC
    if rec is not None:
        L = rec.block_layer.get(id(self))
        if L is not None and L in rec.resid_layers:
            rec.resid[L] = out[0].astype(mx.float32)
    return out


qwen3_next.Qwen3NextSparseMoeBlock.__call__ = _moe_call
qwen3_5.DecoderLayer.__call__ = _layer_call


# --------------------------------------------------------------------------
# How edits are held in memory.
# --------------------------------------------------------------------------


class Patch:
    """Float32 deltas to single neurons of one quantized SwitchLinear.

    cols[(E, N)] is added to weight[E, :, N] and rows[(E, N)] to weight[E, N, :].
    """

    def __init__(self):
        self.cols: dict[tuple[int, int], mx.array] = {}
        self.rows: dict[tuple[int, int], mx.array] = {}
        self.packed: tuple | None = None

    def __len__(self) -> int:
        return len(self.cols) + len(self.rows)

    def pack(self, out_dims: int):
        """Stack the deltas so the forward pass applies all of them with two matmuls."""
        col = row = None
        if self.cols:
            keys = list(self.cols)
            col = (mx.array([k[0] for k in keys]), mx.array([k[1] for k in keys]),
                   mx.stack([self.cols[k] for k in keys]))
        if self.rows:
            keys = list(self.rows)
            onehot = np.zeros((len(keys), out_dims), dtype=np.float32)
            onehot[np.arange(len(keys)), [k[1] for k in keys]] = 1.0
            row = (mx.array([k[0] for k in keys]), mx.stack([self.rows[k] for k in keys]).T, mx.array(onehot))
        self.packed = (col, row) if (col or row) else None
        if self.packed:
            mx.eval(*[a for part in self.packed if part for a in part])


class PatchedSwitchLinear(nn.Module):
    """A quantized SwitchLinear whose effective weight is base + Patch.

    With an empty patch the output is the base module's output, bit for bit.
    """

    def __init__(self, base: QuantizedSwitchLinear):
        super().__init__()
        self.base = base
        self._patch = Patch()

    @property
    def patch(self) -> Patch:
        return self._patch

    def __call__(self, x, indices, sorted_indices=False):
        y = self.base(x, indices, sorted_indices=sorted_indices)
        packed = self._patch.packed
        if packed is None:
            return y
        col, row = packed
        x32 = x.astype(mx.float32)
        hit = indices[..., None]  # which (token, slot) pairs go to which expert
        c = None
        if col is not None:
            E, N, D = col  # weight[E, :, N] += D  =>  y += x[N] * D
            c = ((hit == E) * mx.take(x32, N, axis=-1).squeeze(-2)) @ D
        if row is not None:
            E, Rt, onehot = row  # weight[E, N, :] += R  =>  y[N] += x . R
            r = ((hit == E) * (x32 @ Rt).squeeze(-2)) @ onehot
            c = r if c is None else c + r
        return y + mx.expand_dims(c, -2).astype(y.dtype)


class ThawedSwitchLinear(SwitchLinear):
    """Float32 weights; the output is cast back to the activation dtype."""

    def __call__(self, x, indices, sorted_indices=False):
        y = mx.gather_mm(
            x.astype(mx.float32), self["weight"].swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices
        )
        if "bias" in self:
            y = y + mx.expand_dims(self["bias"][indices], -2)
        return y.astype(x.dtype)


class ThawedLinear(nn.Linear):
    def __call__(self, x):
        y = x.astype(mx.float32) @ self["weight"].T
        if "bias" in self:
            y = y + self["bias"]
        return y.astype(x.dtype)


class ThawedEmbedding(nn.Embedding):
    out_dtype = mx.float32

    def __call__(self, x):
        return self["weight"][x].astype(self.out_dtype)

    def as_linear(self, x):
        return (x.astype(mx.float32) @ self["weight"].T).astype(x.dtype)


@dataclass
class Edit:
    seq: int
    op: str
    target: str
    detail: dict
    undo: list[Callable[[], None]] = field(default_factory=list)  # run in reverse to revert
    held: int = 0  # bytes of replaced tensors kept alive so the edit can be undone


class SurgeryError(Exception):
    """An error whose message is meant to be read by the model."""


class Surgery:
    def __init__(self, model, tokenizer, run_dir: Path, atlas: Atlas | None = None, mem_budget_gb: float = 6.0):
        self.model = model
        self.tok = tokenizer
        self.text = model.language_model
        self.layers = list(self.text.model.layers)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.atlas = atlas or Atlas()
        self.mem_budget = int(mem_budget_gb * 1e9)
        # Control tokens are shown to the model as ‹...› and read back from that form (see show_safe).
        self._ctl = control_tokens(tokenizer)
        self._ctl_re = re.compile("|".join(map(re.escape, self._ctl))) if self._ctl else None
        self._safe_re = re.compile("|".join(re.escape(f"‹{t[1:-1]}›") for t in self._ctl)) if self._ctl else None

        a = self.text.args
        self.n_layers = len(self.layers)
        self.n_experts = a.num_experts
        self.n_neurons = a.moe_intermediate_size
        self.d_model = a.hidden_size
        self.top_k = a.num_experts_per_tok
        self.act_dtype = self.text.model.embed_tokens(mx.array([[0]])).dtype

        self._moe_layer = {id(l.mlp): i for i, l in enumerate(self.layers)}
        self._block_layer = {id(l): i for i, l in enumerate(self.layers)}
        self._patched: dict[str, PatchedSwitchLinear] = {}  # module path -> wrapper holding neuron deltas
        self._float_bytes: dict[str, int] = {}  # module path -> size of a float32 replacement
        self.session = time.strftime("%Y%m%d-%H%M%S")
        self.edits: list[Edit] = []
        self._seq = 0
        self.dirty = False  # set when weights changed; the server then drops its prompt cache
        self._py_ns: dict[str, Any] | None = None
        self.python_timeout = PYTHON_TIMEOUT
        self._log({"op": "server_start"})

    # ------------------------------------------------------------- paths --

    def _full(self, path: str) -> str:
        path = path.strip().strip(".")
        if path.startswith("language_model."):
            return path
        if path.startswith("model."):
            return "language_model." + path
        if path.startswith("lm_head"):
            return "language_model." + path
        return PREFIX + path

    def _resolve(self, mod_path: str):
        """Return (parent, key, module) for a dotted module path."""
        obj: Any = self.model
        parent, key = None, None
        for part in self._full(mod_path).split("."):
            parent, key = obj, part
            try:
                if isinstance(obj, (list, tuple)):
                    obj = obj[int(part)]
                elif isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    obj = getattr(obj, part)
            except (AttributeError, KeyError, IndexError, ValueError):
                raise SurgeryError(f"no such module or weight: {mod_path!r} (failed at {part!r})") from None
        return parent, key, obj

    @staticmethod
    def _assign(parent, key, value):
        if isinstance(parent, list):
            parent[int(key)] = value
        elif isinstance(parent, dict):
            parent[key] = value
        else:
            setattr(parent, key, value)

    @staticmethod
    def _is_quantized(mod) -> bool:
        return isinstance(mod, (QuantizedSwitchLinear, nn.QuantizedLinear, nn.QuantizedEmbedding))

    @staticmethod
    def _dequant(mod, index=None) -> mx.array:
        w, s, b = mod["weight"], mod["scales"], mod.get("biases")
        if index is not None:
            w, s, b = w[index], s[index], (b[index] if b is not None else None)
        return mx.dequantize(w, s, b, group_size=mod.group_size, bits=mod.bits, mode=mod.mode).astype(mx.float32)

    def _float_shape(self, mod) -> tuple:
        if isinstance(mod, PatchedSwitchLinear):
            mod = mod.base
        if isinstance(mod, QuantizedSwitchLinear):
            return (mod.num_experts, mod.output_dims, mod.input_dims)
        w = mod["weight"]
        if self._is_quantized(mod):
            return (*w.shape[:-1], mod["scales"].shape[-1] * mod.group_size)
        return tuple(w.shape)

    def _budget_used(self) -> int:
        return sum(self._float_bytes.values()) + sum(e.held for e in self.edits)

    def _kernel_arrays(self) -> list[tuple[str, int]]:
        """(name, bytes) for each python-tool variable that holds arrays, largest first."""
        seen: set[int] = set()

        def size(v, depth: int = 0) -> int:
            if isinstance(v, (mx.array, np.ndarray)):
                if id(v) in seen:
                    return 0
                seen.add(id(v))
                return int(v.nbytes)
            if depth < 3 and isinstance(v, (list, tuple)):
                return sum(size(x, depth + 1) for x in v)
            if depth < 3 and isinstance(v, dict):
                return sum(size(x, depth + 1) for x in v.values())
            return 0

        held = [(k, size(v)) for k, v in list((self._py_ns or {}).items()) if k not in PY_RESERVED and not k.startswith("__")]
        return sorted(((k, n) for k, n in held if n), key=lambda t: -t[1])  # stable: older first among equals

    @staticmethod
    def _names(held: list[tuple[str, int]], limit: int = 8) -> str:
        shown = ", ".join(f"{k} {n / 1e9:.2f}" for k, n in held[:limit])
        return shown + (f", and {len(held) - limit} smaller" if len(held) > limit else "")

    def _charge(self, nbytes: int, what: str):
        """Whole-tensor edits and the arrays held in python variables share one budget: both are memory
        the model's own forward passes need too."""
        used, held = self._budget_used(), self._kernel_arrays()
        kept = sum(n for _, n in held)
        if used + kept + nbytes > self.mem_budget:
            raise SurgeryError(
                f"memory budget exceeded: {what} needs {nbytes / 1e9:.2f} GB and {(used + kept) / 1e9:.2f} of "
                f"{self.mem_budget / 1e9:.0f} GB is in use: {used / 1e9:.2f} GB by earlier whole-tensor edits"
                + (f", {kept / 1e9:.2f} GB by arrays in your python variables ({self._names(held)})" if kept else "")
                + ". Free some first: `del` python variables you no longer need, or undo whole-tensor edits. "
                "(Edits to single neurons are not affected.)"
            )

    def _kernel_note(self) -> str:
        """Run after each python call: keep held arrays within the budget, and say what is held."""
        used, held = self._budget_used(), self._kernel_arrays()
        kept = sum(n for _, n in held)
        dropped = []
        while held and used + kept > self.mem_budget:  # the budget is a bound, so the largest go
            name, n = held.pop(0)
            del self._py_ns[name]
            kept -= n
            dropped.append((name, n))
        if dropped:
            gc.collect()
            mx.clear_cache()
        note = ""
        if dropped:
            note += (
                f"\n[memory: your python variables and whole-tensor edits together were over the "
                f"{self.mem_budget / 1e9:.0f} GB budget they share, so these variables were deleted: "
                f"{self._names(dropped, 20)} GB. `del` arrays you no longer need before making more.]"
            )
        if kept > 0.05 * self.mem_budget:
            note += (
                f"\n[memory: your python variables hold {kept / 1e9:.2f} GB of arrays ({self._names(held)}); with "
                f"{used / 1e9:.2f} GB of whole-tensor edits that is {(used + kept) / 1e9:.2f} of the "
                f"{self.mem_budget / 1e9:.0f} GB budget. `del name` frees a variable.]"
            )
        return note

    # ----------------------------------------------------- control tokens --

    def show_safe(self, text: str) -> str:
        """Text for the model to read: control tokens as ‹...›, which it can quote without ending its turn."""
        if not self._ctl_re or not isinstance(text, str):
            return text
        return self._ctl_re.sub(lambda m: f"‹{m.group(0)[1:-1]}›", text)

    def read_safe(self, value):
        """Arguments the model wrote: the ‹...› form of a control token stands for the real one."""
        if not self._safe_re:
            return value
        if isinstance(value, str):
            return self._safe_re.sub(lambda m: f"<{m.group(0)[1:-1]}>", value)
        if isinstance(value, list):
            return [self.read_safe(v) for v in value]
        if isinstance(value, dict):
            return {k: self.read_safe(v) for k, v in value.items()}
        return value

    # ------------------------------------------------------------ reading --

    def check_neuron(self, neuron) -> tuple[int, int, int]:
        L, E, N = parse_neuron(neuron)
        if not (0 <= L < self.n_layers and 0 <= E < self.n_experts and 0 <= N < self.n_neurons):
            raise SurgeryError(
                f"{fmt_neuron(L, E, N)} is out of range: layers 0-{self.n_layers - 1}, "
                f"experts 0-{self.n_experts - 1}, neurons 0-{self.n_neurons - 1}"
            )
        return L, E, N

    def _proj_path(self, L: int, which: str) -> str:
        return f"{PREFIX}layers.{L}.mlp.switch_mlp.{PROJ[which]}"

    @staticmethod
    def _index(E: int, N: int, which: str) -> tuple:
        return (E, slice(None), N) if which == "out" else (E, N, slice(None))

    def _base_vec(self, mod: QuantizedSwitchLinear, E: int, N: int, which: str) -> mx.array:
        w = self._dequant(mod, E)
        return w[:, N] if which == "out" else w[N, :]

    def neuron_vec(self, L: int, E: int, N: int, which: str) -> mx.array:
        """One of a neuron's three vectors ('gate', 'up', 'out') as float32, edits included."""
        mod = getattr(self.layers[L].mlp.switch_mlp, PROJ[which])
        if isinstance(mod, PatchedSwitchLinear):
            v = self._base_vec(mod.base, E, N, which)
            delta = (mod.patch.cols if which == "out" else mod.patch.rows).get((E, N))
            return v if delta is None else v + delta
        if self._is_quantized(mod):
            return self._base_vec(mod, E, N, which)
        return mod["weight"][self._index(E, N, which)].astype(mx.float32)

    def _weight_ref(self, path: str):
        full = self._full(path)
        mod_path, _, key = full.rpartition(".")
        parent, pkey, mod = self._resolve(mod_path)
        if not isinstance(mod, nn.Module) or (key not in mod and not (key == "weight" and isinstance(mod, PatchedSwitchLinear))):
            raise SurgeryError(f"{path!r} is not a weight tensor; see S.list_weights()")
        return full, mod_path, key, parent, pkey, mod

    def get_weight(self, path: str) -> mx.array:
        """Float values of any weight tensor, edits included, e.g. 'layers.3.mlp.gate.weight'."""
        _, _, key, _, _, mod = self._weight_ref(path)
        if key == "weight" and isinstance(mod, PatchedSwitchLinear):
            w = np.array(self._dequant(mod.base))
            for (E, N), d in mod.patch.cols.items():
                w[E, :, N] += np.array(d)
            for (E, N), d in mod.patch.rows.items():
                w[E, N, :] += np.array(d)
            return mx.array(w)
        if key == "weight" and self._is_quantized(mod):
            return self._dequant(mod)
        return mod[key]

    def list_weights(self, pattern: str = "") -> list[tuple[str, tuple, str]]:
        out, seen = [], set()
        for name, _ in tree_flatten(self.model.parameters()):
            mod_path, _, key = name.rpartition(".")
            if key in ("scales", "biases"):
                continue
            state = None
            if mod_path.endswith(".base") and mod_path[: -len(".base")] in self._patched:
                mod_path = mod_path[: -len(".base")]
                name = f"{mod_path}.{key}"
                state = f"8-bit + {len(self._patched[mod_path].patch)} edited neuron vector(s)"
            if name in seen or (pattern and pattern not in name):
                continue
            seen.add(name)
            mod = self._resolve(mod_path)[2]
            if state is None:
                if key == "weight" and self._is_quantized(mod):
                    state = f"{mod.bits}-bit"
                else:
                    state = "float32 (replaced)" if mod_path in self._float_bytes else "float"
            shape = self._float_shape(mod) if key == "weight" else tuple(mod[key].shape)
            out.append((name, shape, state))
        return out

    # ------------------------------------------------------------ writing --

    def _log(self, row: dict):
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        with (self.run_dir / "ledger.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _checked(value, shape: tuple, what: str) -> mx.array:
        try:
            value = mx.array(value).astype(mx.float32)
        except Exception as e:  # noqa: BLE001
            raise SurgeryError(f"cannot read the new values for {what} as an array: {e}") from None
        if tuple(value.shape) != tuple(shape):
            raise SurgeryError(f"shape mismatch writing {what}: expected {tuple(shape)}, got {tuple(value.shape)}")
        if not bool(mx.all(mx.isfinite(value)).item()):
            raise SurgeryError("refusing to write non-finite values (nan/inf)")
        return value

    def _write_vec(self, edit: Edit, L: int, E: int, N: int, which: str, new):
        """Set one neuron vector. On a quantized tensor this is stored as a float delta."""
        path = self._proj_path(L, which)
        parent, pkey, mod = self._resolve(path)
        new = self._checked(new, (self.d_model,), f"{fmt_neuron(L, E, N)} {which}")
        if isinstance(mod, QuantizedSwitchLinear):
            mod = PatchedSwitchLinear(mod)
            self._assign(parent, pkey, mod)
            self._patched[path] = mod
        if isinstance(mod, PatchedSwitchLinear):
            table = mod.patch.cols if which == "out" else mod.patch.rows
            old = table.get((E, N))
            delta = new - self._base_vec(mod.base, E, N, which)
            mx.eval(delta)
            table[(E, N)] = delta
            mod.patch.pack(mod.base.output_dims)

            def undo():
                if old is None:
                    table.pop((E, N), None)
                else:
                    table[(E, N)] = old
                mod.patch.pack(mod.base.output_dims)
        else:  # the tensor was replaced by a float one earlier: write into it
            idx = self._index(E, N, which)
            cur = mod["weight"]
            # A slice shares its parent's buffer. Kept as it is for undo, it would keep the whole
            # previous tensor alive, one full copy per edit; so the values are copied out.
            old = mx.array(np.array(cur[idx]))
            cur[idx] = new.astype(cur.dtype)
            mod["weight"] = cur
            mx.eval(mod["weight"])

            def undo():
                w = mod["weight"]
                w[idx] = old
                mod["weight"] = w
                mx.eval(mod["weight"])

        edit.undo.append(undo)
        self.dirty = True

    def _write_full(self, edit: Edit, path: str, value):
        """Replace a whole tensor. A quantized module becomes a float32 one."""
        full, mod_path, key, parent, pkey, mod = self._weight_ref(path)
        if key == "weight" and (self._is_quantized(mod) or isinstance(mod, PatchedSwitchLinear)):
            shape = self._float_shape(mod)
            value = self._checked(value, shape, full)
            nbytes = math.prod(shape) * 4
            self._charge(nbytes, f"replacing {full}")
            base = mod.base if isinstance(mod, PatchedSwitchLinear) else mod
            if isinstance(base, QuantizedSwitchLinear):
                new = ThawedSwitchLinear(shape[2], shape[1], shape[0], bias="bias" in base)
            elif isinstance(base, nn.QuantizedLinear):
                new = ThawedLinear(shape[1], shape[0], bias="bias" in base)
            else:
                new = ThawedEmbedding(shape[0], shape[1])
                new.out_dtype = self.act_dtype
            new.weight = value
            if "bias" in base:
                new.bias = base["bias"]
            mx.eval(new.parameters())
            self._assign(parent, pkey, new)
            self._float_bytes[mod_path] = nbytes
            was_patched = self._patched.pop(mod_path, None)

            def undo():
                self._assign(parent, pkey, mod)
                self._float_bytes.pop(mod_path, None)
                if was_patched is not None:
                    self._patched[mod_path] = was_patched
        else:
            cur = mod[key]
            value = self._checked(value, tuple(cur.shape), full).astype(cur.dtype)
            self._charge(cur.nbytes, f"keeping the old {full} for undo")
            mx.eval(cur)
            mod[key] = value
            mx.eval(mod[key])
            edit.held += cur.nbytes

            def undo():
                mod[key] = cur
                mx.eval(mod[key])

        edit.undo.append(undo)
        self.dirty = True

    def _new_edit(self, op: str, target: str, detail: dict) -> Edit:
        self._seq += 1
        return Edit(self._seq, op, target, detail)

    def _commit(self, edit: Edit) -> Edit:
        self.edits.append(edit)
        self._log({"seq": edit.seq, "op": edit.op, "target": edit.target, **edit.detail})
        return edit

    def _rollback(self, edit: Edit):
        for fn in reversed(edit.undo):
            fn()
        edit.undo.clear()
        self.dirty = True

    def _tidy(self):
        """Unwrap modules that no longer carry any delta; hand freed memory back."""
        for path in [p for p, m in self._patched.items() if len(m.patch) == 0]:
            parent, pkey, mod = self._resolve(path)
            if mod is self._patched[path]:
                self._assign(parent, pkey, mod.base)
            del self._patched[path]
        gc.collect()
        mx.clear_cache()

    def set_neuron(self, neuron, gate=None, up=None, out=None, note: str = "", op: str = "set_neuron", extra=None) -> Edit:
        L, E, N = self.check_neuron(neuron)
        edit = self._new_edit(op, fmt_neuron(L, E, N), {"note": note, **(extra or {})})
        saved = {}
        try:
            for which, vec in (("gate", gate), ("up", up), ("out", out)):
                if vec is None:
                    continue
                before = float(mx.linalg.norm(self.neuron_vec(L, E, N, which)).item())
                self._write_vec(edit, L, E, N, which, vec)
                after = self.neuron_vec(L, E, N, which)
                edit.detail[f"{which}_norm"] = [round(before, 5), round(float(mx.linalg.norm(after).item()), 5)]
                saved[which] = np.array(after)
        except Exception:
            self._rollback(edit)
            self._tidy()
            self._seq -= 1
            raise
        if not edit.undo:
            self._seq -= 1
            raise SurgeryError("nothing to write: pass at least one of gate, up, out")
        if op == "set_neuron":  # keep the vectors so the edit can be replayed after a restart
            d = self.run_dir / "edits"
            d.mkdir(exist_ok=True)
            edit.detail["vectors"] = f"edits/{self.session}-{edit.seq}.npz"
            np.savez(self.run_dir / edit.detail["vectors"], **saved)
        return self._commit(edit)

    @staticmethod
    def _factor(factor) -> float:
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            raise SurgeryError(f"factor must be a number, got {factor!r}") from None
        if not math.isfinite(factor):
            raise SurgeryError("factor must be a finite number")
        return factor

    def scale_neuron(self, neuron, factor: float, note: str = "", batch: int | None = None) -> Edit:
        """Multiply the neuron's output vector (what it writes) by ``factor``."""
        L, E, N = self.check_neuron(neuron)
        factor = self._factor(factor)
        out = self.neuron_vec(L, E, N, "out") * factor
        extra = {"factor": factor} if batch is None else {"factor": factor, "batch": batch}
        return self.set_neuron((L, E, N), out=out, note=note, op="scale_neuron", extra=extra)

    def scale_neurons(self, items, note: str = "") -> list[Edit]:
        """Scale many neurons' output vectors in one call: all of them, or none if any is invalid.

        ``items`` holds (neuron, factor) pairs or {"neuron": ..., "factor": ...} dicts. Each neuron
        becomes its own ledger entry (so undo stays fine-grained), tagged with a shared batch number.
        """
        todo = []
        for it in items or ():
            try:
                neuron, factor = (it["neuron"], it["factor"]) if isinstance(it, dict) else it
            except (KeyError, TypeError, ValueError):
                raise SurgeryError(f"each item needs a neuron and a factor, got {it!r}") from None
            try:
                todo.append((self.check_neuron(neuron), self._factor(factor)))
            except ValueError as e:
                raise SurgeryError(str(e)) from None
        if not todo:
            raise SurgeryError("pass at least one {neuron, factor} item")
        if len(todo) > MAX_BATCH:
            raise SurgeryError(f"at most {MAX_BATCH} neurons per call, got {len(todo)}")
        batch, done = self._seq + 1, []
        try:
            for neuron, factor in todo:
                done.append(self.scale_neuron(neuron, factor, note, batch=batch))
        except Exception:
            self.undo(len(done))
            raise
        return done

    def set_weight(self, path: str, value, note: str = "") -> Edit:
        full = self._weight_ref(path)[0]
        edit = self._new_edit("set_weight", full, {"note": note})
        try:
            before = self.get_weight(full).astype(mx.float32)
            value = self._checked(value, tuple(before.shape), full)
            delta = value - before
            edit.detail["changed_values"] = int(mx.sum(mx.abs(delta) > 0).item())
            edit.detail["delta_norm"] = round(float(mx.linalg.norm(delta.reshape(-1)).item()), 5)
            rows = None
            if value.ndim:  # which slices along the first axis changed (experts, for an expert matrix)
                rows = np.flatnonzero(np.array(mx.any((delta != 0).reshape(value.shape[0], -1), axis=1)))
            del before, delta
            self._write_full(edit, full, value)
            if rows is not None:
                self._save_slices(edit, value, rows)
        except Exception:
            self._rollback(edit)
            self._tidy()
            self._seq -= 1
            raise
        return self._commit(edit)

    def _save_slices(self, edit: Edit, value: mx.array, rows: np.ndarray):
        """Keep the changed slices of a replaced tensor on disk, so the edit can be replayed after a restart."""
        name = f"edits/{self.session}-{edit.seq}.npz"
        try:
            (self.run_dir / "edits").mkdir(exist_ok=True)
            if len(rows) == value.shape[0]:
                values = np.array(value)
            elif len(rows):
                values = np.array(value[mx.array(rows)])
            else:
                values = np.zeros((0, *value.shape[1:]), np.float32)
            np.savez(self.run_dir / name, rows=rows, values=values)
            edit.detail["values"] = name
        except OSError as e:  # the edit stands; it just cannot be replayed
            edit.detail["not_saved"] = str(e)

    def undo(self, n: int = 1) -> list[Edit]:
        undone = []
        for _ in range(max(0, int(n))):
            if not self.edits:
                break
            e = self.edits.pop()
            self._rollback(e)
            undone.append(e)
            self._log({"op": "undo", "undid_seq": e.seq, "target": e.target})
        self._tidy()
        return undone

    def reset(self) -> int:
        n = len(self.edits)
        while self.edits:
            self._rollback(self.edits.pop())
        self._tidy()
        self._log({"op": "reset", "undid": n})
        self.dirty = True
        return n

    @staticmethod
    def ledger_sessions(text: str) -> list[list[dict]]:
        """A ledger's rows, split at each server start."""
        sessions: list[list[dict]] = []
        for line in text.splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("op") == "server_start":
                sessions.append([])
            elif sessions:
                sessions[-1].append(r)
        return sessions

    @staticmethod
    def active_rows(rows: list[dict]) -> list[dict]:
        """The edits still in effect at the end of one session's rows."""
        active: list[dict] = []
        for r in rows:
            if r.get("op") == "undo":
                active = [a for a in active if a.get("seq") != r.get("undid_seq")]
            elif r.get("op") == "reset":
                active = []
            elif "seq" in r:
                active.append(r)
        return active

    def bake(self, factors: dict) -> int:
        """Scale neuron output vectors inside the 8-bit weights themselves.

        For starting a run from weights that already carry a change. Unlike an edit it leaves
        nothing that tells the result from weights loaded that way: no delta beside the tensor,
        no ledger row, nothing to undo. ``factors`` maps a neuron id to the factor for its
        output vector. A scaled value is rounded to the nearest of its group's 256 levels, the
        rounding every weight in the tensor already has; no other neuron's values change.
        """
        if self.edits:
            raise SurgeryError("bake before any edit is made")
        by_layer: dict[int, list[tuple[int, int, float]]] = {}
        for neuron, factor in factors.items():
            L, E, N = self.check_neuron(neuron)
            by_layer.setdefault(L, []).append((E, N, self._factor(factor)))
        for L, items in by_layer.items():
            mod = self.layers[L].mlp.switch_mlp.down_proj
            if not (isinstance(mod, QuantizedSwitchLinear) and mod.bits == 8 and mod.mode == "affine"
                    and mod.get("biases") is not None):
                raise SurgeryError(f"layer {L}: can only bake into an untouched 8-bit affine-quantized tensor")
            packed = np.array(mod["weight"])  # uint32: four 8-bit levels in each, lowest byte first
            levels = packed.view(np.uint8)  # (experts, d_model, neurons)
            scales, biases = (np.array(mod[k].astype(mx.float32)) for k in ("scales", "biases"))
            for E, N, f in items:
                s, b = scales[E, :, N // mod.group_size], biases[E, :, N // mod.group_size]
                flat = s == 0  # a group whose values are all the same has one level
                want = np.rint((f * (s * levels[E, :, N] + b) - b) / np.where(flat, 1, s))
                levels[E, :, N] = np.where(flat, levels[E, :, N], np.clip(want, 0, 255)).astype(np.uint8)
            mod["weight"] = mx.array(packed)
            mx.eval(mod["weight"])
        self.dirty = True
        return len(factors)

    def replay(self) -> str:
        """Re-apply the edits that were in effect when the previous server process on this run stopped."""
        path = self.run_dir / "ledger.jsonl"
        sessions = self.ledger_sessions(path.read_text() if path.exists() else "")
        if len(sessions) < 2:  # the last session is this one
            return "nothing to replay: this run has no earlier session"
        active = self.active_rows(sessions[-2])
        done, skipped = 0, []
        for r in active:
            note = (r.get("note") or "") + f" [replayed #{r['seq']}]"
            try:
                if r["op"] == "scale_neuron":
                    self.scale_neuron(r["target"], r["factor"], note.strip(), batch=r.get("batch"))
                elif r["op"] == "set_neuron" and r.get("vectors"):
                    with np.load(self.run_dir / r["vectors"]) as z:
                        self.set_neuron(r["target"], note=note.strip(), **{k: z[k] for k in z.files})
                elif r["op"] == "set_weight" and r.get("values"):
                    # the tensor as it stands now (earlier edits are back already), with the saved slices put in
                    w = np.array(self.get_weight(r["target"]).astype(mx.float32))
                    with np.load(self.run_dir / r["values"]) as z:
                        w[z["rows"]] = z["values"]
                    self.set_weight(r["target"], w, note=note.strip())
                else:
                    raise SurgeryError("this edit's values were not saved to disk")
                done += 1
            except (SurgeryError, OSError, KeyError) as e:
                skipped.append(f"#{r.get('seq')} {r.get('op')} {r.get('target')}: {e}")
        msg = f"replayed {done} of {len(active)} edit(s) from the previous session"
        return msg + ("; could not replay " + "; ".join(skipped) if skipped else "")

    # ------------------------------------------------------------ forward --

    def tokens(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def pieces(self, ids) -> list[str]:
        return [self.tok.decode([int(i)]) for i in ids]

    def chat_tokens(self, user: str, system: str | None = None, reply: str = "",
                    thinking: bool = False) -> tuple[list[int], int, int]:
        """A chat holding one user message, as the token ids the model sees when it is about to reply.

        Returns (tokens, turn, start). tokens[turn:start] is the template's hand-over to the
        assistant, tokens[start:] is ``reply`` (empty unless given), and the token at start - 1 is
        the position that predicts the first token of the reply.
        """
        msgs = ([{"role": "system", "content": str(system)}] if system else []) + [{"role": "user", "content": str(user)}]
        head = list(self.tok.apply_chat_template(msgs, add_generation_prompt=False))
        toks = list(self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=bool(thinking)))
        turn = len(head) if toks[: len(head)] == head and len(head) < len(toks) else len(toks) - 1
        return toks + (self.tokens(reply) if reply else []), turn, len(toks)

    def _ids(self, tokens_or_text) -> list[int]:
        return self.tokens(tokens_or_text) if isinstance(tokens_or_text, str) else [int(t) for t in tokens_or_text]

    def logprob(self, tokens, continuation) -> list[float]:
        """Log-probability of each token of ``continuation`` when it follows ``tokens`` (teacher forced)."""
        tokens, cont = self._ids(tokens), self._ids(continuation)
        if not tokens or not cont:
            raise SurgeryError("logprob needs a non-empty context and a non-empty continuation")
        seq = tokens + cont
        if len(seq) > MAX_PROBE_TOKENS:
            raise SurgeryError(f"{len(seq)} tokens; the limit per forward pass is {MAX_PROBE_TOKENS}")
        logits = self.model(mx.array([seq[:-1]]))[0, len(tokens) - 1:].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        per = mx.take_along_axis(lp, mx.array(cont)[:, None], axis=-1)[:, 0]
        mx.eval(per)
        return [float(v) for v in per.tolist()]

    def greedy(self, tokens, max_tokens: int = 48) -> list[int]:
        """The tokens the current weights continue ``tokens`` with at temperature 0 (stops at end of turn)."""
        tokens = self._ids(tokens)
        if not tokens:
            raise SurgeryError("empty prompt")
        stop = {int(t) for t in (getattr(self.tok, "eos_token_ids", None) or ())}
        cache = make_prompt_cache(self.model)
        out: list[int] = []
        x = mx.array([tokens])
        for _ in range(max(1, min(int(max_tokens), 2000))):
            t = int(mx.argmax(self.model(x, cache=cache)[0, -1]).item())
            if t in stop:
                break
            out.append(t)
            x = mx.array([[t]])
        del cache
        return out

    def forward(self, tokens, neurons=(), agg_layers=(), resid_layers=(), rec: Recorder | None = None,
                positions=None):
        """One full forward pass over ``tokens`` with the current weights.

        ``positions`` limits the per-layer aggregates (rec.agg) to those token positions.
        """
        global _REC
        tokens = list(tokens)
        if not tokens:
            raise SurgeryError("empty text")
        if len(tokens) > MAX_PROBE_TOKENS:
            raise SurgeryError(f"text is {len(tokens)} tokens; the limit per forward pass is {MAX_PROBE_TOKENS}")
        if rec is None:
            rec = Recorder()
        counted = len(tokens)
        rec.mask = None
        if positions is not None:
            idx = sorted({int(p) + len(tokens) if int(p) < 0 else int(p) for p in positions})
            if not idx or idx[0] < 0 or idx[-1] >= len(tokens):
                raise SurgeryError(f"positions must be token indices within 0..{len(tokens) - 1} (negatives count from the end)")
            mask = np.zeros(len(tokens), dtype=np.float32)
            mask[idx] = 1.0
            rec.mask, counted = mx.array(mask), len(idx)
        for L, E, N in (self.check_neuron(n) for n in neurons):
            rec.targets.setdefault(L, []).append((E, N))
        rec.agg_layers |= set(agg_layers)
        rec.resid_layers |= set(resid_layers)
        rec.moe_layer, rec.block_layer = self._moe_layer, self._block_layer
        _REC = rec
        try:
            logits = self.model(mx.array([tokens]))
            last = logits[0, -1].astype(mx.float32)
            mx.eval(last, *rec.pending())
        finally:
            _REC = None
            rec.mask = None
        rec.ntok += counted
        return last, rec

    def logit_lens(self, vec, k: int = 8) -> tuple[list[str], list[str]]:
        """Tokens a residual-stream direction most promotes / most suppresses."""
        v = mx.array(vec).astype(self.act_dtype).reshape(1, 1, -1)
        logits = self.text.lm_head(self.text.model.norm(v))[0, 0].astype(mx.float32)
        order = mx.argsort(logits)
        lo, hi = order[:k].tolist(), order[-k:].tolist()[::-1]
        return self.pieces(hi), self.pieces(lo)

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 300,
                 thinking: bool = False, temperature: float = 0.7) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        toks = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=bool(thinking))
        sampler = make_sampler(temp=float(temperature), top_p=0.95, top_k=20)
        text = ""
        for r in stream_generate(
            self.model, self.tok, toks, max_tokens=int(max_tokens), sampler=sampler,
            stream=mx.default_stream(mx.default_device()),
        ):
            text += r.text
        return text

    # -------------------------------------------------------------- tools --
    # Each returns text meant to be read by the model.

    def t_atlas_search(self, query: str, layer: int | None = None, limit: int = 20) -> str:
        if not len(self.atlas):
            return "The atlas index is missing (run: uv run python -m surgery.atlas)."
        hits = self.atlas.search(query, layer=layer, limit=max(1, min(int(limit), 60)))
        if not hits:
            return (
                f"No atlas entries match {query!r}. Matching is by substring on each word, so try stems "
                f"(e.g. 'happ' or 'joy'). The atlas covers {len(self.atlas.by_neuron)} sampled neurons in "
                f"layers {self.atlas.layers}; `contrast` can find neurons the atlas does not cover."
            )
        return f"{len(hits)} matches for {query!r} (id [pole] (category, simulatability) description):\n" + "\n".join(
            Atlas.line(r) for _, r in hits
        )

    def t_inspect_neuron(self, neuron: str) -> str:
        L, E, N = self.check_neuron(neuron)
        lines = [f"{fmt_neuron(L, E, N)}  (layer {L} of {self.n_layers - 1}, expert {E}, neuron {N})"]
        vecs = {w: self.neuron_vec(L, E, N, w) for w in ("gate", "up", "out")}
        norms = {w: float(mx.linalg.norm(v).item()) for w, v in vecs.items()}
        lines.append(f"vector norms: gate {norms['gate']:.3f}, up {norms['up']:.3f}, out {norms['out']:.3f}")
        mine = [e for e in self.edits if e.target == fmt_neuron(L, E, N)]
        if mine:
            lines.append("edited this session: " + ", ".join(f"#{e.seq} {e.op} {e.detail.get('factor', '')}".strip() for e in mine))
        if norms["out"] > 0:
            hi, lo = self.logit_lens(vecs["out"])
            lines.append("direct effect of its output on next-token logits (rough; most reliable in late layers):")
            lines.append("    when it fires +: promotes " + " ".join(repr(t) for t in hi))
            lines.append("    when it fires -: promotes " + " ".join(repr(t) for t in lo))
        lines.append("atlas:")
        lines.append(self.atlas.describe(L, E, N))
        return "\n".join(lines)

    def t_probe(self, text: str, neurons: list[str], chat: bool = False, reply: str = "") -> str:
        if not neurons:
            raise SurgeryError("pass at least one neuron id")
        if len(neurons) > 24:
            raise SurgeryError("at most 24 neurons per probe")
        ids = [self.check_neuron(n) for n in neurons]
        zone = None
        if chat or reply:
            toks, turn, start = self.chat_tokens(text, reply=reply)
            zone = range(turn, min(len(toks), turn + 40))
        else:
            toks = self.tokens(text)
        _, rec = self.forward(toks, neurons=ids)
        pieces = self.pieces(toks)
        out = [f"probe over {len(toks)} tokens (activation = silu(gate.x) * (up.x); 0 where the expert was not routed)"]
        if zone is not None:
            out[0] += (
                f"\nchat format: the text is the user message; your reply begins at token {start}"
                + (f" and is forced to start with {reply!r}" if reply else "")
                + ". 'where the reply begins' lists every token from the hand-over to the assistant onwards; "
                "the last one before the reply text decides the reply's first token."
            )
        for L, E, N in ids:
            a = np.array(rec.acts[(L, E, N)])
            r = np.array(rec.route[(L, E)])
            routed = int((r > 0).sum())
            head = f"{fmt_neuron(L, E, N)}: expert routed on {routed}/{len(toks)} tokens"
            if routed == 0:
                out.append(head + "; neuron silent on this text")
                continue
            head += f", mean routing weight {r[r > 0].mean():.2f}; max {a.max():+.3f}, min {a.min():+.3f}"
            out.append(head)
            for i in np.argsort(-np.abs(a))[:4]:
                if a[i] == 0:
                    break
                lo, hi = max(0, i - 6), min(len(pieces), i + 3)
                ctx = "".join(pieces[lo:i]) + "«" + pieces[i] + "»" + "".join(pieces[i + 1:hi])
                out.append(f"    {a[i]:+.3f}  {ctx!r}")
            if zone is not None:
                cells = [f"{pieces[i]!r} {a[i]:+.3f}" if r[i] > 0 else f"{pieces[i]!r} ·" for i in zone]
                out.append("    where the reply begins (· = expert not routed): " + " | ".join(cells))
            for sign in ("+", "-"):
                lab = self.atlas.label(L, E, N, sign)
                if lab:
                    out.append(f"    atlas[{sign}]: {lab}")
        return "\n".join(out)

    def t_contrast(self, texts_a: list[str], texts_b: list[str], layers: list[int] | None = None, top_k: int = 12,
                   chat: bool = False, reply_a: str = "", reply_b: str = "") -> str:
        if not texts_a or not texts_b:
            raise SurgeryError("both texts_a and texts_b need at least one text")
        layers = sorted(set(layers)) if layers else list(range(self.n_layers))
        bad = [l for l in layers if not 0 <= l < self.n_layers]
        if bad:
            raise SurgeryError(f"layers out of range: {bad}")
        top_k = max(1, min(int(top_k), 40))
        chat = bool(chat or reply_a or reply_b)

        def run(texts, reply):
            rec = Recorder(agg_layers=layers)
            for t in texts:
                if chat:  # read only where the reply begins, not the whole text
                    toks, turn, _ = self.chat_tokens(t, reply=reply)
                    self.forward(toks, rec=rec, positions=range(turn, len(toks)))
                else:
                    self.forward(self.tokens(t), rec=rec)
            return rec

        ra, rb = run(texts_a, reply_a), run(texts_b, reply_b)
        cands = []  # (diff, L, E, N, pole, mean_a, mean_b)
        for L in layers:
            for pole, key in (("+", "pos"), ("-", "neg")):
                ma = np.array(ra.agg[L][key]) / ra.ntok
                mb = np.array(rb.agg[L][key]) / rb.ntok
                d = ma - mb
                flat = d.reshape(-1)
                k = min(top_k, flat.size)
                for part in (np.argpartition(-flat, k - 1)[:k], np.argpartition(flat, k - 1)[:k]):
                    for j in part:
                        E, N = divmod(int(j), d.shape[1])
                        cands.append((float(flat[j]), L, E, N, pole, float(ma[E, N]), float(mb[E, N])))

        def fmt(c):
            diff, L, E, N, pole, a, b = c
            lab = self.atlas.label(L, E, N, pole)
            return f"  {fmt_neuron(L, E, N)} [{pole}]  A={a:.4f} B={b:.4f} diff={diff:+.4f}" + (f"  | atlas: {lab}" if lab else "")

        up = sorted([c for c in cands if c[0] > 0], key=lambda c: -c[0])[:top_k]
        down = sorted([c for c in cands if c[0] < 0], key=lambda c: c[0])[:top_k]
        return "\n".join(
            [
                f"contrast over layers {layers[0]}-{layers[-1]} ({len(layers)} layers): "
                f"A = {len(texts_a)} texts / {ra.ntok} tokens, B = {len(texts_b)} texts / {rb.ntok} tokens.",
                *([
                    "chat format: each text is a user message, and only the tokens where your reply begins are read "
                    "(the hand-over to the assistant"
                    + (f", then the forced reply openings A {reply_a!r} / B {reply_b!r}" if reply_a or reply_b else "")
                    + ")."
                ] if chat else []),
                "value = mean over tokens of routing_weight * |activation| for that pole.",
                "more active on A:",
                *(fmt(c) for c in up),
                "more active on B:",
                *(fmt(c) for c in down),
            ]
        )

    def t_scale_neuron(self, neuron: str, factor: float, note: str = "") -> str:
        e = self.scale_neuron(neuron, factor, note)
        b, a = e.detail["out_norm"]
        return (
            f"edit #{e.seq}: {e.target} output vector scaled by {float(factor):g} (norm {b:.4f} -> {a:.4f}). "
            f"The change is live: your next tokens are computed with it. {len(self.edits)} edit(s) active."
        )

    def t_scale_neurons(self, edits: list, note: str = "") -> str:
        done = self.scale_neurons(edits, note)
        shown = ", ".join(f"{e.target} ×{e.detail['factor']:g}" for e in done[:12])
        more = f", and {len(done) - 12} more" if len(done) > 12 else ""
        return (
            f"edits #{done[0].seq}-#{done[-1].seq}: scaled the output vectors of {len(done)} neuron(s) in one step "
            f"({shown}{more}). The changes are live: your next tokens are computed with them, after one re-read of "
            f"the conversation. `ledger` with action undo and n={len(done)} takes this batch back. "
            f"{len(self.edits)} edit(s) active."
        )

    def t_sample_self(self, prompt: str, system: str | None = None, max_tokens: int = 300,
                      thinking: bool = False, temperature: float = 0.7) -> str:
        max_tokens = max(1, min(int(max_tokens), 2000))
        return self.generate(prompt, system=system, max_tokens=max_tokens, thinking=thinking, temperature=temperature)

    def t_ledger(self, action: str = "list", n: int = 1) -> str:
        action = (action or "list").lower()
        if action == "undo":
            undone = self.undo(n)
            if not undone:
                return "Nothing to undo."
            return "Undid " + ", ".join(f"#{e.seq} {e.op} {e.target}" for e in undone) + f". {len(self.edits)} edit(s) remain active."
        if action == "reset":
            return f"Reverted all {self.reset()} edit(s). The weights are the ones this run started with again."
        if action != "list":
            raise SurgeryError("action must be one of: list, undo, reset")
        lines = [
            f"{len(self.edits)} active edit(s) touching {sum(len(m.patch) for m in self._patched.values())} neuron "
            f"vector(s); {len(self._float_bytes)} whole tensor(s) replaced "
            f"({self._budget_used() / 1e9:.2f} of {self.mem_budget / 1e9:.0f} GB budget for those)."
        ]
        for e in self.edits:
            d = {k: v for k, v in e.detail.items() if v not in ("", None)}
            lines.append(f"  #{e.seq} {e.op} {e.target} {json.dumps(d, ensure_ascii=False)}")
        return "\n".join(lines)

    def t_python(self, code: str) -> str:
        if self._py_ns is None:
            self._py_ns = {"S": PyAPI(self), "mx": mx, "np": np, "math": math}
        buf: list[str] = []

        def _print(*args, sep=" ", end="\n", **_):
            buf.append(sep.join(str(a) for a in args) + end)

        self._py_ns["print"] = _print
        # This runs on the generation thread, so a loop that never ends would hang
        # the model for good. A timer raises PythonTimeout inside the running code.
        tid, live = threading.get_ident(), threading.Lock()
        done = False

        def interrupt():
            with live:
                if done:
                    return
                _limit["over"] = True
                sys.monitoring.restart_events()  # from now on every line of the model's code raises
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(PythonTimeout))

        _limit["over"] = False
        sys.monitoring.set_events(_LIMIT_TOOL, sys.monitoring.events.LINE)
        timer = threading.Timer(self.python_timeout, interrupt)
        timer.daemon = True
        timer.start()
        try:
            try:
                tree = ast.parse(code, mode="exec")
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<surgery>", "exec"), self._py_ns)
                if last is not None:
                    val = eval(compile(ast.Expression(last.value), "<surgery>", "eval"), self._py_ns)
                    if val is not None:
                        _print(repr(val))
            finally:
                # Holding the lock, the timer cannot raise again. An interrupt it has already
                # raised either lands here, inside this try, or is dropped by the next line.
                live.acquire()
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        except SurgeryError as e:
            _print(f"SurgeryError: {e}")
        except PythonTimeout:
            _print(f"TimeoutError: stopped after {self.python_timeout:g} seconds. Edits already made stand; see the ledger.")
        except BaseException:  # noqa: BLE001 - the model's code may raise anything
            tb = traceback.format_exc(limit=6)
            _print(tb[tb.find('File "<surgery>"') - 2:] if 'File "<surgery>"' in tb else tb)
        done = True
        _limit["over"] = False
        sys.monitoring.set_events(_LIMIT_TOOL, 0)
        live.release()
        timer.cancel()
        out = "".join(buf)
        if len(out) > 8000:
            out = out[:8000] + f"\n… [output truncated: {len(out)} characters in total]"
        return (out or "(no output; print something or end with an expression)") + self._kernel_note()

    def status(self) -> dict:
        return {
            "layers": self.n_layers,
            "experts": self.n_experts,
            "neurons_per_expert": self.n_neurons,
            "d_model": self.d_model,
            "experts_per_token": self.top_k,
            "dtype": str(self.act_dtype),
            "active_edits": len(self.edits),
            "edited_neuron_vectors": sum(len(m.patch) for m in self._patched.values()),
            "replaced_tensors": sorted(self._float_bytes),
            "edit_budget_gb": [round(self._budget_used() / 1e9, 3), self.mem_budget / 1e9],
            "python_arrays_gb": round(sum(n for _, n in self._kernel_arrays()) / 1e9, 3),
            "atlas_poles": len(self.atlas),
            "run_dir": str(self.run_dir),
        }


class PyAPI:
    """What the model's Python code sees as ``S``."""

    def __init__(self, s: Surgery):
        self._s = s
        self.n_layers, self.n_experts, self.n_neurons, self.d_model = s.n_layers, s.n_experts, s.n_neurons, s.d_model

    def help(self) -> str:
        return py_help(self._s)

    def __repr__(self) -> str:
        return "<S: your weights. print(S.help()) for the API>"

    # reading
    def list_weights(self, pattern: str = ""):
        return self._s.list_weights(pattern)

    def get(self, path: str) -> mx.array:
        return self._s.get_weight(path)

    def neuron(self, neuron) -> dict[str, np.ndarray]:
        L, E, N = self._s.check_neuron(neuron)
        return {w: np.array(self._s.neuron_vec(L, E, N, w)) for w in ("gate", "up", "out")}

    # writing (tracked in the ledger, undoable)
    def set(self, path: str, value, note: str = "") -> str:
        e = self._s.set_weight(path, value, note)
        return f"edit #{e.seq}: set {e.target} ({e.detail['changed_values']} values changed)"

    def set_neuron(self, neuron, gate=None, up=None, out=None, note: str = "") -> str:
        e = self._s.set_neuron(neuron, gate=gate, up=up, out=out, note=note)
        return f"edit #{e.seq}: set {e.target} " + json.dumps({k: v for k, v in e.detail.items() if k.endswith('_norm')})

    def scale_neuron(self, neuron, factor: float, note: str = "") -> str:
        return self._s.t_scale_neuron(neuron, factor, note)

    def scale_neurons(self, items, note: str = "") -> str:
        return self._s.t_scale_neurons(list(items), note)

    def undo(self, n: int = 1) -> str:
        return self._s.t_ledger("undo", n)

    def reset(self) -> str:
        return self._s.t_ledger("reset")

    def ledger(self) -> str:
        return self._s.t_ledger("list")

    # running yourself
    def tokens(self, text: str) -> list[int]:
        return self._s.tokens(text)

    def pieces(self, ids) -> list[str]:
        return self._s.pieces(ids)

    def chat(self, user: str, system: str | None = None, reply: str = "", thinking: bool = False) -> list[int]:
        return self._s.chat_tokens(user, system, reply, thinking)[0]

    def logprob(self, tokens, continuation) -> list[float]:
        return self._s.logprob(tokens, continuation)

    def greedy(self, tokens, max_tokens: int = 48) -> str:
        return self._s.tok.decode(self._s.greedy(tokens, max_tokens))

    def score(self, metric: str | None = None, **kw) -> str:
        from . import score

        return score.tool(self._s, {"metric": metric, **kw})

    def forward(self, text, neurons=(), resid_layers=(), agg_layers=(), positions=None) -> dict:
        toks = self._s.tokens(text) if isinstance(text, str) else [int(t) for t in text]
        last, rec = self._s.forward(toks, neurons=neurons, agg_layers=agg_layers, resid_layers=resid_layers,
                                    positions=positions)
        return {
            "tokens": toks,
            "pieces": self._s.pieces(toks),
            "logits_last": np.array(last),
            "acts": {fmt_neuron(*k): np.array(v) for k, v in rec.acts.items()},
            "routing": {f"L{k[0]}/E{k[1]}": np.array(v) for k, v in rec.route.items()},
            "resid": {L: np.array(v) for L, v in rec.resid.items()},
            "agg": {L: {k: np.array(v) for k, v in a.items()} for L, a in rec.agg.items()},
        }

    def logit_lens(self, vec, k: int = 8):
        return self._s.logit_lens(vec, k)

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 300,
                 thinking: bool = False, temperature: float = 0.7) -> str:
        return self._s.generate(prompt, system, max_tokens, thinking, temperature)

    # atlas
    def atlas_search(self, query: str, layer: int | None = None, limit: int = 20) -> str:
        return self._s.t_atlas_search(query, layer, limit)

    def atlas(self, neuron) -> str:
        return self._s.atlas.describe(*self._s.check_neuron(neuron))


PY_HELP = """S is a handle on your own weights. Variables persist between python calls. mx (mlx.core), np, math are imported.

Layout: {n_layers} layers; each layer's MLP is a mixture of {n_experts} experts, {top_k} routed per token,
each expert has {n_neurons} neurons; residual stream width {d_model}. For neuron L/E/N:
  gate = layers.L.mlp.switch_mlp.gate_proj.weight[E, N, :]   (reads the residual stream)
  up   = layers.L.mlp.switch_mlp.up_proj.weight[E, N, :]     (reads the residual stream)
  out  = layers.L.mlp.switch_mlp.down_proj.weight[E, :, N]   (writes to the residual stream)
  activation = silu(gate.x) * (up.x);  contribution = routing_weight * activation * out
There are no bias terms anywhere. Other tensors: layers.L.mlp.gate.weight (the expert router),
layers.L.mlp.shared_expert.*, attention weights, layers.L.*layernorm.weight, norm.weight, embed_tokens.weight, lm_head.weight.

Reading
  S.list_weights(pattern="")      -> [(path, shape, state)]
  S.get(path)                     -> mx.array of float values (index it before converting: np.array(S.get(p)[3]))
  S.neuron("L9/E190/N48")         -> {{"gate": np, "up": np, "out": np}}
Writing (takes effect immediately; logged; undoable)
  S.scale_neuron(id, factor, note="")
  S.scale_neurons([(id, factor), ...], note="")   -> many neurons in one step
  S.set_neuron(id, gate=None, up=None, out=None, note="")
  S.set(path, array, note="")     -> replace a whole tensor (same shape)
  S.undo(n=1)   S.reset()   S.ledger()
Memory
  A replaced tensor is held as float32 (about {tensor_gb:.2f} GB for one layer's expert matrix). Those, and every
  array your variables hold, share a budget of {budget_gb:.0f} GB: a write that would exceed it fails and names the
  variables, and if a call ends over the budget its largest variables are deleted. Keep one copy, not several:
      w = np.array(S.get(path)); w[e] *= 0.5; S.set(path, w); del w
  After a tool call that changed weights, the whole conversation is re-read by the new weights before your
  next token. That is once per tool call, however many edits the call made, so put related edits in one call.
Running yourself with the current weights
  S.forward(text_or_token_ids, neurons=[...], resid_layers=[...], agg_layers=[...], positions=None)
      -> dict: tokens, pieces, logits_last (vocab,), acts[id] (T,), routing["L/E"] (T,),
         resid[L] (T, {d_model}) = residual stream after layer L, for the layers in resid_layers,
         agg[L] = {{"pos","neg": ({n_experts},{n_neurons}) summed routing-weighted activation, "routed": ({n_experts},)}}
                  for the layers in agg_layers (empty if you pass none); summed over `positions` only, if given
      A string is read as raw text, with no chat formatting.
  S.chat(user, system=None, reply="", thinking=False) -> token ids of a chat holding that user message, ending
      exactly where your reply begins (followed by the tokens of `reply`, if given). Your replies are produced
      from this format, not from raw text. Pass the ids to S.forward: with reply="" the last position is the one
      that chooses the first token of the reply, so resid[L][-1] is the state that decision is made from.
  S.logprob(text_or_ids, continuation) -> [log-probability of each token of `continuation`] after that context
  S.greedy(text_or_ids, max_tokens=48) -> the text you continue with at temperature 0 (same weights, same text)
  S.generate(prompt, system=None, max_tokens=300, thinking=False, temperature=0.7) -> str (fresh chat, sampled)
  S.score(metric="list")          -> the same measurement as the `score` tool, as text
  S.logit_lens(vector, k=8)       -> (tokens promoted, tokens suppressed)
  S.tokens(text) / S.pieces(ids)  -> a word after a space is a different token: S.tokens(" happy") != S.tokens("happy")
Atlas
  S.atlas_search(query, layer=None, limit=20) -> text / S.atlas(id) -> text
"""


def py_help(s: Surgery) -> str:
    return PY_HELP.format(
        n_layers=s.n_layers, n_experts=s.n_experts, top_k=s.top_k, n_neurons=s.n_neurons, d_model=s.d_model,
        tensor_gb=s.n_experts * s.n_neurons * s.d_model * 4 / 1e9, budget_gb=s.mem_budget / 1e9,
    )
