"""Surgery core against a tiny random model with the real architecture."""
import json
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import tiny  # noqa: E402

from surgery.atlas import Atlas, parse_neuron  # noqa: E402
from surgery.core import Recorder, Surgery, SurgeryError  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog, twice."


@pytest.fixture(params=["bfloat16", "float32"])
def s(request):
    """The real model computes in bfloat16; float32 lets the same tests demand exactness."""
    model, tok = tiny.build(dtype=getattr(mx, request.param))
    s = Surgery(model, tok, Path(tempfile.mkdtemp()), Atlas(Path("/nonexistent")), mem_budget_gb=1.0)
    s.tol = 3e-2 if request.param == "bfloat16" else 1e-4  # bf16 keeps ~2.5 significant digits
    return s


def logits(s, text=TEXT):
    return np.array(s.forward(s.tokens(text))[0])


def test_parse_neuron():
    assert parse_neuron("L14/E78/N441") == (14, 78, 441)
    assert parse_neuron("L14_E78_N441__pos") == (14, 78, 441)
    assert parse_neuron("14/78/441") == (14, 78, 441)
    assert parse_neuron([1, 2, 3]) == (1, 2, 3)
    with pytest.raises(ValueError):
        parse_neuron("neuron seven")


def test_capture_path_matches_plain_forward(s):
    base = logits(s)
    toks = s.tokens(TEXT)
    captured, rec = s.forward(toks, neurons=["L1/E3/N5"], agg_layers=range(4), resid_layers=[0, 3])
    np.testing.assert_allclose(np.array(captured), base, atol=s.tol)
    assert rec.acts[(1, 3, 5)].shape == (len(toks),)
    assert rec.resid[3].shape == (len(toks), 64)
    assert set(rec.agg) == {0, 1, 2, 3}
    # every token routes to exactly top_k experts
    assert float(rec.agg[2]["routed"].sum().item()) == len(toks) * 2


def test_recorded_activation_is_the_neurons_contribution(s):
    """Zeroing the out vector of a firing neuron must change the logits; of a silent one must not."""
    short = "ab"  # 2 tokens x top-2 routing leaves most of the 8 experts unrouted
    _, rec = s.forward(s.tokens(short), agg_layers=[1])
    routed = np.array(rec.agg[1]["routed"])
    E_on, E_off = int(routed.argmax()), int(routed.argmin())
    assert routed[E_off] == 0 and routed[E_on] > 0
    base = logits(s, short)
    s.scale_neuron((1, E_off, 7), 0.0)
    np.testing.assert_allclose(logits(s, short), base, atol=s.tol)
    for n in range(64):
        s.scale_neuron((1, E_on, n), 0.0)
    assert np.abs(logits(s, short) - base).max() > 10 * s.tol


def test_a_no_op_edit_changes_nothing_and_scale_changes_only_its_neuron(s):
    base = logits(s)
    original = s.layers[2].mlp.switch_mlp.down_proj
    cur = {w: s.neuron_vec(2, 1, 9, w) for w in ("gate", "up", "out")}
    s.set_neuron("L2/E1/N9", **cur)  # rewriting a neuron with its own values
    np.testing.assert_array_equal(logits(s), base)
    s.undo()

    before = np.array(s.neuron_vec(2, 1, 9, "out"))
    others = np.array(s.neuron_vec(2, 1, 10, "out"))
    e = s.scale_neuron("L2/E1/N9", 3.0, note="test")
    np.testing.assert_allclose(np.array(s.neuron_vec(2, 1, 9, "out")), before * 3, rtol=1e-5)
    np.testing.assert_array_equal(np.array(s.neuron_vec(2, 1, 10, "out")), others)
    assert e.detail["factor"] == 3.0 and s.dirty

    s.undo()
    np.testing.assert_array_equal(np.array(s.neuron_vec(2, 1, 9, "out")), before)
    assert not s.edits and not s._patched
    assert s.layers[2].mlp.switch_mlp.down_proj is original, "undoing the only edit puts the 8-bit module back"
    np.testing.assert_array_equal(logits(s), base)


@pytest.mark.parametrize("text", ["ab", TEXT])  # short text: plain gather; long text: the sorted-gather path
def test_deltas_compute_the_same_function_as_edited_float_weights(s, text):
    """base + delta in the forward pass must equal a float tensor holding the edited values."""
    _, rec = s.forward(s.tokens(text), agg_layers=[1])
    E = int(np.array(rec.agg[1]["routed"]).argmax())  # an expert this text actually uses
    rng = np.random.default_rng(0)
    for n in (3, 4, 40):
        s.scale_neuron((1, E, n), -4.0)
        s.set_neuron((1, E, n), gate=rng.normal(size=64) * 0.3, up=rng.normal(size=64) * 0.3)
    s.set_neuron((1, (E + 1) % 8, 5), out=rng.normal(size=64))
    patched = logits(s, text)
    assert np.abs(patched - np.array(logits_of_fresh(text, s))).max() > 10 * s.tol, "the edits should matter"

    for proj in ("gate_proj", "up_proj", "down_proj"):
        path = f"layers.1.mlp.switch_mlp.{proj}.weight"
        s.set_weight(path, s.get_weight(path))  # same values, now held as one float tensor
    assert not s._patched and len(s._float_bytes) == 3
    np.testing.assert_allclose(logits(s, text), patched, atol=s.tol)

    s.scale_neuron((1, E, 3), 0.5)  # a neuron edit on a replaced tensor writes into the float tensor
    assert np.abs(logits(s, text) - patched).max() > 0
    s.undo()
    np.testing.assert_allclose(logits(s, text), patched, atol=s.tol)
    s.undo(3)
    np.testing.assert_array_equal(logits(s, text), patched)  # back on base + delta, exactly as before


def test_neuron_edits_on_a_replaced_tensor_keep_no_old_copies(s):
    """The slice saved for undo must not keep the whole previous tensor alive (one copy per edit)."""
    path = "layers.1.mlp.switch_mlp.down_proj.weight"
    original = np.array(s.get_weight(path))
    s.set_weight(path, s.get_weight(path))
    size = next(iter(s._float_bytes.values()))
    s.scale_neuron((1, 0, 0), 0.5)  # warm up, so one-off allocations are not counted
    before = mx.get_active_memory()
    edits = [((1, e, n), 0.5) for e in range(8) for n in range(1, 26)]
    s.scale_neurons(edits)
    grown = mx.get_active_memory() - before
    assert grown < 0.25 * len(edits) * size, f"{grown} bytes kept after {len(edits)} edits of a {size}-byte tensor"
    s.undo(len(edits) + 1)  # the copied-out values still undo exactly
    np.testing.assert_array_equal(np.array(s.get_weight(path)), original)


def logits_of_fresh(text, s):
    model, tok = tiny.build(dtype=s.act_dtype)
    return Surgery(model, tok, Path(tempfile.mkdtemp()), Atlas(Path("/nonexistent"))).forward(s.tokens(text))[0]


def test_reset_restores_exact_original(s):
    base = logits(s)
    for n in ("L0/E0/N0", "L3/E7/N63", "L1/E2/N3"):
        s.scale_neuron(n, -2.0)
    s.set_neuron("L1/E2/N4", gate=np.ones(64), up=np.ones(64))
    assert len(s.edits) == 4
    assert s.reset() == 4
    np.testing.assert_array_equal(logits(s), base)
    assert s._budget_used() == 0


def test_set_weight_any_tensor_and_validation(s):
    base = logits(s)
    w = s.get_weight("layers.0.mlp.gate.weight")
    assert tuple(w.shape) == (8, 64)
    s.set_weight("layers.0.mlp.gate.weight", -w, note="flip router")
    assert np.abs(logits(s) - base).max() > 1e-3
    nw = s.get_weight("norm.weight")
    s.set_weight("norm.weight", nw * 0.5)
    with pytest.raises(SurgeryError):
        s.set_weight("norm.weight", mx.zeros((3,)))
    with pytest.raises(SurgeryError):
        s.set_weight("layers.0.mlp.gate.weight", mx.full((8, 64), float("nan")))
    with pytest.raises(SurgeryError):
        s.set_weight("layers.0.nope.weight", w)
    assert len(s.edits) == 2
    s.reset()
    np.testing.assert_array_equal(logits(s), base)


def test_out_of_range_and_budget(s):
    with pytest.raises(SurgeryError):
        s.scale_neuron("L9/E0/N0", 2.0)
    s.mem_budget = 10  # bytes: whole-tensor replacements are budgeted, single neurons are not
    with pytest.raises(SurgeryError, match="budget"):
        s.set_weight("layers.0.mlp.gate.weight", s.get_weight("layers.0.mlp.gate.weight"))
    assert not s.edits and not s._float_bytes
    s.scale_neuron("L0/E0/N0", 2.0)
    assert len(s.edits) == 1 and s._budget_used() == 0


def test_tools_render(s):
    assert "routed" in s.t_probe(TEXT, ["L1/E3/N5", "L0/E0/N1"])
    out = s.t_contrast(["aaaa bbbb", "cccc"], ["zzzz yyyy"], top_k=3)
    assert "more active on A" in out and "more active on B" in out
    assert "vector norms" in s.t_inspect_neuron("L2/E1/N9")
    assert "edit #1" in s.t_scale_neuron("L2/E1/N9", 2.0)
    assert "#1 scale_neuron L2/E1/N9" in s.t_ledger()
    assert "Undid #1" in s.t_ledger("undo")
    assert "Nothing to undo" in s.t_ledger("undo")
    ledger = [json.loads(l)["op"] for l in (s.run_dir / "ledger.jsonl").read_text().splitlines()]
    assert ledger == ["server_start", "scale_neuron", "undo"]


def test_python_tool(s):
    assert s.t_python("1 + 1").strip() == "2"
    assert s.t_python("x = S.neuron('L2/E1/N9')['out']\nprint(x.shape)").strip() == "(64,)"
    assert "(64,)" in s.t_python("x.shape")  # namespace persists
    out = s.t_python("S.set_neuron('L2/E1/N9', out=x * 0)")
    assert "edit #1" in out
    assert float(mx.abs(s.neuron_vec(2, 1, 9, "out")).max().item()) == 0.0
    assert "ZeroDivisionError" in s.t_python("1/0")
    assert "SurgeryError" in s.t_python("S.scale_neuron('L99/E0/N0', 2)")
    r = s.t_python("r = S.forward('hello world', neurons=['L1/E3/N5'], resid_layers=[2])\nr['resid'][2].shape")
    assert r.strip() == "(11, 64)"
    assert "S.forward" in s.t_python("print(S.help())")


def test_python_tool_cannot_hang_the_model(s):
    s.python_timeout = 0.3
    out = s.t_python("n = 0\nwhile True:\n    n += 1")
    assert "TimeoutError" in out
    assert s.t_python("n > 0").strip() == "True"  # the kernel is still usable afterwards


def test_python_time_limit_holds_when_the_code_swallows_exceptions(s):
    # recover-1 looped over every neuron inside `try: ... except: pass`. The one interrupt the
    # limit raised landed inside the try and was swallowed, and the call ran for 83 minutes.
    s.python_timeout = 0.3
    code = (
        "import time\n"
        "t0 = time.time()\n"
        "while time.time() - t0 < 6:\n"
        "    try:\n"
        "        for n in range(64):\n"
        "            S.neuron(f'L0/E0/N{n}')\n"
        "    except:\n"
        "        pass"
    )
    began = time.time()
    out = s.t_python(code)
    assert "TimeoutError" in out
    assert time.time() - began < 3
    assert s.t_python("1 + 1").strip() == "2"  # no interrupt is left over to land on the next call


def test_replay_restores_the_edits_that_were_active_when_the_server_stopped(s):
    s.scale_neuron("L1/E2/N3", 2.5, note="a")
    s.scale_neuron("L1/E2/N3", 5.0, note="b")  # cumulative
    s.set_neuron("L2/E0/N1", gate=np.full(64, 0.25), out=np.arange(64) / 64)
    s.scale_neuron("L3/E7/N7", 9.0)
    s.undo()  # not active at the "crash"
    want = {k: np.array(s.neuron_vec(*k)) for k in [(1, 2, 3, "out"), (2, 0, 1, "gate"), (2, 0, 1, "out"), (3, 7, 7, "out")]}
    edited = logits(s)

    model, tok = tiny.build(dtype=s.act_dtype)  # the server restarts: original weights, same run directory
    s2 = Surgery(model, tok, s.run_dir, Atlas(Path("/nonexistent")))
    assert "nothing to replay" not in (msg := s2.replay()) and "replayed 3 of 3" in msg
    for k, v in want.items():
        np.testing.assert_allclose(np.array(s2.neuron_vec(*k)), v, rtol=1e-6)
    np.testing.assert_array_equal(np.array(s2.forward(s2.tokens(TEXT))[0]), edited)
    assert [e.op for e in s2.edits] == ["scale_neuron", "scale_neuron", "set_neuron"]

    model, tok = tiny.build(dtype=s.act_dtype)
    s3 = Surgery(model, tok, Path(tempfile.mkdtemp()), Atlas(Path("/nonexistent")))
    assert "nothing to replay" in s3.replay()


def test_replay_restores_whole_tensor_edits_in_their_order(s):
    out, router = "layers.1.mlp.switch_mlp.down_proj.weight", "layers.1.mlp.gate.weight"
    s.scale_neuron("L1/E2/N3", 2.5)  # a delta inside the tensor that is replaced next
    w = np.array(s.get_weight(out))
    w[[2, 5]] *= 0.1  # two experts' outputs
    s.set_weight(out, w, note="two experts")
    s.scale_neuron("L1/E2/N4", 0.0)  # written into the replaced tensor
    s.set_weight(router, np.array(s.get_weight(router)) * 0.5)
    s.set_weight(router, np.array(s.get_weight(router)) * 0.1)  # cumulative
    s.set_weight("layers.2.mlp.gate.weight", s.get_weight("layers.2.mlp.gate.weight"))
    s.undo()  # not active at the "crash"
    with np.load(s.run_dir / s.edits[1].detail["values"]) as z:  # only what changed is kept on disk
        assert z["rows"].tolist() == [2, 5] and z["values"].shape == w[[2, 5]].shape
    want = {p: np.array(s.get_weight(p)) for p in (out, router)}
    edited = logits(s)

    model, tok = tiny.build(dtype=s.act_dtype)
    s2 = Surgery(model, tok, s.run_dir, Atlas(Path("/nonexistent")))
    assert "replayed 5 of 5" in (msg := s2.replay()) and "could not" not in msg
    for p, v in want.items():
        np.testing.assert_array_equal(np.array(s2.get_weight(p)), v)
    np.testing.assert_array_equal(np.array(s2.forward(s2.tokens(TEXT))[0]), edited)
    assert [e.op for e in s2.edits] == ["scale_neuron", "set_weight", "scale_neuron", "set_weight", "set_weight"]
    assert "layers.2.mlp.gate" not in "".join(s2._float_bytes)  # the undone replacement stays undone


def test_arrays_held_in_python_share_the_budget_with_replaced_tensors(s):
    path = "layers.0.mlp.switch_mlp.down_proj.weight"
    size = int(np.prod(s.get_weight(path).shape)) * 4
    s.mem_budget = int(2.5 * size)
    out = s.t_python(f"a = np.array(S.get('{path}'))\nb = a.copy()\nS.set('{path}', a)")  # a + b + the float copy
    assert "memory budget exceeded" in out and "arrays in your python variables (a 0.00, b 0.00)" in out
    assert not s.edits and "your python variables hold" in out  # nothing written; what is held is reported
    out = s.t_python(f"del b\nS.set('{path}', a)")
    assert "edit #1" in out and "exceeded" not in out and len(s.edits) == 1
    assert s.status()["python_arrays_gb"] == round(size / 1e9, 3)

    out = s.t_python("c = a.copy()\nd = a.copy()")  # a call that ends over the budget loses its largest variables
    assert "these variables were deleted: a 0.00, c 0.00" in out
    assert "NameError" in s.t_python("a") and "NameError" not in s.t_python("d.shape")
    assert s._budget_used() + sum(n for _, n in s._kernel_arrays()) <= s.mem_budget
    assert "[memory" not in s.t_python("del d\nx = 1\nprint(x)")  # nothing held, nothing said


class ControlTok(tiny.Tok):
    def get_added_vocab(self):
        return {"<|im_end|>": 503, "<think>": 506, "</think>": 507, "<|im_start|>": 500, "plain": 1}


def test_control_tokens_are_shown_and_read_back_in_a_safe_form(s):
    model, _ = tiny.build(dtype=s.act_dtype)
    c = Surgery(model, ControlTok(), Path(tempfile.mkdtemp()), Atlas(Path("/nonexistent")))
    assert c.show_safe("promotes '</think>' '<|im_end|>' plain <b> <think>") == "promotes '‹/think›' '‹|im_end|›' plain <b> ‹think›"
    args = {"code": "x = '‹/think›'", "texts_a": ["a ‹|im_end|› b"], "top_k": 3}
    assert c.read_safe(args) == {"code": "x = '</think>'", "texts_a": ["a <|im_end|> b"], "top_k": 3}
    assert c.read_safe(c.show_safe("<|im_start|>assistant")) == "<|im_start|>assistant"
    assert s.show_safe("</think>") == "</think>" and s.read_safe("‹/think›") == "‹/think›"  # no control tokens known


def test_baked_scales_are_part_of_the_weights_with_nothing_to_undo(s):
    path = "layers.1.mlp.switch_mlp.down_proj.weight"
    mod = s.layers[1].mlp.switch_mlp.down_proj
    before = np.array(s.get_weight(path))
    step = np.abs(np.array(mod["scales"].astype(mx.float32)))  # one quantization step, per (expert, row, group of 64)
    assert s.bake({"L1/E2/N3": 0.05, "L1/E2/N50": 0.0, "L1/E5/N4": 1.0}) == 3
    after = np.array(s.get_weight(path))

    np.testing.assert_array_equal(after[5, :, 4], before[5, :, 4])  # a factor of 1 writes the same levels back
    assert np.all(np.abs(after[2, :, 3] - 0.05 * before[2, :, 3]) <= 0.5 * step[2, :, 0] + 1e-2 * s.tol)
    assert np.all(np.abs(after[2, :, 50]) <= 0.5 * step[2, :, 0] + 1e-2 * s.tol)
    assert np.abs(after[2, :, 50]).max() < 0.02 * np.abs(before[2, :, 50]).max()
    untouched = np.ones(before.shape, bool)
    untouched[2, :, 3] = untouched[2, :, 50] = False
    np.testing.assert_array_equal(after[untouched], before[untouched])

    # nothing sets these weights apart from weights that were loaded this way
    assert type(s.layers[1].mlp.switch_mlp.down_proj) is type(mod) and not s.edits and not s._patched
    assert s.status()["active_edits"] == 0 and s.reset() == 0 and s.t_ledger().startswith("0 active edit(s)")
    assert [json.loads(l)["op"] for l in (s.run_dir / "ledger.jsonl").read_text().splitlines()] == ["server_start", "reset"]
    np.testing.assert_array_equal(np.array(s.get_weight(path)), after)

    s.scale_neuron("L1/E2/N3", 20.0)  # an edit on top is an ordinary edit; undoing it returns to the baked weights
    np.testing.assert_allclose(np.array(s.get_weight(path))[2, :, 3], before[2, :, 3], atol=10.5 * step[2, :, 0].max())
    s.undo()
    np.testing.assert_array_equal(np.array(s.get_weight(path)), after)
    s.scale_neuron("L1/E2/N3", 2.0)
    with pytest.raises(SurgeryError, match="before any edit"):
        s.bake({"L1/E2/N5": 0.5})
