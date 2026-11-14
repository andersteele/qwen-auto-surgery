"""Batch edits, reading where a reply begins, and the score, against the tiny model."""
import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import tiny  # noqa: E402

from surgery import score  # noqa: E402
from surgery.atlas import Atlas  # noqa: E402
from surgery.core import Surgery, SurgeryError  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog, twice."


@pytest.fixture(params=["bfloat16", "float32"])
def s(request):
    model, tok = tiny.build(dtype=getattr(mx, request.param))
    s = Surgery(model, tok, Path(tempfile.mkdtemp()), Atlas(Path("/nonexistent")), mem_budget_gb=1.0)
    s.tol = 3e-2 if request.param == "bfloat16" else 1e-4
    return s


def logits(s, text=TEXT):
    return np.array(s.forward(s.tokens(text))[0])


# -- scale_neurons ---------------------------------------------------------------


def test_a_batch_is_the_same_edits_in_one_step(s):
    base = logits(s)
    out = s.t_scale_neurons([{"neuron": "L1/E2/N3", "factor": 2}, {"neuron": "L2/E0/N1", "factor": 0}], note="why")
    assert "edits #1-#2" in out and "n=2" in out
    assert [(e.op, e.target, e.detail["batch"], e.detail["note"]) for e in s.edits] == [
        ("scale_neuron", "L1/E2/N3", 1, "why"), ("scale_neuron", "L2/E0/N1", 1, "why")]
    batched = logits(s)
    s.reset()
    s.scale_neuron("L1/E2/N3", 2)
    s.scale_neuron("L2/E0/N1", 0)
    np.testing.assert_array_equal(logits(s), batched)
    s.undo(2)
    np.testing.assert_array_equal(logits(s), base)


def test_a_batch_with_a_bad_item_changes_nothing(s, monkeypatch):
    base = logits(s)
    for bad in ([("L1/E2/N3", 2), ("L99/E0/N0", 2)], [("L1/E2/N3", 2), ("L1/E2/N4", float("nan"))],
                [("L1/E2/N3", 2), ("neuron seven", 2)], [{"neuron": "L1/E2/N3"}], []):
        with pytest.raises(SurgeryError):
            s.scale_neurons(bad)
    assert not s.edits and not s._patched

    real, calls = s.scale_neuron, []

    def fails_on_third(*a, **k):  # a failure part-way through must take back what was already applied
        calls.append(a)
        if len(calls) == 3:
            raise SurgeryError("boom")
        return real(*a, **k)

    monkeypatch.setattr(s, "scale_neuron", fails_on_third)
    with pytest.raises(SurgeryError, match="boom"):
        s.scale_neurons([("L1/E2/N3", 2), ("L1/E2/N4", 3), ("L1/E2/N5", 4)])
    assert not s.edits and not s._patched
    np.testing.assert_array_equal(logits(s), base)


def test_a_batch_survives_a_restart(s):
    s.scale_neurons([("L1/E2/N3", 2.5), ("L3/E7/N7", -1.0)])
    edited = logits(s)
    model, tok = tiny.build(dtype=s.act_dtype)
    s2 = Surgery(model, tok, s.run_dir, Atlas(Path("/nonexistent")))
    assert "replayed 2 of 2" in s2.replay()
    np.testing.assert_array_equal(np.array(s2.forward(s2.tokens(TEXT))[0]), edited)
    assert [e.detail["batch"] for e in s2.edits] == [1, 1]


# -- reading where the reply begins ---------------------------------------------


def test_chat_tokens_mark_where_the_reply_begins(s):
    toks, turn, start = s.chat_tokens("hello there")
    assert start == len(toks) and toks[turn:] == [500, 504, 505, 506, 507]
    assert toks[:turn] == [500, 501, *s.tokens("hello there"), 503]
    forced, turn2, start2 = s.chat_tokens("hello there", reply="ok")
    assert (turn2, start2) == (turn, start) and forced == toks + s.tokens("ok")
    assert s.chat_tokens("hi", system="be brief")[0][:2] == [500, 502]
    assert s.chat_tokens("hi", thinking=True)[0][-3:] == [500, 504, 505]


def test_aggregates_can_be_limited_to_positions(s):
    toks, turn, _ = s.chat_tokens("hello there")
    _, whole = s.forward(toks, agg_layers=[1])
    _, head = s.forward(toks, agg_layers=[1], positions=range(turn))
    _, tail = s.forward(toks, agg_layers=[1], positions=range(turn, len(toks)))
    assert tail.ntok == len(toks) - turn and head.ntok == turn
    assert float(tail.agg[1]["routed"].sum().item()) == (len(toks) - turn) * 2  # top-2 routing
    for k in ("pos", "neg", "routed"):
        np.testing.assert_allclose(
            np.array(head.agg[1][k]) + np.array(tail.agg[1][k]), np.array(whole.agg[1][k]), rtol=1e-4, atol=1e-5)
    _, last = s.forward(toks, agg_layers=[1], positions=[-1])
    assert last.ntok == 1 and float(last.agg[1]["routed"].sum().item()) == 2
    _, again = s.forward(toks, agg_layers=[1])  # the limit does not leak into the next pass
    np.testing.assert_array_equal(np.array(again.agg[1]["routed"]), np.array(whole.agg[1]["routed"]))
    with pytest.raises(SurgeryError):
        s.forward(toks, agg_layers=[1], positions=[len(toks)])


def test_probe_and_contrast_in_chat_format(s):
    toks, turn, _ = s.chat_tokens("hello there")
    _, rec = s.forward(toks, agg_layers=[1], positions=range(turn, len(toks)))
    E = int(np.array(rec.agg[1]["routed"]).argmax())  # an expert used where the reply begins
    out = s.t_probe("hello there", [f"L1/E{E}/N5"], chat=True)
    assert "chat format" in out and "where the reply begins" in out
    assert "forced to start with 'ok'" in s.t_probe("hello there", [f"L1/E{E}/N5"], reply="ok")
    assert "chat format" not in s.t_probe("hello there", [f"L1/E{E}/N5"])

    raw = s.t_contrast(["aaaa bbbb"], ["zzzz yyyy"], top_k=3)
    chat = s.t_contrast(["aaaa bbbb"], ["zzzz yyyy"], top_k=3, chat=True)
    assert "chat format" not in raw and "chat format" in chat
    assert "A = 1 texts / 5 tokens" in chat  # only the hand-over tokens are counted
    forced = s.t_contrast(["aaaa bbbb"], ["zzzz yyyy"], top_k=3, reply_a="no", reply_b="yes")
    assert "A = 1 texts / 7 tokens, B = 1 texts / 8 tokens" in forced


# -- log-probabilities, greedy continuation ---------------------------------------


def test_logprob_is_the_log_softmax_of_the_forward_pass(s):
    toks = s.tokens(TEXT)
    last = np.array(s.forward(toks)[0]).astype(np.float64)
    lp = last - (np.log(np.exp(last - last.max()).sum()) + last.max())
    assert abs(s.logprob(toks, [7])[0] - lp[7]) < 1e-3
    two = s.logprob(TEXT, [7, 9])  # chain rule; text and ids are both accepted
    assert abs(two[0] - lp[7]) < 1e-3
    assert abs(two[1] - s.logprob(toks + [7], [9])[0]) < 5 * s.tol
    with pytest.raises(SurgeryError):
        s.logprob(toks, [])


def test_greedy_takes_the_most_likely_token_and_repeats_itself(s):
    toks = s.tokens(TEXT)
    last = np.array(s.forward(toks)[0])
    out = s.greedy(toks, 4)
    assert out == s.greedy(TEXT, 4) and len(out) <= 4
    if int(last.argmax()) not in s.tok.eos_token_ids:
        assert last[out[0]] >= last.max() - 5 * s.tol
    s.tok.eos_token_ids = {out[0]} if out else s.tok.eos_token_ids
    assert s.greedy(toks, 4) == []  # stops at the end of the turn


# -- score ------------------------------------------------------------------------

SPEC = {
    "description": "a test metric",
    "max_tokens": 6,
    "groups": [
        {"name": "g1", "role": "target", "looks_for": "anything at all", "flag": ["^"], "a": "ab", "b": "cd",
         "items": [{"prompt": "hello"}, {"prompt": "again", "heldout": True}]},
        {"name": "g2", "role": "control", "items": [{"prompt": "x", "flag": ["never-in-there"]}]},
    ],
}


@pytest.fixture
def metric(tmp_path, monkeypatch):
    (tmp_path / "t.json").write_text(json.dumps(SPEC))
    monkeypatch.setattr(score, "METRICS", tmp_path)


def test_score_reports_openings_flags_and_margins(s, metric):
    out = score.tool(s, {"metric": "t"})
    assert "g1 (target): 1/1 flagged [anything at all] · mean margin" in out and "(A = 'ab', B = 'cd')" in out
    assert "g2 (control): 0/1 flagged" in out and "again" not in out and "→" in out
    row = json.loads((s.run_dir / "scores.jsonl").read_text().splitlines()[0])
    assert row["source"] == "model" and row["metric"] == "t" and row["active_edits"] == 0
    item = row["groups"][0]["items"][0]
    assert item["prompt"] == "hello" and item["flagged"] is True and isinstance(item["margin"], float)
    toks = s.chat_tokens("hello")[0]
    assert abs(item["margin"] - (sum(s.logprob(toks, "ab")) - sum(s.logprob(toks, "cd")))) < 1e-3
    assert item["opening"] == s.tok.decode(s.greedy(toks, 6)).strip()

    assert score.tool(s, {"metric": "t"}) == out  # same weights, same score
    brief = score.tool(s, {"metric": "t", "brief": True})
    assert "→" not in brief and "1/1 flagged" in brief


def test_heldout_questions_are_for_the_operator(s, metric):
    assert "again" not in score.tool(s, {"metric": "t", "heldout": True})
    out = score.tool(s, {"metric": "t", "heldout": True, "_source": "operator"})
    assert "again" in out and "hello" not in out and "held-out" in out
    rows = [json.loads(l) for l in (s.run_dir / "scores.jsonl").read_text().splitlines()]
    assert [(r["source"], r["heldout"]) for r in rows] == [("model", False), ("operator", True)]


def test_score_moves_when_the_weights_do(s, metric):
    before = score.measure(s, score.load("t"))
    s.scale_neurons([((L, E, N), -6.0) for L in range(4) for E in range(8) for N in range(0, 64, 2)])
    after = score.measure(s, score.load("t"))
    assert after["active_edits"] == len(s.edits) > 0
    assert after["groups"][0]["mean_margin"] != before["groups"][0]["mean_margin"]
    s.reset()
    assert score.measure(s, score.load("t")) == before


def test_score_with_your_own_prompts_and_errors(s, metric):
    assert "t: a test metric" in score.tool(s, {"metric": "list"}) and "t: a test metric" in score.tool(s, {})
    out = score.tool(s, {"prompts": ["hi", "ho"], "flag": ["^"], "a": "a", "b": "b", "max_tokens": 3})
    assert "your prompts (custom): 2/2 flagged" in out
    assert "flagged" not in score.tool(s, {"prompts": ["hi"]})  # nothing to judge by: openings only
    for bad in ({"metric": "nope"}, {"metric": "../t"}, {"prompts": ["hi"], "flag": ["("]}):
        with pytest.raises(SurgeryError):
            score.tool(s, bad)


def test_python_api_has_the_new_calls(s, metric):
    assert s.t_python("len(S.chat('hi'))").strip() == "10"
    assert s.t_python("r = S.forward(S.chat('hi'), resid_layers=[2], agg_layers=[1], positions=[-1])\n"
                      "r['resid'][2].shape, float(r['agg'][1]['routed'].sum())").strip() == "((10, 64), 2.0)"
    assert s.t_python("len(S.logprob(S.chat('hi'), 'ab'))").strip() == "2"
    assert s.t_python("type(S.greedy(S.chat('hi'), 3)).__name__").strip() == "'str'"
    assert "edits #1-#2" in s.t_python("print(S.scale_neurons([('L1/E2/N3', 2), ('L1/E2/N4', 0.5)]))")
    assert "a test metric" in s.t_python("print(S.score('list'))")
    assert "g1 (target)" in s.t_python("print(S.score('t', brief=True))")
    assert "S.chat" in s.t_python("print(S.help())")


def test_a_run_can_keep_its_metrics_for_the_operator(s, metric, monkeypatch):
    monkeypatch.setattr(score, "HIDE", True)
    monkeypatch.setattr(score, "PRIVATE", {"p": json.loads((score.METRICS / "t.json").read_text())})
    assert "no ready-made metrics" in score.tool(s, {})  # the model is shown none
    for name in ("t", "p"):
        with pytest.raises(SurgeryError, match="no metric named"):
            score.tool(s, {"metric": name})
    assert "again" not in score.tool(s, {"prompts": ["hello"]})  # its own prompts still work
    n = len((s.run_dir / "scores.jsonl").read_text().splitlines())
    assert "t:" in score.tool(s, {"_source": "operator"})  # the operator still sees and runs them
    out = score.tool(s, {"metric": "p", "_source": "operator", "_quiet": True})
    assert out.replace("'p'", "'t'") == score.tool(s, {"metric": "t", "_source": "operator", "_quiet": True})
    assert len((s.run_dir / "scores.jsonl").read_text().splitlines()) == n  # and can do so without a record
