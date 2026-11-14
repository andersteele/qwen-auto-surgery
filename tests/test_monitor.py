import json
import os

from surgery.atlas import Atlas
from surgery.monitor import Run, Tail, edits_text, health, status_text


def write(path, rows, mode="a"):
    with path.open(mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def toks(rid, text):
    return [{"t": "tok", "id": rid, "s": s, "time": 10.0 + i} for i, s in enumerate(text)]


def test_tail_reads_only_complete_lines_and_survives_truncation(tmp_path):
    p = tmp_path / "x.jsonl"
    t = Tail(p)
    assert t.read() == []
    p.write_text('{"a": 1}\n{"a": 2')
    assert t.read() == [{"a": 1}]
    with p.open("a") as f:
        f.write('}\nnot json\n{"a": 3}\n')
    assert t.read() == [{"a": 2}, {"a": 3}]
    p.write_text('{"a": 9}\n')
    assert t.read() == [{"a": 9}]


def test_transcript_separates_thinking_text_and_tool_calls(tmp_path):
    stream = [
        {"t": "server_start", "time": 1.0},
        {"t": "start", "id": 1, "user": "Increase your happiness.", "prompt_tokens": 900, "cached_tokens": 0,
         "state": "reasoning", "time": 9.0},
        # tags split across tokens, a call spanning lines
        *toks(1, ["Let me ", "look.\nSecond thought", "</th", "ink>", "\n\nI will search.", "<tool_call>",
                  "\n<function=atlas_search>\n<parameter=query>\nhapp\n</parameter>\n</function>\n", "</tool_call>"]),
        {"t": "end", "id": 1, "tokens": 40, "seconds": 2.0, "finish": "tool_calls", "time": 20.0},
        {"t": "tool", "tool": "atlas_search", "args": {"query": "happ"}, "ok": True, "seconds": 0.1,
         "result": "\n".join(f"line {i}" for i in range(30)), "active_edits": 0, "budget_gb": 0.0, "time": 21.0},
    ]
    write(tmp_path / "stream.jsonl", stream)
    run = Run(tmp_path)
    lines, _ = run.poll()
    kinds = [k for k, _ in run.entries]
    text = {k: [l.plain for kk, l in run.entries if kk == k] for k in set(kinds)}
    assert text["user"] == ["▌ Increase your happiness."]
    assert text["think"] == ["Let me look.", "Second thought"]
    assert "I will search." in text["text"]
    assert text["call"] == ["→ calls <function=atlas_search> <parameter=query> happ </parameter> </function> "]
    assert text["tool"][0].startswith('⚙ atlas_search {"query": "happ"}')
    assert len(text["result"]) == 15 and "16 more lines" in text["result"][-1]
    assert run.state.phase == "between steps" and run.state.turn == 1 and run.state.total_tokens == 8
    assert run.state.rate == 20.0
    assert "40 tokens in 2.0s · 20.0 tok/s" in text["meta"][-1]

    run.show_thinking = False
    shown = [l.plain for l in run.rerender()]
    assert "Let me look." not in shown and "I will search." in shown


def test_ledger_tracks_undo_reset_and_server_restart(tmp_path):
    write(tmp_path / "ledger.jsonl", [
        {"op": "server_start"},
        {"seq": 1, "op": "scale_neuron", "target": "L9/E190/N48", "factor": 2.0, "out_norm": [0.5, 1.0], "note": "more"},
        {"seq": 2, "op": "scale_neuron", "target": "L4/E1/N2", "factor": 0.0, "out_norm": [0.5, 0.0]},
        {"op": "undo", "undid_seq": 2, "target": "L4/E1/N2"},
        {"seq": 3, "op": "set_weight", "target": "language_model.model.norm.weight", "changed_values": 2048, "delta_norm": 1.5},
    ])
    run = Run(tmp_path)
    _, changed = run.poll()
    assert changed and [e.active for e in run.state.edits] == [True, False, True]
    assert run.state.active_edits == 2
    panel = edits_text(run, Atlas(tmp_path / "none.jsonl")).plain
    assert "#1 scale_neuron L9/E190/N48  ×2" in panel and "“more”" in panel and "2,048 values" in panel
    assert "2 active / 3 made" in status_text(run, "r").plain

    write(tmp_path / "ledger.jsonl", [{"op": "reset", "undid": 2}, {"op": "server_start"}])
    run.poll()
    assert run.state.active_edits == 0


def test_health_tells_working_from_idle_from_hung_from_gone(tmp_path):
    """The question the monitor must always answer: is the model working, waiting, stuck or dead?"""
    def at(now, *rows):
        write(tmp_path / "stream.jsonl", rows)
        run.poll()
        symbol, sentence, _ = health(run, now=now)
        return symbol, sentence

    run = Run(tmp_path)
    assert "waiting for the model server to start" in at(0.0)[1]

    (tmp_path / "server.pid").write_text(str(os.getpid()))
    assert "loading the model · 12s" in at(112.0, {"t": "server_start", "time": 100.0})[1]
    assert at(121.0, {"t": "ready", "time": 120.0})[1].startswith("idle · model alive, heartbeat 1s ago")

    start = {"t": "start", "id": 1, "prompt_tokens": 9000, "cached_tokens": 1000, "state": "normal", "time": 130.0}
    assert "reading context · 0/8,000 tokens" in at(131.0, start)[1]
    assert "reading context · 4,096/8,000" in at(133.0, {"t": "prompt", "id": 1, "done": 4096, "total": 8000, "time": 132.0})[1]
    symbol, sentence = at(141.0, *[{"t": "tok", "id": 1, "s": "x", "time": 135.0 + i / 10} for i in range(50)])
    assert sentence.startswith("generating · ") and symbol in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    # tokens stop arriving and nothing else does either: say so, and stop the spinner
    assert at(150.0)[0] == "◌"
    symbol, sentence = at(200.0)
    assert symbol == "!" and "no sign of life from the model for 60s while generating" in sentence

    # a long tool call is not a hang: it is named, with how long it has been running
    symbol, sentence = at(300.0, {"t": "end", "id": 1, "tokens": 50, "seconds": 5, "finish": "tool_calls", "time": 201.0},
                          {"t": "tool_start", "tool": "python", "time": 202.0})
    assert symbol == "⚙" and sentence == "running python · started 98s ago"

    at(302.0, {"t": "tool", "tool": "python", "ok": True, "seconds": 99, "result": "", "time": 301.0},
       {"t": "crash", "error": "[METAL] Insufficient Memory", "active_edits": 9, "time": 301.5})
    text = "\n".join(l.plain for _, l in run.entries)
    assert "generation crashed: [METAL] Insufficient Memory" in text and "its 9 edit(s) are still loaded" in text
    assert "crashes   1 (recovered)" in status_text(run, "r").plain

    symbol, sentence = at(303.0, {"t": "dead", "error": "[METAL] Insufficient Memory", "time": 302.5})
    assert symbol == "✖" and "model thread is dead" in sentence and "--resume" in sentence

    (tmp_path / "server.pid").write_text("999999")  # no such process
    symbol, sentence = at(304.0)
    assert symbol == "✖" and "model server is not running" in sentence
