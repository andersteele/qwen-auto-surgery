import json

from surgery.atlas import Atlas, _snippet


def write(tmp_path, rows):
    p = tmp_path / "index.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return Atlas(p)


def row(L, E, N, pole, desc, **kw):
    return {"pid": f"L{L}_E{E}_N{N}", "L": L, "E": E, "N": N, "pole": pole, "cat": "mono", "desc": desc,
            "sim": None, "modes": [], "islands": [], **kw}


def test_search_ranks_by_terms_matched_and_filters_layer(tmp_path):
    a = write(tmp_path, [
        row(4, 1, 2, "+", "Words for happiness and joy"),
        row(9, 1, 2, "+", "Joyce Theater names"),
        row(9, 3, 4, "-", "Tax forms", islands=[{"name": "joyful refunds", "n": 3, "description": "", "examples": []}]),
        row(9, 5, 6, "+", "Unrelated"),
    ])
    hits = a.search("happ joy")
    assert [r["desc"] for _, r in hits] == ["Words for happiness and joy", "Joyce Theater names", "Tax forms"]
    assert [r["L"] for _, r in a.search("joy", layer=9)] == [9, 9]
    assert a.search("zzz") == [] and a.search("  ") == []


def test_describe_handles_missing_fields_and_unlabelled_neurons(tmp_path):
    a = write(tmp_path, [row(4, 1, 2, "+", "x", islands=[{"name": "", "n": None, "description": "",
                                                          "examples": [{"act": None, "text": "t"}]}])])
    assert "pole +" in a.describe(4, 1, 2)
    assert "not in the atlas" in a.describe(4, 1, 3)
    assert a.label(4, 1, 2, "+") == "x" and a.label(4, 1, 2, "-") is None


def test_snippet_marks_the_focal_token():
    toks = [[" a", 0.0], [" happy", 3.0], [" day", 0.1]]
    assert _snippet(toks, 1) == " a« happy» day"
    assert _snippet(toks, None) == " a« happy» day"
