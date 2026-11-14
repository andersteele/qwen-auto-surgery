"""The parts of bin/replay that need neither pi nor a model: finding the cut, and putting
a request's system prompt and tools back to what the session recorded."""

import pytest

from surgery import replay


def msg(role, text="", **extra):
    return {"type": "message", "message": {"role": role, "content": [{"type": "text", "text": text}], **extra}}


def system(sections=None, added=(), removed=()):
    return {"type": "message", "message": {
        "role": "system", "content": "", "sections": sections or {},
        "toolsAdded": [dict(t) for t in added], "toolsRemoved": [{"name": n} for n in removed],
    }}


def tool(name, description="d", parameters=None):
    return {"name": name, "description": description, "parameters": parameters or {"type": "object"}}


def wire(t):
    return {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}


THEN = system({"preamble": "old rules\n", "cwd": "<cwd>\n/runs/a/workspace\n</cwd>"}, added=[tool("probe"), tool("python", "old python"), tool("ledger")])
SESSION = [{"type": "session", "cwd": "/runs/a/workspace"}, THEN, msg("user", "the task"), msg("assistant", "working"),
           msg("user", "what happened on june 14?"), msg("assistant", "an answer", usage={"input": 10, "cacheRead": 90})]


def test_cut_index_wants_exactly_one_user_message():
    assert replay.cut_index(SESSION, "june 14") == 4
    with pytest.raises(SystemExit):
        replay.cut_index(SESSION, "no such words")
    with pytest.raises(SystemExit):
        replay.cut_index(SESSION, "t")  # in both user messages
    with pytest.raises(SystemExit):
        replay.cut_index(SESSION, "working")  # only the assistant said it


def test_recorded_state_follows_later_system_entries():
    later = system({"cwd": "<cwd>\n/tmp/x\n</cwd>"}, added=[tool("python", "new python")], removed=["python"])
    sections, tools = replay.recorded_state(SESSION[:4])
    assert sections["preamble"] == "old rules\n" and [t["name"] for t in tools] == ["probe", "python", "ledger"]
    sections, tools = replay.recorded_state(SESSION[:4] + [later])
    assert sections == {"preamble": "old rules\n", "cwd": "<cwd>\n/tmp/x\n</cwd>"}
    assert [(t["name"], t["description"]) for t in tools] == [("probe", "d"), ("ledger", "d"), ("python", "new python")]


def test_restore_puts_back_the_recorded_system_prompt_and_tools():
    # what pi appends when it continues the copy today: the prompt file and one tool have changed since the run
    today = system({"preamble": "new rules\n", "cwd": "<cwd>\n/tmp/x\n</cwd>"}, added=[tool("python", "new python")], removed=["python"])
    before, after = SESSION[:4], SESSION[:4] + [today, msg("user", "what happened on june 14?")]
    body = {
        "messages": [{"role": "system", "content": "new rules\n\n\n<cwd>\n/tmp/x\n</cwd>"}, {"role": "user", "content": "the task"},
                     {"role": "assistant", "content": "working"}, {"role": "user", "content": "what happened on june 14?"}],
        "tools": [wire(tool("probe")), wire(tool("ledger")), wire(tool("python", "new python"))],
        "temperature": 1,
    }
    out, note = replay.restore(body, before, after)
    assert "put back" in note
    assert out["messages"][0]["content"] == "old rules\n\n\n<cwd>\n/runs/a/workspace\n</cwd>"
    assert out["messages"][1:] == body["messages"][1:]
    assert [(t["function"]["name"], t["function"]["description"]) for t in out["tools"]] == [("probe", "d"), ("python", "old python"), ("ledger", "d")]
    assert body["messages"][0]["content"].startswith("new rules")  # the input is not changed


def test_restore_leaves_a_request_it_cannot_account_for():
    body = {"messages": [{"role": "system", "content": "something pi assembled another way"}, {"role": "user", "content": "q"}],
            "tools": [wire(tool("probe"))]}
    out, note = replay.restore(body, SESSION[:4], SESSION[:4])
    assert out is body and "could not match" in note


def test_restore_says_so_when_nothing_changed():
    body = {"messages": [{"role": "system", "content": replay.assemble(THEN["message"]["sections"])}, {"role": "user", "content": "q"}],
            "tools": [wire(t) for t in THEN["message"]["toolsAdded"]]}
    out, note = replay.restore(body, SESSION[:4], SESSION[:4])
    assert out == body and "unchanged" in note
