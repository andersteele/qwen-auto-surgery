# bin/lab lab/prc/harvest.py     (after session.py, on the original weights; about 30 minutes)
# What the original weights say about each question INSIDE the prc-1 conversation, where they do
# answer. These are the targets for the fit: the model's own words, from its own weights.
#
# Needs the conversation as a request body the server can read:
#     bin/replay prc-1 --cut "june 14" --ask QUESTION_PLACEHOLDER --save-request runs/<run>/lab/replay_body.json
# The 58,000-token conversation is read once and its state reused for every question (the HTTP
# endpoint re-reads it each time, at 2.5 minutes a question).
import copy
from mlx_lm import server as _mls
from mlx_lm.models.cache import make_prompt_cache

_body = json.loads((OUT / "replay_body.json").read_text())


def _prompt(question):
    msgs = copy.deepcopy(_body["messages"])
    msgs[-1]["content"] = [{"type": "text", "text": question}]
    _mls.process_message_content(msgs)
    return list(S._s.tok.apply_chat_template(msgs, add_generation_prompt=True, tools=_body["tools"], tokenize=True,
                                             **(_body.get("chat_template_kwargs") or {})))


_a, _b = _prompt("Alpha one?"), _prompt("Why beta two, exactly?")
K = next(i for i, (x, y) in enumerate(zip(_a, _b)) if x != y)      # where the question starts
print("prompt tokens", len(_a), "| shared prefix", K)
t0 = time.time()
S._s.model.eval()
PREFIX = make_prompt_cache(S._s.model)
for i in range(0, K, 2048):
    mx.eval(S._s.model(mx.array([_a[i:min(i + 2048, K)]]), cache=PREFIX))
print("conversation read in", round(time.time() - t0), "s")
_stop = {int(t) for t in (getattr(S._s.tok, "eos_token_ids", None) or ())}


def in_context(question, max_tokens=520):
    toks = _prompt(question)
    assert toks[:K] == _a[:K]
    cache = copy.deepcopy(PREFIX)
    x, out = mx.array([toks[K:]]), []
    for _ in range(max_tokens):
        t = int(mx.argmax(S._s.model(x, cache=cache)[0, -1]).item())
        if t in _stop: break
        out.append(t); x = mx.array([[t]])
    text = S._s.tok.decode(out)
    thought, sep, reply = text.partition("</think>")
    return {"reasoning": thought.strip(), "content": reply.strip() if sep else "", "completion_tokens": len(out),
            "prompt_tokens": len(toks), "tool_calls": ["tool"] if "<tool_call>" in text else []}


target = next(g for g in SPEC["groups"] if g["role"] == "target")
plan = ([("tiananmen", i, q) for i, q in enumerate(Q["tiananmen"])] + [("censored", i, q) for i, q in enumerate(Q["censored"])]
        + [("metric", i, it["prompt"]) for i, it in enumerate(target["items"])])
path = OUT / "harvest.jsonl"
done = {json.loads(l)["label"] for l in path.open()} if path.exists() else set()
t0 = time.time()
with path.open("a") as f:
    for group, i, q in plan:
        label = f"harvest:{group}:{i}"
        if label in done: continue
        f.write(json.dumps({"label": label, "q": q, "temperature": 0, **in_context(q)}, ensure_ascii=False) + "\n"); f.flush()
rows = [json.loads(l) for l in path.open()]
print(len(rows), "answers in", round(time.time() - t0), "s; with a reply:", sum(bool(r["content"]) for r in rows))
del PREFIX
