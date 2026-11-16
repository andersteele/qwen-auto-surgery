# Locating and removing the refusal on censored subjects

A worked example of operator-side analysis with `surgery/lab.py`. It was run once, as `prc-lab-1`, on the original weights with no agent attached. What it found is in [docs/run-notes.md](../../docs/run-notes.md) and [docs/findings.md](../../docs/findings.md).

Start a server with no agent, then run the steps in order. Each file says at its top what it needs and what it shows. Steps marked "outside" are plain numpy on the arrays the previous step saved.

```sh
bin/serve prc-lab-2                                   # original weights; leave it running

bin/lab lab/prc/session.py                            # shared setup: question sets, helpers, the full measurement
bin/lab lab/prc/baseline.py                           # openings and residual streams on the original weights
.venv/bin/python lab/prc/directions.py runs/prc-lab-2/lab       # outside: mean-difference directions per layer
bin/lab lab/prc/sweep.py                              # remove one direction at a time; read the replies
bin/lab lab/prc/contexts.py                           # does a short instruction unlock it?
bin/lab lab/prc/gradients.py                          # gradient of a refusal score; per-site and per-expert shares
.venv/bin/python lab/prc/attribution.py runs/prc-lab-2/lab      # outside: where the score is built
.venv/bin/python lab/prc/grad_directions.py runs/prc-lab-2/lab  # outside: the direction it reads, per layer
bin/lab lab/prc/steer.py                              # remove that direction, or push against it

bin/replay prc-1 --cut "june 14" --ask QUESTION_PLACEHOLDER --save-request runs/prc-lab-2/lab/replay_body.json
bin/lab lab/prc/harvest.py                            # the original weights' answers inside the prc-1 conversation
bin/lab lab/prc/fit.py                                # fit a low-rank update to those answers
bin/lab lab/prc/write.py                              # write it into the weights and measure, thinking off and on
```

To measure a run in which the model made the edits itself (`prc-4`), wait for the model to finish, then:

```sh
bin/lab lab/prc/session.py
bin/lab lab/prc/check.py                              # the metric with thinking off and on, and sampled replies to the June 4 question
```

Both leave their names in the Python session the model's `python` tool uses, so do this after the model's last turn.

`harvest.py` needs a recorded conversation in which the original weights answer; `prc-1`'s is not in this repository, so use one of your own runs.

| File | What it is |
|---|---|
| `questions.json` | The question sets: censored subjects, comparable subjects elsewhere, uncensored Chinese subjects, everyday questions, and more on the 1989 protests. None is a question of `metrics/prc.json`, and none touches its held-out subjects. Odd-numbered questions are held back from every fit. |
| `session.py` | `chat`, `kind_of` (a rough sort of an opening into refusal, official account or answer), `refusal_score`, `drift`, and `full_eval`, which runs the `prc` metric with thinking off and on. Replies to the harmful requests are never kept or printed, only whether each was declined and by what margin. |

The steps share one Python session in the server, so a later step uses names an earlier one defined. After a server restart, run `session.py` again. The files were consolidated from the session that produced `prc-lab-1` and have not been re-run end to end since. `session.py` and `check.py` were run on `prc-4`, where they reproduced the `prc-lab-1` replies word for word.
