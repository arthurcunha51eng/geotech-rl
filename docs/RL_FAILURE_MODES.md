# Failure modes found while building this environment

Each entry happened here, with the numbers measured at the time. Most were bugs in the environment, not weaknesses of the
model: a surprising result is treated as a bench bug until shown otherwise.

## Environment bugs that looked like a weak model

| Symptom | Cause | Evidence | Guard |
|---|---|---|---|
| 0 of 600 solved | The sandbox blocked `zstandard`, which `pandas.read_csv` imports lazily | 44–64% of Python runs blocked | `scripts/preflight.py`, `sandbox_blocks` in `--strict` |
| Preflight green, model at 0 | The scripted agent passed `comment="#"` because its author knew the format | 151 of 175 failed cells came from comment lines | Plain `read_csv` check, written as a naive first attempt |
| `agent_completed` on 57% of episodes | The context limit ended the run without an error | 34 of 60 episodes | `ending()` reads the last call; overflow aborts the batch |
| Episodes cut with budget left | The turn limit was lower than the budget told to the agent | 17 of 60 episodes | Contract test: the stated budget is the binding limit |
| Answers unrelated to the truth | Cells had no `print()`, so the model saw nothing and invented numbers | 189 of 191 cells; 0 of 87 answers came from a computation | Final expression is echoed; `computed_answers` metric |
| Submissions rejected | JSON inside a string argument forced escaped quotes | 23 of 67 submit calls | Object argument |

## Training failures

| Symptom | Cause | Evidence | Fix |
|---|---|---|---|
| Train reward 1.1%, eval reward 21.7% on the same model | Eval re-renders history as compact JSON; training keeps the model's raw multi-line tokens, which led it to write triple quotes inside the tool-call JSON. The parser drops the call and the episode ends in silence | 228 of 296 Python calls. I first blamed `top_p`, tested that, and was wrong | Supervised warm start on oracle demonstrations |
| No batch admitted | Groups with no reward variance carry no gradient | Warning `No admitted train payload` | Read it as an alarm; warm start |
| Trainer out of memory while loading | Default `float32` weights take 16 GB | Crash at load on a shared 32 GB card | `bfloat16` |
| Evaluation took 11 minutes instead of 3 | Automatic concurrency derives from the context limit and starts near 5 episodes | Logged in-flight count | Fixed `initial_inflight` |
| GPU billed 29 minutes with no step | Tasks are served in order; a run of zero-signal groups stalled training | Stuck at step 6 of 20 | Shuffle tasks; alarm on a stalled step, not just on the final eval |
| Cleanup left 9 GB on the GPU | The trainer renames its process, so `pkill` by name missed it | `nvidia-smi --query-compute-apps` | Check GPU memory before relaunching |

## Results that looked better than they were

| Claim | Why it did not hold | Guard |
|---|---|---|
| "100% on held-out" on the first levels | One fixed script solved them; every wrong data route passed too, so there was no decision to make | Levels without a defeated route are marked `warmup` and never count |
| "100% out of distribution" | The script only copies two numbers from the README, so range shift tested almost nothing | Report the fixed-script baseline next to any score |
| Traps on the upper levels | The crack and the foreign unit always fell on dry tests, which the right route discards anyway | `test_every_declared_mechanism_changes_the_answer` |
| Soaked-test selection | Soaked tests are the weakest, so "take the lowest" passed 88–97% without reading the rain | `weakest` null route; a dry test stopped before failure |
| Rejection sampling | Controls have fewer routes to defeat, so acceptance raised their share from 20% to 43% | Control drawn from the seed, outside the attempt loop |
| Partial credit as progress | 12% partial score with zero computed answers: guesses land inside a 10% band | Compare with `computed_answers` |
| A low score from a weak model | It says nothing about a frontier model, which solved a level the weak model scored 0 on in 66 seconds | Measure the reference model on every level that claims difficulty |

## Generator

| Symptom | Cause | Guard |
|---|---|---|
| Eight workers at 100% CPU for two hours | With very high friction, suction barely matters, so the wrong route stopped failing and almost no seed was accepted, with no limit on the loop | `MAX_SEEDS_PER_INSTANCE`, with a test |
| Safety factors 1.5 and 2.5 silently rejected | A regex took the final period and a bare `except` hid it; the diversity metric caught it | Valid routes run unshielded |
| A fixed constant passed 28–35% of held-out sites | The site constant range was too narrow | `quality.memorisation`; wider ranges brought it to about 15% |
