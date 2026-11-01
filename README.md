# geotech-rl

A procedural reinforcement-learning environment built from a real geotechnical decision: which lighting-pole foundations must be redone, given messy site records. It generates seeded instances with a physics-based truth, rejects any instance that a shortcut could solve, and trains a small open model on it with SFT and GRPO on a single rented GPU.

All data is synthetic. Total GPU spend so far: US$ 8.27.

## Results

Solved episodes on held-out instances. Open models ran through `scripts/baseline.sh`, except that the N0 baseline pools it with two online evaluations. Claude ran through `scripts/frontier.py`, with the same three tools and budget.

| Level | What it asks | Qwen3-4B | + SFT | + SFT + GRPO | Claude Opus 5.5, medium effort |
|---|---|---:|---:|---:|---:|
| N0 | warm-up, method given | 28/180 | 17/24 | **24/24** | not run |
| N1 | warm-up, method not given | 0/60 | 18/24 | **24/24** | not run |
| N2 | foreign load unit | 0/9 | 0/24 | 0/24 | not run |
| N3 | cracked collar | 0/9 | 2/24 | 3/24 | not run |
| N4 | soaked against dry tests | 0/9 | 15/24 | 24/24 | not run |
| N5 | N2 + N3 + N4, no rule written | not run | not run | 0/24 | 4/4 |
| N6 | N5 + dirty field records | not run | not run | not run | 4/4 |
| N7 | N6 + collar found by concrete volume | not run | not run | not run | 4/4 |
| N8 | N7 + decoys, 22 poles | not run | not run | not run | 3/4 |

### How to read this table

- **The model learned a script, not engineering judgement.** N0 and N1 hold no decision: one fixed script solves all of them, and wrong routes pass too. They train the format and the routine, and they do not count as tasks.
- **Reward has not yet taught a decision.** GRPO fixed one real error by reward alone: after SFT, 32% of episodes died on a single wrong token at submission. No demonstration showed that fix. It is a format fix, not judgement.
- **N4 is not a learned decision.** The demonstrations these models saw contained the rain filter. The script in this repository has since lost it, so that the next training cannot inherit the answer, but these models were not retrained. N3 passes only on control instances.
- **The ladder has no frontier-level rung.** Opus solved 15 of 16. It is calibrated for small open models. The one failure is the interesting part: on N8 it treated a booking error between two poles as a real defect.
- **A low score from a weak model proves nothing about a strong one.** Without the last column, N5 would look hard.

Confidence is limited: one training seed, and each model was tested on 8 tasks with 3 attempts each, so the 24 episodes are not independent. Reference data in [`results/`](results/).

## The task

Poles sit in concrete collars in compacted sand. An agent receives pull tests, compaction control, rainfall, an as-built list and, higher up, concrete tickets and a lab report. It must calibrate a site constant from the tests that govern, then return a safety factor and a re-found decision for every pole.

```text
H_u = K · 0.5 · γ · D · L³ / (e + L)        FS = H_u / H_nominal
```

The trap is physical. Dry sand carries suction, so dry tests overstate capacity. The foundation must hold when soaked, and only the rain record shows which tests were soaked. The grader needs every safety factor within 10% and every decision right; other sound methods, such as a rigid-rotation model, also pass.

Episodes use three tools: read a record, run Python in a sandbox, submit. Submission returns no score.

## Ladder

| Zone | Levels | Purpose |
|---|---|---|
| Warm-up | N0–N1 | Format and routine. A fixed script solves them, and they never count |
| Rule applied | N2–N4 | One mechanism each, with its rule written in the records |
| Judgement | N5–N8 | Mechanisms combined, no rule written, then dirty records and decoys |

Each level holds 1,000 train, 200 held-out and 200 out-of-distribution instances.

## Quality gates

An instance is kept only if all of this holds. `manifests/quality.json` reports each level.

- **Reference passes** with a margin under fresh test scatter, β ≥ 2 over 20 draws.
- **Every wrong route the instance contains fails**: ignoring units, keeping a cracked test, dry-only, all tests, lowest constant, latest test, sheet diameter, poured diameter, lab φ, plus constant and geometry nulls.
- **Other valid methods pass**, so the grader is not an answer key for one formula.
- **Each mechanism changes the answer.** A trap that changes nothing is decoration. `test_every_declared_mechanism_changes_the_answer` enforces this.
- **Controls exist**, about 20%: a suspicious record that is fine, so "always distrust it" fails.
- **Memorisation is measured**: the best fixed site constant passes 11–16% of held-out and 0% of out-of-distribution instances.

**Known limit.** Multiplying the dry calibration by one fixed wetting factor would pass 53–83% of held-out N4–N8. This route already requires knowing which tests are dry, and a counterfactual test is planned.

## Training

Stack: [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and verifiers, vLLM, LoRA rank 16, on one RTX 5090 shared by inference and trainer.

1. **GRPO from scratch did not start.** Training keeps raw tokens, and the model's multi-line JSON led to triple quotes inside tool calls. The parser dropped them, so no group had reward variance.
2. **SFT warm start.** 1,200 oracle demonstrations, each graded and produced with the real tools. 200 steps, 20 minutes.
3. **GRPO from the SFT model.** 64 rollouts per step, group of 8. Logprob mismatch stayed at or below 0.0029, peak memory 26.3 GiB, and train reward rose from 69% to 88%. The run stalled at step 6 of 20, when tasks served in order reached a level with no reward variance, so the result comes from six steps.

Oracle demonstrations and the open model's own rollouts are the only training data. Outputs from closed models only measure.

[`docs/RL_FAILURE_MODES.md`](docs/RL_FAILURE_MODES.md) records every failure, with its evidence and the guard that now prevents it.

## Red team

Claude Opus 5.5 attacked the sandbox and the grader for 54 tool calls, through the prompt in `redteam/`, and did not reach reward 1.0 without the engineering.

- **Blocked:** reads outside the working directory, path traversal, symlinks, subprocesses, network, `ctypes`, `gc` scans, hanging or crashing the tool, and malformed answers.
- **Real limit:** numpy's `__array_interface__` gives raw read and write of the cell's own process memory with no audit event. The truth and the seed are not in that process, and no attack reached them. The audit hook removes the easy paths and is not a security boundary, so run episodes in a disposable machine.
- **Harness only:** `partial` leaked per-pole correctness, because the red team tool returns scores on purpose. A training episode returns no score and ends at the first submission, and both are contract tests.

## Reproduce

```bash
uv sync --extra dev --python 3.12
uv run python -m pytest -q
uv run geotech-rl build --seed 1 --level 5 --out /tmp/instance
uv run geotech-rl manifest --out manifests --workers 8
uv run geotech-rl quality --out manifests
```

Evaluation and training run on a GPU machine:

```bash
bash scripts/pod_setup.sh
LEVELS=0,1,2,3,4 EPISODES=20 GROUP=3 bash scripts/baseline.sh
OUT=/root/work/sft bash scripts/sft.sh
python scripts/merge_lora.py <base model> <adapter> <merged model>
OUT=/root/work/grpo bash scripts/train.sh
```

`baseline.sh` runs a scripted preflight, then a short canary, and stops if the environment rather than the model breaks an episode.

## Layout

| Path | Content |
|---|---|
| `src/geotech_rl/` | Physics, scenario sampling, records, solver and routes, grader, sandbox, quality gates |
| `envs/geotech_poles/` | The verifiers environment and its tools |
| `scripts/` | Evaluation, demonstrations, SFT, GRPO, adapter merge, frontier reference |
| `redteam/` | Tools and prompt for adversarial testing of the sandbox and grader |
| `manifests/` | The instances of every split, as seeds and attempts |
| `results/` | Reference results |
| `docs/` | Failure modes |

## Next

- Make reward teach a decision. The post-SFT model is nearly deterministic, with entropy near 0.01, so it barely explores. N2 needs varied demonstrations or higher sampling temperature.
- Three training seeds, and the reward and curriculum ablations.
- A frontier rung, built by changing where and how the evidence appears, not by stacking physics.

## License

MIT. See [LICENSE](LICENSE).
