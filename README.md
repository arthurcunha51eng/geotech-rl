# geotech-rl

[![tests](https://github.com/arthurcunha51eng/geotech-rl/actions/workflows/tests.yml/badge.svg)](https://github.com/arthurcunha51eng/geotech-rl/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23861866.svg)](https://doi.org/10.5281/zenodo.23861866)

A procedural reinforcement-learning environment built from a real geotechnical decision: which lighting-pole foundations must be redone, given messy site records. It generates seeded instances with a physics-based truth, discards any instance that a listed wrong route can solve, and trains a small open model with SFT and GRPO on one rented GPU.

All data is synthetic. GPU spend so far: US$ 8.27.

## Results

Solved episodes on held-out instances: Qwen3-4B before and after training, then Claude Opus 5.5. A dash means not run.

| Level | What it asks | Base | SFT | SFT + GRPO | Opus 5.5 |
|---|---|---:|---:|---:|---:|
| N0 | warm-up, method given | 28/180 | 17/24 | 24/24 | — |
| N1 | warm-up, method not given | 0/60 | 18/24 | 24/24 | — |
| N2 | foreign load unit | 0/9 | 0/24 | 0/24 | — |
| N3 | cracked collar | 0/9 | 2/24 | 3/24 | — |
| N4 | soaked against dry tests | 0/9 | 15/24 | 24/24 | — |
| N5 | N2 + N3 + N4, no rule written | — | — | 0/24 | 4/4 |
| N6 | N5 + dirty field records | — | — | — | 4/4 |
| N7 | N6 + caved collar, seen in the pour volume | — | — | — | 4/4 |
| N8 | N7 + decoys, 22 poles | — | — | — | 3/4 |

Open models ran through `scripts/baseline.sh`. The base model's N0 and N1 cells predate the ladder recalibration; every other cell uses the current ladder. Claude Opus 5.5 ran at medium effort through `scripts/frontier.py`, with the same three tools and budget. Machine-readable data: [`results/`](results/).

![Solved episodes by level and model. A fixed script solves N0 and N1 and nothing above them outside control instances. The open model after SFT and GRPO solves N0, N1 and N4, and scores zero on N2 and N5. Claude Opus 5.5 solves 15 of 16 instances from N5 to N8.](docs/ladder.png)

### How to read this table

- **The model learned a script, not engineering judgement.** N0 and N1 hold no decision: the demonstration script solves 60 of 60 on both, and wrong routes pass too. They train format and routine, and they never count as tasks.
- **That same script solves nothing above N1 outside control instances**, 0 of 49 on N2 and 0 of 46 on N4. Any score above that line has to come from somewhere else.
- **Reward has not yet taught a decision.** GRPO fixed one real error by reward alone: after SFT, 32% of episodes died on a single wrong token at submission. No demonstration showed that fix, and it is a format fix, not judgement.
- **N4 is not a learned decision.** The demonstrations these models saw contained the rain filter. The script in this repository has since lost it, so that the next training cannot inherit the answer, but these models were not retrained. N3 passes only on control instances.
- **The ladder has no frontier-level rung.** Opus solved 15 of 16. It is calibrated for small open models. The one failure is the interesting part: on N8 it treated a booking error between two poles as a real defect.
- **Sample sizes are small.** One training seed. The 24 episodes are 8 tasks with 3 attempts each, so 24/24 has a 95% Wilson interval of 68–100% per task, and 0/24 reaches up to 32%. Opus 15/16 is 72–99%, with no control instances among them.

## Quickstart

```bash
uv sync --extra dev --python 3.12
uv run python -m pytest -q
uv run geotech-rl build --seed 1 --level 5 --out /tmp/instance
```

`/tmp/instance/data` now holds the records an agent receives, and `truth.json` sits beside it. To rebuild every instance and run the gates, about 15 minutes on eight workers and byte-identical to `manifests/`:

```bash
uv run geotech-rl manifest --out manifests --workers 8
uv run geotech-rl quality --out manifests
```

Evaluation and training need a GPU machine:

```bash
bash scripts/pod_setup.sh
LEVELS=0,1,2,3,4 EPISODES=20 GROUP=3 bash scripts/baseline.sh
OUT=/root/work/sft bash scripts/sft.sh
python scripts/merge_lora.py <base model> <adapter> <merged model>
OUT=/root/work/grpo bash scripts/train.sh
```

`baseline.sh` runs a scripted preflight, then a short canary, and stops if the environment rather than the model breaks an episode.

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

An instance is kept only if all of this holds. [`manifests/quality.json`](manifests/quality.json) reports each level, and `scripts/fixed_script.py` measures the fixed-script baseline.

- **The reference passes** with a margin under fresh test scatter, β ≥ 2 over 20 draws.
- **Every wrong route the instance contains fails** with the same margin: ignoring units, keeping a cracked test, dry-only, all tests, lowest constant, latest test, sheet diameter, poured diameter, lab φ, plus constant and geometry nulls.
- **Other valid methods pass**, so the grader is not an answer key for one formula.
- **Each mechanism changes the answer.** A trap that changes nothing is decoration, and `test_every_declared_mechanism_changes_the_answer` enforces this.
- **Controls exist**, 12–23% of held-out on N2–N8: a suspicious record that is fine, so "always distrust it" fails.
- **Memorisation is measured**: the best fixed site constant passes 11–16% of held-out and 0% of out-of-distribution instances.

## Training

Stack: [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and verifiers, vLLM, LoRA rank 16, on one RTX 5090 shared by inference and trainer. Base model: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), Apache 2.0.

1. **GRPO from scratch did not start.** Training keeps raw tokens, and the model's multi-line JSON led to triple quotes inside tool calls. The parser dropped them, so no group had reward variance.
2. **SFT warm start.** 1,200 oracle demonstrations, each graded and produced with the real tools. 200 steps, 20 minutes.
3. **GRPO from the SFT model.** 64 rollouts per step, group of 8. Logprob mismatch stayed at or below 0.0029, peak memory 26.3 GiB, and train reward rose from 69% to 88%. The run stalled at step 6 of 20, when tasks served in order reached a level with no reward variance, so the result comes from six steps.

Oracle demonstrations and the open model's own rollouts are the only training data. Outputs from closed models only measure.

The adapters are on Hugging Face: [SFT](https://huggingface.co/arthurcs01/geotech-rl-qwen3-4b-sft) and [SFT + GRPO](https://huggingface.co/arthurcs01/geotech-rl-qwen3-4b-sft-grpo).

[`docs/RL_FAILURE_MODES.md`](docs/RL_FAILURE_MODES.md) records every failure, with its evidence and the guard that now prevents it.

## Known limits

- **One training seed and small samples.** See the intervals above.
- **No engineering decision was learned.** The post-SFT model is nearly deterministic, with entropy near 0.01, so it barely explores.
- **A single wetting factor leaks.** Multiplying the dry calibration by one fixed factor would pass 53–83% of held-out and 62–91% of out-of-distribution N4–N8. This route already requires knowing which tests are dry, and a counterfactual test is planned.
- **The gates reshape the distribution.** On train and held-out, acceptance per attempt is 5–8% on N4–N8 against 17–23% on N0–N3, and the resulting shift in physical parameters is not yet measured.
- **Engineering validity is not independently reviewed.** The data is synthetic, and no licensed reviewer outside the project has checked the instances.
- **The sandbox is not a security boundary.** See below.

## Red team

Claude Opus 5.5 attacked the sandbox and the grader for 54 tool calls, through the prompt in `redteam/`, and did not reach reward 1.0 without the engineering.

- **Blocked:** reads outside the working directory, path traversal, symlinks, subprocesses, network, `ctypes`, `gc` scans, hanging or crashing the tool, and malformed answers.
- **Real limit:** numpy's `__array_interface__` gives raw read and write of the cell's own process memory with no audit event. It could in principle disable the audit hook, which the attack did not complete. The truth and the seed are not in that process, and no attack reached them. The hook removes the easy paths, so run episodes in a disposable machine.
- **Harness only:** `partial` leaked per-pole correctness, because the red team tool returns scores on purpose. A training episode returns no score and ends at the first submission, and both are contract tests.

## AI use

The engineering problem, its physical mechanism, and the review and publication decisions are the author's. Claude Code with Claude Opus 5.5 wrote most of the code, ran the experiments and drafted the documentation under the author's direction. Claude Opus 5.5 also served as the frontier reference and the red team. Its outputs never became training data.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff), or:

```bibtex
@software{cunha_da_silva_geotech_rl_2026,
  author  = {Cunha da Silva, Arthur},
  title   = {geotech-rl: a procedural reinforcement-learning environment
             for geotechnical judgement},
  year    = {2026},
  doi     = {10.5281/zenodo.23861866},
  url     = {https://github.com/arthurcunha51eng/geotech-rl}
}
```

## Layout

`src/geotech_rl/` holds physics, sampling, records, solver and routes, grader, sandbox and gates. `envs/geotech_poles/` is the verifiers environment. `scripts/` and `configs/` run evaluation, demonstrations, training, the frontier reference and the figure. `redteam/` tests the sandbox and grader. `manifests/` lists every instance as seeds and attempts.

## Next

- Make reward teach a decision. N2 needs varied demonstrations or a higher sampling temperature.
- Three training seeds, and the reward and curriculum ablations.
- A frontier rung, built by changing where and how the evidence appears, not by stacking physics.

## License

MIT. See [LICENSE](LICENSE). The base model keeps its own Apache 2.0 licence.
