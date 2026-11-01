"""Command line: build one instance, export a Harbor dataset, or summarise a range of seeds."""
import argparse
import collections
import json
import tempfile
import time
from pathlib import Path

from . import harbor, instance, quality
from .scenario import LEVELS


def _seeds(text: str) -> range:
    first, last = text.split(":")
    return range(int(first), int(last))


def _levels(text: str) -> list[int]:
    return list(LEVELS) if text == "all" else [int(x) for x in text.split(",")]


def cmd_build(args) -> None:
    out = Path(args.out)
    built = instance.build(args.seed, args.level, out / "data")
    (out / "truth.json").write_text(json.dumps(built.truth, indent=1), encoding="utf-8")
    print(f"level {args.level} seed {args.seed}: control={built.scenario.control or 'none'} attempt={built.scenario.attempt}")


def cmd_harbor(args) -> None:
    for level in _levels(args.levels):
        for seed in _seeds(args.seeds):
            harbor.export(seed, level, Path(args.out))
    print(f"exported to {args.out}")


def cmd_stats(args) -> None:
    summary = {}
    for level in _levels(args.levels):
        attempts, chars, controls = [], [], collections.Counter()
        passed = collections.Counter()
        seeds = _seeds(args.seeds)
        for seed in seeds:
            with tempfile.TemporaryDirectory() as tmp:
                built = instance.build(seed, level, Path(tmp))
                chars.append(sum(p.stat().st_size for p in Path(tmp).rglob("*") if p.is_file()))
            attempts.append(built.scenario.attempt + 1)
            controls[built.scenario.control or "none"] += 1
            for route, result in built.routes.items():
                passed[route] += result["reward"]
        summary[level] = dict(
            mean_attempts=round(sum(attempts) / len(attempts), 1), max_attempts=max(attempts),
            mean_data_bytes=round(sum(chars) / len(chars)), max_data_bytes=max(chars), controls=dict(controls),
            route_pass_rate={k: round(v / len(seeds), 3) for k, v in passed.items()})
    print(json.dumps(summary, indent=1))


def cmd_manifest(args) -> None:
    out = Path(args.out)
    for split, count in (("train", args.train), ("heldout", args.heldout), ("ood", args.ood)):
        for level in _levels(args.levels) if count else ():
            start = time.time()
            rows, tried = quality.build_manifest(split, level, count, args.draws, args.workers)
            quality.write_manifest(out / f"{split}_n{level}.jsonl", rows)
            print(f"{split} n{level}: {len(rows)} instances from {tried} seeds in {time.time() - start:.0f} s", flush=True)


def cmd_quality(args) -> None:
    out = Path(args.out)
    report = {}
    for level in _levels(args.levels):
        rows = {s: quality.read_manifest(out / f"{s}_n{level}.jsonl") for s in ("train", "heldout", "ood")}
        betas = [r["beta_reference"] for r in rows["train"]]
        wrong = [r["beta_wrong"] for r in rows["train"] if r["beta_wrong"] is not None]
        report[level] = dict(
            beta_reference_p5=round(float(sorted(betas)[len(betas) // 20]), 2),
            beta_wrong_p5=round(float(sorted(wrong)[len(wrong) // 20]), 2) if wrong else None,
            single_wetting_factor=quality.single_factor(rows["train"], dict(heldout=rows["heldout"], ood=rows["ood"])),
            nulls=quality.null_table(rows["heldout"]),
            train=quality.diversity(rows["train"]),
            memorisation_heldout=quality.memorisation(rows["train"], rows["heldout"]),
            memorisation_ood=quality.memorisation(rows["train"], rows["ood"]))
        print(f"level {level} done", flush=True)
    (out / "quality.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(prog="geotech-rl")
    commands = parser.add_subparsers(required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--out", default="manifests")
    manifest.add_argument("--levels", default="all")
    manifest.add_argument("--train", type=int, default=1000)
    manifest.add_argument("--heldout", type=int, default=200)
    manifest.add_argument("--ood", type=int, default=200)
    manifest.add_argument("--draws", type=int, default=20)
    manifest.add_argument("--workers", type=int, default=8)
    manifest.set_defaults(run=cmd_manifest)
    report = commands.add_parser("quality")
    report.add_argument("--out", default="manifests")
    report.add_argument("--levels", default="all")
    report.set_defaults(run=cmd_quality)
    build = commands.add_parser("build")
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--level", type=int, required=True)
    build.add_argument("--out", required=True)
    build.set_defaults(run=cmd_build)
    dataset = commands.add_parser("harbor")
    dataset.add_argument("--seeds", default="0:20")
    dataset.add_argument("--levels", default="all")
    dataset.add_argument("--out", required=True)
    dataset.set_defaults(run=cmd_harbor)
    stats = commands.add_parser("stats")
    stats.add_argument("--seeds", default="0:200")
    stats.add_argument("--levels", default="all")
    stats.set_defaults(run=cmd_stats)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
