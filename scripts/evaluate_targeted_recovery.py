"""Evaluate candidates selected solely on validation against frozen DAgger."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from scripts.diagnose_dagger_robustness import MODELS, REFERENCE, run_episode, verify_references
from scripts.train_targeted_recovery import ARMS, validation_rank


def trial(job):
    name, path, seed, temperature, family, prefix = job
    torch.set_num_threads(1)
    if name not in MODELS:
        MODELS[name] = MaskablePPO.load(path, device="cpu")
    result, trace, _ = run_episode(name, seed, temperature, prefix=prefix)
    if not prefix and result["victory"]:
        assert (result["points"], result["coins"], result["score"]) == (81, 11, 136)
    return dict(**result, family=family, oscillations=int(np.count_nonzero(
        (trace[2:, 6:8] == trace[:-2, 6:8]).all(1) & (trace[2:, 6:8] != trace[1:-1, 6:8]).any(1))),
                non_greedy=int((trace[:, 4] != trace[:, 5]).sum()))


def describe(rows):
    return dict(games=len(rows), wins=sum(row["victory"] for row in rows),
                **{key: sum(r[key] for r in rows)/len(rows) for key in
                   ("score", "points", "coins", "health", "steps", "oscillations", "non_greedy")})


def compare(rows, base):
    assert [(r["family"], r["seed"]) for r in rows] == [(r["family"], r["seed"]) for r in base]
    diff = np.array([int(a["victory"]) - int(b["victory"]) for a, b in zip(rows, base)])
    groups = defaultdict(list)
    for row, value in zip(rows, diff):
        groups[row["seed"]].append(value)
    totals = np.array([sum(group) for group in groups.values()])
    counts = np.array([len(group) for group in groups.values()])
    indices = np.random.default_rng(697).integers(len(totals), size=(10000, len(totals)))
    interval = np.quantile(totals[indices].sum(1)/counts[indices].sum(1)*100., (.025, .975))
    return dict(gained=int((diff > 0).sum()), lost=int((diff < 0).sum()),
                points=float(diff.mean()*100.), interval_95=interval.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    verify_references()
    manifest = json.loads((args.data / "manifest.json").read_text())
    candidates = dict(REFERENCE)
    selected = {}
    arms = defaultdict(list)
    for path in args.runs:
        settings = json.loads((path / "result.json").read_text())
        arms[settings["arm"]].append((path, settings))
    for arm in ARMS:
        path, settings = max(arms[arm], key=lambda pair: validation_rank(
            pair[1]["selected"], pair[1]["baseline_validation"]))
        chosen = settings["selected"]
        candidates[arm] = path / chosen["checkpoint"]
        selected[arm] = dict(training_seed=settings["seed"], checkpoint=chosen,
                             validation_baseline=settings["baseline_validation"])
    # The holdout is qualified before the first test and excludes exact training observations.
    test = json.loads((args.data / "holdout.json").read_text())["episodes"]
    unique_test = list({(r["seed"], tuple(r["prefix"])): r for r in reversed(test)}.values())[::-1]
    assert len(unique_test) > 0
    full = manifest["full_test"]
    assert len(full) == 100 and len(set(full)) == 100
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(full=full, stochastic=full[:40],
                    recovery=[(r["family"], r["seed"]) for r in unique_test],
                    models={n: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for n,p in candidates.items()},
                    selections=selected, omitted_duplicate_states=len(test)-len(unique_test))
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    jobs = [(arm,str(path),seed,0.,"full",[]) for seed in full for arm,path in candidates.items()]
    jobs += [(arm,str(path),seed,1.,"exploration",[]) for seed in protocol["stochastic"]
             for arm,path in candidates.items()]
    jobs += [(arm,str(path),row["seed"],0.,row["family"],row["prefix"]) for row in unique_test
             for arm,path in candidates.items()]
    games = defaultdict(list)
    with (ProcessPoolExecutor(max_workers=args.workers) as pool,
          (args.output / "games.jsonl").open("x") as stream):
        for index, row in enumerate(pool.map(trial, jobs, chunksize=2), 1):
            games[row["model"]].append(row)
            stream.write(json.dumps(row) + "\n")
            if index % 50 == 0:
                stream.flush()
                print(index, "/", len(jobs), flush=True)
    results = {}
    for name, rows in games.items():
        result = {}
        for group, lo, hi in (("full", 0, 100), ("exploration", 100, 140), ("recovery", 140, len(rows))):
            result[group] = dict(**describe(rows[lo:hi]), paired=compare(rows[lo:hi], games["dagger42"][lo:hi]))
        result["families"] = {family: describe([r for r in rows[140:] if r["family"] == family])
                              for family in sorted({r["family"] for r in rows[140:]})}
        results[name] = result
        print(name, *((key, result[key]["wins"]) for key in ("full", "exploration", "recovery")), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    verify_references()


if __name__ == "__main__":
    main()
