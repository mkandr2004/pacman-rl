"""Run the locked 200-seed evaluation and paired certified recovery cases."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rl.imitation_eval import play_network, summary
from scripts.evaluate_dagger import metrics_after_wall
from scripts.pretrain_dagger_critic import FROZEN, verify_frozen
from scripts.train_ppo_finetuning import FULL_TEST, FULL_VALIDATION


MODELS = {}


def trial(job):
    name, filename, seed, family, prefix = job
    torch.set_num_threads(1)
    if filename not in MODELS:
        MODELS[filename] = MaskablePPO.load(filename, device="cpu")
    result = play_network(MODELS[filename], seed, danger=False, prefix=prefix,
                          measure_topology=True)
    if not prefix and result["victory"] and (result["points"], result["coins"]) != (81, 11):
        raise RuntimeError("Victory verification failed.")
    return dict(model=name, family=family, **result)


def aggregate(rows):
    if not rows:
        raise ValueError("No completed evaluations.")
    result = summary(rows)
    result["escapes"] = sum(row["escape_step"] is not None for row in rows)
    result.update(metrics_after_wall(rows))
    for key in ("wall_events", "useful_wall_events", "newly_reachable_items"):
        result[key] = sum(row[key] for row in rows)
    return result


def paired(reference, candidate, resamples):
    difference = np.array([int(row["victory"]) for row in candidate]) - reference
    intervals = np.quantile(difference[resamples].mean(axis=1), [.025,.975]) * 100
    return dict(gained=int((difference > 0).sum()), lost=int((difference < 0).sum()),
                difference_points=100 * difference.mean(),
                interval_95_points=intervals.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, default=Path("runs/ppo_recovery_verified/qualified.json"))
    parser.add_argument("--extra-recovery", type=Path, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    verify_frozen()
    candidates = {"BC42": Path("runs/frozen_bc_86_7/selected.zip"),
                  "DAgger42": Path("runs/dagger_select_bc42_run42/full_update.zip"),
                  "DAgger241": FROZEN}
    rows = json.loads(args.recovery.read_text())
    for path in args.extra_recovery:
        extra = json.loads(path.read_text())
        if any(row["split"] != "test" for row in extra):
            raise ValueError("The additional file must contain only held-out test cases.")
        rows.extend(extra)
    recovery = [row for row in rows if row["qualified"] and row["split"] == "test"]
    reserve = set(FULL_TEST).union(row["seed"] for row in recovery)
    used = set(FULL_VALIDATION).union(row["seed"] for row in rows if row["split"] != "test")
    if not recovery:
        raise ValueError("No certified recovery test cases.")
    for run in args.runs:
        result = json.loads((run / "result.json").read_text())
        assert tuple(result["reserved_full_test"]) == FULL_TEST
        used.update(row[1] for row in result["validation_cases"])
        candidates[run.name] = run / result["selected"]
    if used.intersection(reserve) or len(candidates) != len(args.runs)+3:
        raise RuntimeError("Overlapping splits or duplicate evaluation models.")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(full_seeds=FULL_TEST, recovery_cases=[(r["family"],r["seed"]) for r in recovery],
                    map_hash=hashlib.sha256(Path("maps/map_2.txt").read_bytes()).hexdigest(),
                    models={name: dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                            for name,path in candidates.items()})
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2))
    jobs = [(name,str(path),seed,"full",[]) for seed in FULL_TEST for name,path in candidates.items()]
    jobs += [(name,str(path),row["seed"],row["family"],row["prefix"])
             for row in recovery for name,path in candidates.items()]
    games = {name: defaultdict(list) for name in candidates}
    started = perf_counter()
    with (ProcessPoolExecutor(max_workers=args.workers) as pool,
          (args.output / "games.jsonl").open("x", encoding="utf8") as stream):
        for step, row in enumerate(pool.map(trial,jobs,chunksize=2), start=1):
            games[row["model"]][row["family"]].append(row)
            stream.write(json.dumps(row) + "\n")
            if step % 90 == 0:
                stream.flush()
                print(step, "/", len(jobs), "evaluations", flush=True)
    rng = np.random.default_rng(241)
    full_reference = np.array([int(row["victory"]) for row in games["DAgger241"]["full"]])
    recovery_reference = np.array([int(row["victory"]) for family,group in games["DAgger241"].items()
                                   if family != "full" for row in group])
    full_samples = rng.integers(len(full_reference),size=(10000,len(full_reference)))
    recovery_samples = rng.integers(len(recovery_reference),size=(10000,len(recovery_reference)))
    outcome = {}
    for name, families in games.items():
        full = families["full"]
        assert [row["seed"] for row in full] == list(FULL_TEST)
        resumes = [row for family,group in families.items() if family != "full" for row in group]
        assert len(resumes) == len(recovery)
        outcome[name] = dict(full=aggregate(full), recovery=aggregate(resumes),
                             by_family={key: aggregate(values) for key,values in families.items() if key != "full"},
                             paired_full=paired(full_reference,full,full_samples),
                             paired_recovery=paired(recovery_reference,resumes,recovery_samples))
        print(name,outcome[name]["full"]["victory"],"full;",
              outcome[name]["recovery"]["victory"],"recovery wins",flush=True)
    (args.output / "results.json").write_text(json.dumps(
        dict(**manifest, seconds=perf_counter()-started, results=outcome), indent=2))


if __name__ == "__main__":
    main()
