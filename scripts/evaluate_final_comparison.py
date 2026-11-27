"""Evaluate the five learning approaches on games held out from training and selection."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from scripts.diagnose_dagger_robustness import REFERENCE, verify_references
from scripts.evaluate_hybrid_ppo_pilot import compare, prepare, trial
from scripts.train_clipping_ablation import PREVIOUS, SEEDS
from scripts.train_hybrid_ppo_pilot import outcomes
from scripts.train_interaction_dqn import save_json
from scripts.train_ppo_from_scratch import SEEDS as SCRATCH_SEEDS, rank_scratch


FINAL = tuple(range(93800, 93900))
STOCHASTIC = FINAL[:24]
RECOVERY_BASE = 94200


def parameter_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chosen(folder):
    run = json.loads((folder / "result.json").read_text())
    return folder / run["selected"]["file"], run


def models_for(root, scratch_roots):
    models = {"dagger42": REFERENCE["dagger42"]}
    training = {}
    for seed in SEEDS:
        global_dir = root / f"global_{seed}"
        split_dir = root / f"split_{seed}"
        _, reproduction = chosen(global_dir)
        assert reproduction["historical_reproduction"] == {"4096": True, "8192": True}
        for name, folder in (("hybrid", PREVIOUS / f"random_t05_{seed}"),
                             ("warm", PREVIOUS / f"warm_t05_{seed}"),
                             ("split", split_dir)):
            variant = f"{name}_{seed}"
            models[variant], training[variant] = chosen(folder)
        event = reproduction["rollout_diagnostics"][0]["trajectory_sha256"]
        assert event == training[f"split_{seed}"]["rollout_diagnostics"][0]["trajectory_sha256"]
        assert reproduction["initial_critic"] == training[f"split_{seed}"]["initial_critic"]
        assert reproduction["initial_actor"] == training[f"split_{seed}"]["initial_actor"]
    scratch = defaultdict(list)
    for folder in scratch_roots:
        _, run = chosen(folder)
        scratch[run["seed"]].append((folder, run))
    assert set(scratch) == set(SCRATCH_SEEDS)
    for seed, stages in scratch.items():
        latest, run = max(stages, key=lambda pair: pair[1]["actual_steps"])
        picks = [(folder, row) for folder, stage in stages for row in stage["checkpoints"] if row["steps"] > 0]
        folder, row = max(picks, key=lambda pair: rank_scratch(pair[1]))
        model_name = f"scratch_{seed}"
        models[model_name] = folder / row["file"]
        training[model_name] = dict(seed=seed, temperature=1., final_stage=str(latest),
                                    actual_steps=run["actual_steps"], selected=row,
                                    training_seconds=sum(stage["training_seconds"] for _, stage in stages))
    return models, training


def evaluate(args):
    verify_references()
    models, training = models_for(args.runs, args.scratch)
    recovery = json.loads(args.recovery.read_text())["selected"]
    assert {r["seed"] for r in recovery}.isdisjoint(FINAL)
    assert len({r["seed"] for r in recovery if r["qualified"] and r["split"]=="test"}) == len(recovery) == 24
    args.output.mkdir(exist_ok=False)
    save_json(args.output / "protocol.json", dict(full_seeds=FINAL, stochastic_seeds=STOCHASTIC,
        cases=[(r["family"], r["seed"]) for r in recovery],
        models={n: dict(path=str(p), sha256=parameter_hash(p)) for n, p in models.items()},
        training={n: {key: d[key] for key in ("seed", "temperature", "actual_steps", "selected", "training_seconds")}
                  for n, d in training.items()}))
    conditions = [("normal", seed, "full", [], 0.) for seed in FINAL]
    conditions += [("recovery", row["seed"], row["family"], row["prefix"], 0.) for row in recovery]
    conditions += [(mode, seed, "full", [], temp) for mode, temp in (("t1", 1.), ("t05", .5))
                   for seed in STOCHASTIC]
    jobs = [(name, path, *case) for case in conditions for name, path in models.items()]
    games = defaultdict(lambda: defaultdict(list))
    with (ProcessPoolExecutor(max_workers=args.workers) as pool,
          (args.output / "games.jsonl").open("x", encoding="utf8") as stream):
        for i, row in enumerate(pool.map(trial, jobs, chunksize=3), 1):
            if row["mode"] == "normal" and row["victory"]:
                assert (row["points"], row["coins"], row["score"]) == (81, 11, 136)
            games[row["model"]][row["mode"]].append(row)
            stream.write(json.dumps(row) + "\n")
            if i % 130 == 0:
                stream.flush()
                print(i, "/", len(jobs), flush=True)
    report = {}
    for name, modes in games.items():
        report[name] = {mode: dict(**outcomes(rows), paired_dagger42=compare(rows, games["dagger42"][mode]))
                        for mode, rows in modes.items()}
        print(name, [(mode, line["victory"]) for mode, line in report[name].items()], flush=True)
    clip_effects = {str(seed): {mode: compare(rows, games[f"warm_{seed}"][mode])
                               for mode, rows in games[f"split_{seed}"].items()}
                    for seed in SEEDS}
    save_json(args.output / "results.json", dict(models=report, paired_clipping=clip_effects, games=len(jobs)))
    verify_references()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--scratch", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.recovery, args.workers, base_seed=RECOVERY_BASE, retry_short=True)
    elif any(value is None for value in (args.runs, args.scratch, args.output)):
        parser.error("--runs, --scratch and --output are required for the comparison")
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
