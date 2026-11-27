"""Run the paired critic factorial benchmark on reserved game seeds."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.critic_rollouts import SPLITS, TEMPERATURES
from scripts.diagnose_dagger_robustness import REFERENCE, verify_references
from scripts.evaluate_hybrid_ppo_pilot import compare, prepare, trial
from scripts.train_critic_ablation import SEEDS
from scripts.train_hybrid_ppo_pilot import VALIDATION, outcomes
from scripts.train_interaction_dqn import save_json


FINAL = tuple(range(92000, 92100))
STOCHASTIC = FINAL[:24]
RECOVERY_BASE = 92200


def runs_and_models(root):
    models = {"dagger42": REFERENCE["dagger42"]}
    pairs = {}
    for label in TEMPERATURES:
        for seed in SEEDS:
            pair = {}
            for critic in ("random", "warm"):
                name = f"{critic}_{label}_{seed}"
                folder = root / name
                row = json.loads((folder / "result.json").read_text())
                assert (row["critic"], row["temperature"], row["seed"]) == (critic, TEMPERATURES[label], seed)
                models[name] = Path(row["historical_equivalent_checkpoint"]) if row[
                    "historical_equivalent_checkpoint"] else folder / row["selected"]["file"]
                pair[critic] = row
            random, warm = pair["random"], pair["warm"]
            for key in ("budget", "actual_steps", "settings", "rewards", "validation_normal",
                        "initial_actor", "base_critic"):
                assert random[key] == warm[key], key
            assert random["initial_critic"] != warm["initial_critic"]
            signatures = [r["rollout_diagnostics"][0]["trajectory_sha256"] for r in (random, warm)]
            assert signatures[0] == signatures[1], (label, seed)
            with (np.load(root / f"random_{label}_{seed}" / "first_rollout.npz") as a,
                  np.load(root / f"warm_{label}_{seed}" / "first_rollout.npz") as b):
                # Controls and candidates have the same actions and rewards at rollout one.
                differences = dict(gae_sign_flip=float(np.mean((a["advantages"] > 0) != (b["advantages"] > 0))),
                                   gae_correlation=float(np.corrcoef(a["advantages"].ravel(),
                                                                    b["advantages"].ravel())[0, 1]),
                                   gae_rms_change=float(np.sqrt(np.mean((a["advantages"]-b["advantages"])**2))))
            pairs[f"{label}_{seed}"] = dict(first_rollout_sha256=signatures[0], **differences)
    return models, pairs


def evaluate(args):
    verify_references()
    models, gae = runs_and_models(args.runs)
    recovery = json.loads(args.recovery.read_text())["selected"]
    recovery_seeds = {case["seed"] for case in recovery}
    assert len(recovery_seeds) == len(recovery) == 24
    assert all(case["qualified"] and case["split"] == "test" for case in recovery)
    fitted_seeds = set(VALIDATION).union(*SPLITS.values())
    assert fitted_seeds.isdisjoint(FINAL)
    assert recovery_seeds.isdisjoint(fitted_seeds.union(FINAL))
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "protocol.json", dict(full_seeds=FINAL, sampling_seeds=STOCHASTIC,
        recovery=[(case["family"], case["seed"]) for case in recovery], first_rollouts=gae,
        models={n: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for n, p in models.items()}))
    conditions = [("full", seed, "full", [], 0.) for seed in FINAL]
    conditions += [("recovery", case["seed"], case["family"], case["prefix"], 0.) for case in recovery]
    conditions += [(f"sample_{label}", seed, "full", [], temp)
                   for label, temp in TEMPERATURES.items() for seed in STOCHASTIC]
    jobs = [(name, path, *condition) for condition in conditions for name, path in models.items()]
    games = defaultdict(lambda: defaultdict(list))
    with (ProcessPoolExecutor(max_workers=args.workers) as pool,
          (args.output / "games.jsonl").open("x", encoding="utf8") as stream):
        for i, row in enumerate(pool.map(trial, jobs, chunksize=3), 1):
            games[row["model"]][row["mode"]].append(row)
            stream.write(json.dumps(row) + "\n")
            if i % 130 == 0:
                stream.flush()
                print(i, "/", len(jobs), flush=True)

    report = {}
    for name, modes in games.items():
        report[name] = {mode: dict(**outcomes(rows), vs_dagger42=compare(rows, games["dagger42"][mode]))
                        for mode, rows in modes.items()}
        print(name, [(m, row["victory"]) for m, row in report[name].items()], flush=True)
    effects = {}
    for label in TEMPERATURES:
        for seed in SEEDS:
            pair = f"{label}_{seed}"
            warm, random = games[f"warm_{pair}"], games[f"random_{pair}"]
            effects[pair] = {mode: compare(rows, random[mode]) for mode, rows in warm.items()}
    save_json(args.output / "results.json", dict(models=report, critic_effects=effects, games=len(jobs)))
    verify_references()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.recovery, args.workers, base_seed=RECOVERY_BASE, retry_short=True)
    elif args.runs is None or args.output is None:
        parser.error("--runs and --output are required when benchmarking")
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
