"""Complete the clipping comparison at an identical 8,192-decision checkpoint."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from scripts.evaluate_final_comparison import FINAL, STOCHASTIC, parameter_hash
from scripts.evaluate_hybrid_ppo_pilot import compare, trial
from scripts.train_clipping_ablation import PREVIOUS, SEEDS
from scripts.train_hybrid_ppo_pilot import outcomes
from scripts.train_interaction_dqn import save_json


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def inputs(root):
    common = read(root / "benchmark/protocol.json")
    cases = read(root / "recovery.json")["selected"]
    assert common["full_seeds"] == list(FINAL) and common["stochastic_seeds"] == list(STOCHASTIC)
    assert common["cases"] == [[row["family"], row["seed"]] for row in cases]
    conditions = [("normal", seed, "full", [], 0.) for seed in FINAL]
    conditions += [("recovery", row["seed"], row["family"], row["prefix"], 0.) for row in cases]
    conditions += [(mode, seed, "full", [], temp) for mode, temp in (("t1", 1.), ("t05", .5))
                   for seed in STOCHASTIC]
    models, reuse = {}, {}
    for seed in SEEDS:
        for family, folder in (("warm", PREVIOUS / f"warm_t05_{seed}"), ("split", root / f"split_{seed}")):
            name = f"{family}_{seed}"
            fixed = folder / "ppo_008192.zip"
            entry = common["models"][name]
            models[name] = dict(path=str(fixed), sha256=parameter_hash(fixed))
            if Path(entry["path"]) == fixed:
                assert entry["sha256"] == models[name]["sha256"]
                reuse[name] = fixed
    return conditions, models, reuse


def run(args):
    conditions, models, reuse = inputs(args.runs)
    output = args.runs / "clipping_8192"
    if args.merge:
        fresh = read(output / "evaluated.json")
        games = defaultdict(lambda: defaultdict(list))
        for row in fresh:
            games[row["model"]][row["mode"]].append(row)
        with (args.runs / "benchmark/games.jsonl").open(encoding="utf8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["model"] in reuse:
                    games[row["model"]][row["mode"]].append(row)
        for name, modes in games.items():
            for mode, rows in modes.items():
                expected = [(seed, family) for label, seed, family, _, _ in conditions if mode == label]
                assert [(row["seed"], row["family"]) for row in rows] == expected, (name, mode)
        assert set(games) == set(models)
        results = {name: {mode: outcomes(rows) for mode, rows in modes.items()} for name, modes in games.items()}
        paired = {str(seed): {mode: compare(rows, games[f"warm_{seed}"][mode])
                              for mode, rows in games[f"split_{seed}"].items()} for seed in SEEDS}
        save_json(output / "results.json", dict(models=results, effects=paired, reused=list(reuse),
                                                  independently_played=len(fresh), budget=8192, paths=models))
        return
    output.mkdir(exist_ok=False)
    new = [(name, Path(entry["path"]), *condition) for condition in conditions
           for name, entry in models.items() if name not in reuse]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(trial, new, chunksize=3))
    save_json(output / "evaluated.json", results)
    print("Fixed-budget extra games", len(results), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--merge", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
