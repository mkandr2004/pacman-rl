"""Compare trained, classic and random slow ghosts on reserved paired games."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from main.arena_models import ARENA_MODELS, load_model
from rl.ghost_env import SlowGhostEnv
from rl.ghost_evaluation import play_episode, summarize
from scripts.train_interaction_dqn import save_json
from scripts.train_slow_ghost import BUDGET, SEEDS, TEST, VALIDATION, rank


OPPONENTS = {}
CONTROLLERS = {}


def checkpoint_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def play(job):
    opponent, name, checkpoint, seed = job
    if opponent not in OPPONENTS:
        OPPONENTS[opponent] = load_model(ARENA_MODELS[opponent])
    if checkpoint and checkpoint not in CONTROLLERS:
        CONTROLLERS[checkpoint] = MaskablePPO.load(checkpoint, device="cpu")
    env = SlowGhostEnv(OPPONENTS[opponent])
    try:
        stochastic = name.endswith("_sampled")
        result = play_episode(env, seed, CONTROLLERS[checkpoint] if checkpoint else name,
                              stochastic=stochastic)
    finally:
        env.close()
    assert not result["window_closed"]
    return dict(opponent=ARENA_MODELS[opponent].name, controller=name, **result)


def paired(candidate, control):
    seeds = [r["seed"] for r in candidate]
    assert seeds == [r["seed"] for r in control] == list(TEST)
    positions = np.random.default_rng(63076).integers(len(seeds), size=(10000, len(seeds)))
    result = {}
    for key in ("ghost_win", "lives_taken", "slow_lives_taken", "timeout"):
        delta = np.array([float(x[key]) - float(y[key]) for x, y in zip(candidate, control, strict=True)])
        result[key] = dict(delta=float(delta.mean()),
                          ci95=np.quantile(delta[positions].mean(axis=1), [.025, .975]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.runs / "benchmark"
    output.mkdir(exist_ok=False)
    controllers = {"classic": None, "random": None}
    validation = {}
    archived = {}
    for seed in SEEDS:
        folder = args.runs / f"seed_{seed}"
        report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        assert report["seed"] == seed and report["decisions"] == BUDGET
        assert report["frozen_verified"] and report["opponent_sha256"] == ARENA_MODELS[0].sha256
        assert report["validation_seeds"] == list(VALIDATION)
        assert report["reserved_opponent"] == ARENA_MODELS[1].name
        name = f"ppo_{seed}"
        validation[name] = report["selected"]
        controllers[name] = folder / report["selected"]["path"]
        for path in sorted(folder.glob("*.zip")):
            path.chmod(stat.S_IREAD)
            archived[str(path)] = checkpoint_hash(path)
    assert set(TEST).isdisjoint(VALIDATION) and set(TEST).isdisjoint(range(95400, 95410))
    selected_before_test = max(validation, key=lambda key: rank(validation[key]))
    if not validation[selected_before_test]["slow_lives_taken"]:
        # The fixed argmax candidate can win through its teammates alone.
        # Check sampling on the validation opponent before opening the test.
        torch.set_num_threads(1)
        env = SlowGhostEnv(load_model(ARENA_MODELS[0]))
        try:
            for name in list(validation):
                checkpoint = controllers[name]
                model = MaskablePPO.load(checkpoint, device="cpu")
                games = []
                for seed in VALIDATION:
                    games.append(play_episode(env, seed, model, stochastic=True))
                sampled = f"{name}_sampled"
                controllers[sampled] = checkpoint
                validation[sampled] = summarize(games)
        finally:
            env.close()
        eligible = [key for key, row in validation.items() if row["slow_lives_taken"] > 0 and row["ghost_win"] > 0]
        if eligible:
            selected_before_test = max(eligible, key=lambda key: rank(validation[key]))
    save_json(output / "protocol.json", dict(seeds=TEST, opponents=[entry.name for entry in ARENA_MODELS],
        held_out_opponent=ARENA_MODELS[1].name, validation_candidate=selected_before_test,
        validation=validation,
        checkpoints={key: dict(path=str(path), sha256=archived[str(path)], stochastic=key.endswith("_sampled"))
                     for key, path in controllers.items() if path},
        protected_checkpoints=archived))
    jobs = [(opponent, name, path, seed) for seed in TEST for opponent in range(len(ARENA_MODELS))
            for name, path in controllers.items()]
    rows = defaultdict(lambda: defaultdict(list))
    with (ProcessPoolExecutor(max_workers=args.workers) as pool,
          (output / "games.jsonl").open("x", encoding="utf-8") as trace):
        for done, row in enumerate(pool.map(play, jobs, chunksize=2), 1):
            rows[row["opponent"]][row["controller"]].append(row)
            trace.write(json.dumps(row) + "\n")
            if done % 40 == 0:
                trace.flush()
                print(done, "/", len(jobs), flush=True)
    summaries = {opponent: {key: summarize(games) for key, games in rules.items()}
                 for opponent, rules in rows.items()}
    contrasts = {opponent: {key: {control: paired(games, rules[control]) for control in ("classic", "random")}
                            for key, games in rules.items() if key.startswith("ppo_")}
                 for opponent, rules in rows.items()}
    save_json(output / "results.json", dict(summaries=summaries, paired=contrasts, games=len(jobs)))
    for opponent, rules in summaries.items():
        print(opponent, {key: (value["ghost_win"], value["slow_lives_taken"], value["timeout"])
                         for key, value in rules.items()}, flush=True)


if __name__ == "__main__":
    main()
