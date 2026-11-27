"""View a DQN or evaluate all candidates on the declared validation seeds."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from rl.imitation_eval import play_network, summary
from rl.interaction_dqn import load_dqn
from scripts.train_interaction_dqn import save_json


_MODELS = {}


def evaluate_job(job):
    path, seed = job
    if path not in _MODELS:
        torch.set_num_threads(1)
        _MODELS[path] = load_dqn(path)
    game = play_network(_MODELS[path], seed, danger=False, measure_topology=True)
    if game["victory"] and (game["points"], game["coins"], game["score"]) != (81, 11, 136):
        raise RuntimeError("Victory without the original game's full collection.")
    return game


def summarize(games):
    result = summary(games)
    result.update(games_with_lemon=sum(game["lemons"] > 0 for game in games),
                  games_with_opening=sum(game["walls"] > 0 for game in games),
                  useful_openings=sum(game["useful_wall_events"] for game in games),
                  games_with_useful_opening=sum(game["useful_wall_events"] > 0 for game in games))
    return result


def validate(directory, workers):
    report = json.loads((directory / "report.json").read_text())
    path = directory / "validation.json"
    if path.exists():
        raise FileExistsError(path)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for filename in report["checkpoints"]:
            checkpoint = directory / filename
            games = list(pool.map(evaluate_job, [(checkpoint, seed) for seed in report["validation_seeds"]]))
            outcome = summarize(games)
            results.append(dict(checkpoint=filename, outcome=outcome, games=games))
            print(checkpoint.name, "wins", outcome["victory"], "lemons", outcome["games_with_lemon"],
                  "useful openings", outcome["games_with_useful_opening"], "score", outcome["score"], flush=True)
    selected = max(results, key=lambda row: (
        row["outcome"]["victory"], row["outcome"]["games_with_useful_opening"],
        row["outcome"]["points"] + row["outcome"]["coins"], -row["outcome"]["max_stagnation"]))
    victory = next((game for game in selected["games"] if game["victory"]), None)
    if victory is not None:
        torch.set_num_threads(1)
        replay = play_network(load_dqn(directory / selected["checkpoint"]), victory["seed"], danger=False, record=True)
        if not replay["victory"] or (replay["steps"], replay["score"]) != (victory["steps"], victory["score"]):
            raise RuntimeError("The validation win did not reproduce.")
        save_json(directory / "validation_win_trace.json", replay)
    save_json(path, dict(seeds=report["validation_seeds"], selected=selected["checkpoint"], results=results))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", type=Path)
    group.add_argument("--model", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=82000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.experiment:
        validate(args.experiment, args.workers)
    else:
        torch.set_num_threads(1)
        result = play_network(load_dqn(args.model), args.seed, danger=False, render=args.render, measure_topology=True)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
