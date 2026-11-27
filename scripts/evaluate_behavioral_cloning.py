"""Evaluate a saved checkpoint on fresh seeds or play one visible episode."""

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from rl.imitation_eval import play_network, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=54000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="Use the original PPO observation contract.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--scenarios", type=Path, help="Manifest supplying the diagnostic prefixes.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("At least one episode is required.")
    if args.legacy and args.scenarios:
        parser.error("Diagnostic prefixes use the imitation environment.")
    torch.set_num_threads(1)
    config = {} if args.legacy else json.loads((args.model.parent / "config.json").read_text())
    model = MaskablePPO.load(args.model, device="cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    options = dict(danger=config.get("danger", False), stochastic=args.stochastic,
                   render=args.render, legacy=args.legacy)
    games = []
    for seed in range(args.seed, args.seed + args.episodes):
        row = play_network(model, seed, **options)
        games.append(row)
        print(seed, row["score"], row["victory"], row["steps"], flush=True)
        if row["victory"]:
            replay = play_network(model, seed, **{**options, "render": False}, record=True)
            if not replay["victory"] or replay["steps"] != row["steps"]:
                raise RuntimeError("A reported win could not be reproduced.")
            (args.output / f"win_seed_{seed}.json").write_text(json.dumps(replay, indent=2))
    diagnostics = []
    if args.scenarios:
        episodes = json.loads(args.scenarios.read_text())["episodes"]
        by_length = {len(row["prefix"]): row["prefix"] for row in episodes if row["prefix"]}
        for length, prefix in sorted(by_length.items()):
            for seed in (54040, 54041, 54042):
                try:
                    diagnostics.append(play_network(model, seed, **options, prefix=prefix))
                except RuntimeError as error:
                    if str(error) != "Scenario prefix terminated; invalid scenario":
                        raise
                    diagnostics.append(dict(seed=seed, prefix_length=length, setup_terminated=True))
    report = dict(model=str(args.model), **options, summary=summary(games),
                  games=games, scenarios=diagnostics)
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
