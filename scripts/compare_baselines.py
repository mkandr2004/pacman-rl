"""Compare random and heuristic policies on the same evaluation seeds."""

import argparse
from statistics import mean

from scripts.heuristic_baseline import run_episode as run_heuristic
from scripts.random_baseline import run_episode as run_random


def summarize(results):
    return {
        "mean_score": round(mean(item["score"] for item in results), 2),
        "mean_reward": round(mean(item["reward"] for item in results), 2),
        "victories": sum(item["victory"] for item in results),
        "game_overs": sum(item["game_over"] for item in results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    random_results = [
        run_random(seed, args.max_steps) for seed in range(args.episodes)
    ]
    heuristic_results = [
        run_heuristic(seed, args.max_steps) for seed in range(args.episodes)
    ]

    print({"policy": "random", **summarize(random_results)})
    print({"policy": "heuristic", **summarize(heuristic_results)})


if __name__ == "__main__":
    main()
