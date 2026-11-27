"""Evaluate or visually demonstrate a saved DQN model."""

import argparse
import json

from stable_baselines3 import DQN

from rl.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    model = DQN.load(args.model, device="cpu")
    summary, episodes = evaluate_model(
        model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_mode="human" if args.render else None,
    )
    print(json.dumps({"summary": summary, "episodes": episodes}, indent=2))


if __name__ == "__main__":
    main()
