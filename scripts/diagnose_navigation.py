"""Measure real movement and short loops for an existing masked PPO."""
import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO
from rl.pacman_env import PacmanEnv
from rl.wrappers import NormalizePositions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = MaskablePPO.load(args.model, device="cpu")
    env = NormalizePositions(PacmanEnv(max_episode_steps=3000))
    obs, _ = env.reset(seed=args.seed)
    trace = []
    try:
        for step in range(3000):
            before = env.unwrapped._last_raw_observation
            masks = env.unwrapped.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, terminal, truncated, info = env.step(action)
            after = env.unwrapped._last_raw_observation
            trace.append(dict(step=step + 1, action=int(action), masks=masks.tolist(),
                before=before["pacman"], after=after["pacman"],
                score=info["score"], points=info["points_remaining"],
                coins=info["coins_remaining"], effects=after["effects"]))
            if terminal or truncated:
                break
    finally:
        env.close()
    positions = [tuple(item["after"]) for item in trace]
    reversals = sum(positions[i] == positions[i - 2] != positions[i - 1]
                    for i in range(2, len(positions)))
    summary = dict(seed=args.seed, decisions=len(trace),
        stationary=sum(item["before"] == item["after"] for item in trace),
        two_step_returns=reversals, distinct_positions=len(set(positions)),
        most_visited=Counter(positions).most_common(5), final=trace[-1])
    (output / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("LAST_DECISIONS", json.dumps(trace[-12:], indent=2))


if __name__ == "__main__":
    main()
