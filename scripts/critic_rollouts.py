"""Generate Monte Carlo data from the frozen DAgger 42 actor at each PPO temperature."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rl.imitation_env import make_imitation_env
from rl.tempered_ppo import TemperedCategorical
from scripts.diagnose_dagger_robustness import REFERENCE, verify_references
from scripts.pretrain_dagger_critic import GAMMA
from scripts.train_interaction_dqn import save_json


TEMPERATURES = {"t1": 1., "t05": .5}
SPLITS = {"train": range(91300, 91332), "validation": range(91400, 91412),
          "holdout": range(91500, 91512)}
ACTOR = None


def init_worker():
    global ACTOR
    torch.set_num_threads(1)
    ACTOR = MaskablePPO.load(REFERENCE["dagger42"], device="cpu")
    ACTOR.policy.set_training_mode(False)
    ACTOR.policy.requires_grad_(False)


def collect(job):
    label, split, seed, folder = job
    torch.manual_seed(seed)
    policy = ACTOR.policy
    policy.action_dist = TemperedCategorical(ACTOR.action_space.n, TEMPERATURES[label])
    env = make_imitation_env(danger=False)
    try:
        obs, _ = env.reset(seed=seed)
        columns = {key: [] for key in obs}
        actions, masks, rewards, log_probs = [], [], [], []
        for _ in range(3000):
            mask = env.unwrapped.action_masks()
            with torch.no_grad():
                distribution = policy.get_distribution(policy.obs_to_tensor(obs)[0], action_masks=mask)
                action = distribution.get_actions()
                log_probs.append(distribution.log_prob(action).item())
            action = action.item()
            assert mask[action]
            for key, value in obs.items():
                columns[key].append(value)
            actions.append(action)
            masks.append(mask)
            obs, reward, terminal, truncated, info = env.step(action)
            rewards.append(reward)
            if terminal or truncated:
                break
        returns = np.empty(len(rewards), dtype=np.float32)
        total = 0.
        for index in reversed(range(len(rewards))):
            total = rewards[index] + GAMMA * total
            returns[index] = total
        np.savez_compressed(folder / f"{seed}.npz",
                            **{key: np.stack(values) for key, values in columns.items()},
                            rewards=rewards, mc_returns=returns, masks=np.stack(masks),
                            actions=actions, log_probs=np.asarray(log_probs, dtype=np.float32))
        return dict(temperature=label, split=split, seed=seed, steps=len(rewards),
                    terminal=terminal, truncated=truncated, victory=info["victory"],
                    reward=sum(rewards), score=info["score"])
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    verify_references()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    jobs = []
    for label in TEMPERATURES:
        folder = args.output / label
        folder.mkdir()
        jobs.extend((label, split, seed, folder) for split, seeds in SPLITS.items() for seed in seeds)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        episodes = list(pool.map(collect, jobs))
    save_json(args.output / "episodes.json", dict(gamma=GAMMA, episodes=episodes,
                                                  seconds=perf_counter() - started))
    for label in TEMPERATURES:
        rows = [row for row in episodes if row["temperature"] == label]
        print(label, sum(row["steps"] for row in rows), "decisions",
              sum(row["victory"] for row in rows), "/", len(rows), "wins", flush=True)


if __name__ == "__main__":
    main()
