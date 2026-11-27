"""Gated local training for one ghost, against frozen DAgger 42."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from main.arena_models import ARENA_MODELS, load_model
from rl.anchored_ppo import AuditedMaskablePPO
from rl.ghost_env import REWARDS, SlowGhostEnv
from rl.ghost_evaluation import play_episode, summarize
from scripts.train_interaction_dqn import save_json


SEEDS = (101, 227)
CONTROL_DECISIONS = 2048
BUDGET = 32768
VALIDATION = tuple(range(95000, 95012))
TEST = tuple(range(95200, 95230))
SETTINGS = dict(n_steps=512, batch_size=64, n_epochs=4, learning_rate=3e-4, gamma=.995,
                gae_lambda=.95, ent_coef=.02, clip_range=.2, target_kl=.02)


def first_training_seed(seed):
    return 21_500_000 + 100_000 * seed


def policy_digest(model):
    sha = hashlib.sha256()
    for parameter in model.policy.state_dict().values():
        sha.update(parameter.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


class TrainingStarts(gym.Wrapper):
    def __init__(self, opponent, seed):
        super().__init__(SlowGhostEnv(opponent))
        self.next_seed = first_training_seed(seed)

    def reset(self, *, seed=None, options=None):
        seed = self.next_seed
        self.next_seed += 1
        return self.env.reset(seed=seed, options=options)

    def action_masks(self):
        return self.env.action_masks()


class RolloutMetrics(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.total = dict(captures=0, slow_captures=0, ghost_wins=0, pacman_wins=0, timeouts=0)

    def _on_step(self):
        info, = self.locals["infos"]
        self.total["captures"] += len(info["capture_events"])
        self.total["slow_captures"] += info["slow_captures"]
        self.total["ghost_wins"] += info["ghost_win"]
        self.total["pacman_wins"] += info["victory"]
        self.total["timeouts"] += info["timeout"]
        return True

    def _on_rollout_end(self):
        self.rows.append(dict(decisions=self.num_timesteps, **self.total))


def control(directory):
    opponent = load_model(ARENA_MODELS[0])
    before = policy_digest(opponent)
    env = SlowGhostEnv(opponent)
    rng = np.random.default_rng(819)
    observation, _ = env.reset(seed=95400)
    stats = dict(decisions=CONTROL_DECISIONS, ghost_wins=0, pacman_wins=0, timeouts=0,
                 captures=0, slow_captures=0, blocked_frames=0)
    episode_seed = 95400
    counts = np.zeros(env.action_space.n, dtype=np.int64)
    started = perf_counter()
    try:
        for _ in range(CONTROL_DECISIONS):
            assert env.observation_space.contains(observation)
            mask = env.action_masks()
            assert env.action_space.n == len(mask) and mask.any()
            action = rng.choice(np.flatnonzero(mask))
            counts[action] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward) and reward == sum(info["ghost_reward_terms"].values())
            assert not info["window_closed"] and not (info["timeout"] and info["ghost_win"])
            stats["captures"] += len(info["capture_events"])
            stats["slow_captures"] += info["slow_captures"]
            stats["blocked_frames"] += info["ghost_blocked_frames"]
            if terminated or truncated:
                stats["ghost_wins"] += info["ghost_win"]
                stats["pacman_wins"] += info["victory"]
                stats["timeouts"] += info["timeout"]
                episode_seed += 1
                observation, _ = env.reset(seed=episode_seed)
    finally:
        env.close()
    assert before == policy_digest(opponent)
    report = dict(**stats, actions=counts.tolist(), seconds=perf_counter()-started, passed=True,
                  seed_start=95400, pacman=ARENA_MODELS[0].name, pacman_sha256=ARENA_MODELS[0].sha256)
    save_json(directory / "control.json", report)
    print(report, flush=True)


def rank(row):
    return row["ghost_win"], row["slow_lives_taken"], row["lives_taken"], -row["timeout"]


def train(directory, seed, gate):
    check = json.loads(gate.read_text(encoding="utf-8"))
    assert check["passed"] and check["decisions"] == CONTROL_DECISIONS
    assert check["pacman_sha256"] == ARENA_MODELS[0].sha256
    opponent = load_model(ARENA_MODELS[0])
    source = policy_digest(opponent)
    config = dict(seed=seed, decisions=BUDGET, settings=SETTINGS, architecture=[128, 64],
                  reward=REWARDS, environment_version=SlowGhostEnv.version,
                  opponent=ARENA_MODELS[0].name, opponent_sha256=ARENA_MODELS[0].sha256,
                  validation_seeds=VALIDATION, reserved_opponent=ARENA_MODELS[1].name,
                  first_training_seed=first_training_seed(seed),
                  versions={package: version(package) for package in
                            ("pygame-ce", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")})
    save_json(directory / "config.json", config)
    env = TrainingStarts(opponent, seed)
    metrics = RolloutMetrics()
    elapsed = 0.
    try:
        model = AuditedMaskablePPO("MultiInputPolicy", env, seed=seed, device="cpu",
                                  policy_kwargs=dict(net_arch=dict(pi=[128, 64], vf=[128, 64])), **SETTINGS)
        initial = policy_digest(model)
        assert model.num_timesteps == 0 and not model.policy.optimizer.state_dict()["state"]
        model.save(directory / "initial.zip")
        for stop in range(8192, BUDGET + 1, 8192):
            started = perf_counter()
            model.learn(8192, reset_num_timesteps=False, callback=metrics, log_interval=None)
            elapsed += perf_counter() - started
            assert model.num_timesteps == stop
            assert all(torch.isfinite(p).all() for p in model.policy.parameters())
            name = f"ppo_{stop:06}.zip"
            model.save(directory / name)
            print(seed, stop, round(elapsed, 1), metrics.total, flush=True)
        assert policy_digest(model) != initial
        assert source == policy_digest(opponent)
        save_json(directory / "training.json", dict(actor_initial_sha256=initial,
            trained_seconds=elapsed, rollout=metrics.rows, diagnostics=model.rollout_diagnostics))
    finally:
        env.close()
    validate(directory, config, opponent)


def validate(directory, config, opponent):
    # SDL owns one game at a time. Validation only starts after training closes.
    training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    source = policy_digest(opponent)
    initial = MaskablePPO.load(directory / "initial.zip", device="cpu")
    assert initial.num_timesteps == 0 and not initial.policy.optimizer.state_dict()["state"]
    assert policy_digest(initial) == training["actor_initial_sha256"]
    validation_env = SlowGhostEnv(opponent)
    try:
        results = []
        for stop in range(8192, BUDGET + 1, 8192):
            name = f"ppo_{stop:06}.zip"
            policy = MaskablePPO.load(directory / name, device="cpu")
            assert policy.num_timesteps == stop and policy_digest(policy) != training["actor_initial_sha256"]
            games = (play_episode(validation_env, validation_seed, policy) for validation_seed in VALIDATION)
            row = dict(checkpoint_decisions=stop, path=name, **summarize(list(games)))
            results.append(row)
            print("validation", config["seed"], stop, row["ghost_win"], row["slow_lives_taken"], flush=True)
    finally:
        validation_env.close()
    assert source == policy_digest(opponent)
    save_json(directory / "results.json", dict(**config, **training, validation=results,
        selected=max(results, key=rank), frozen_verified=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--control", action="store_true")
    modes.add_argument("--seed", type=int, choices=SEEDS)
    modes.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.seed is not None and args.gate is None:
        parser.error("--gate is required before training")
    torch.set_num_threads(1)
    if args.validate_existing:
        config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        assert config["opponent_sha256"] == ARENA_MODELS[0].sha256 and config["seed"] in SEEDS
        validate(args.output, config, load_model(ARENA_MODELS[0]))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    if args.control:
        control(args.output)
    else:
        train(args.output, args.seed, args.gate)


if __name__ == "__main__":
    main()
