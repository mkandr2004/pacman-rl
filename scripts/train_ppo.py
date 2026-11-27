"""Train masked PPO on the unchanged original map; keep validation/test distinct."""

import argparse
import json
from datetime import datetime
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.pacman_env import PacmanEnv
from rl.wrappers import NormalizePositions


def make_env(logfile=None, render=False, potential_shaping=False):
    torch.set_num_threads(1)
    env = PacmanEnv(
        render_mode="human" if render else None,
        max_episode_steps=3000,
    )
    if potential_shaping:
        from rl.shaping import PotentialShaping
        env = PotentialShaping(env, gamma=0.995)
    return Monitor(NormalizePositions(env),
        filename=logfile, info_keywords=("score", "victory", "game_over"))


def evaluate(model, episodes=20, render=False):
    results = []
    env = make_env(render=render)
    try:
        for seed in range(1000, 1000 + episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for step in range(3000):
                action, _ = model.predict(obs, deterministic=True,
                    action_masks=env.unwrapped.action_masks())
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            results.append(dict(seed=seed, score=info["score"], reward=total,
                victory=info["victory"], game_over=info["game_over"],
                truncated=truncated, steps=step + 1))
    finally:
        env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=262144)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help="Evaluate an existing checkpoint instead of training")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.model:
        results = evaluate(MaskablePPO.load(args.model, device="cpu"), args.episodes, args.render)
        print(json.dumps(results, indent=2))
        return
    root = Path("runs") / datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    config = dict(algorithm="MaskablePPO", environment_version=PacmanEnv.ENVIRONMENT_VERSION,
        map="maps/map_2.txt", seed=args.seed, requested_timesteps=args.timesteps,
        workers=args.workers, net_arch=dict(pi=[256, 256], vf=[256, 256]),
        learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, ent_coef=0.02, clip_range=0.2,
        max_episode_steps=3000, frames_per_action=4, frame_ms=10,
        rewards=dict(PacmanEnv.REWARD_CONFIG),
        validation_seed_start=2000, test_seeds=list(range(1000, 1000 + args.episodes)),
        versions={name: version(name) for name in ("torch", "stable-baselines3", "sb3-contrib", "gymnasium")},
        map_sha256=sha256(Path("maps/map_2.txt").read_bytes()).hexdigest())
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"RUN={root}", flush=True)
    train_env = SubprocVecEnv([
        partial(make_env, str(root / f"worker_{index}"))
        for index in range(args.workers)
    ], start_method="spawn")
    train_env.seed(args.seed)
    validation_env = make_env()
    validation_env.reset(seed=2000)
    callback = MaskableEvalCallback(validation_env, n_eval_episodes=5,
        eval_freq=max(1, 16384 // args.workers), log_path=str(root),
        best_model_save_path=str(root), deterministic=True)
    checkpoint = CheckpointCallback(save_freq=max(1, 32768 // args.workers),
        save_path=str(root / "checkpoints"), name_prefix="ppo")
    model = MaskablePPO("MultiInputPolicy", train_env,
        policy_kwargs=dict(net_arch=config["net_arch"]),
        **{key: config[key] for key in ("learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "ent_coef", "clip_range")},
        seed=args.seed, device="cpu", verbose=1)
    print(f"PARAMETERS={sum(p.numel() for p in model.policy.parameters())}", flush=True)
    try:
        model.learn(args.timesteps, callback=[callback, checkpoint], log_interval=8)
        model.save(root / "final_model")
    finally:
        train_env.close()
        validation_env.close()
    selected = MaskablePPO.load(root / "best_model.zip", device="cpu") if (root / "best_model.zip").exists() else model
    results = evaluate(selected, args.episodes)
    report = dict(config=config, actual_timesteps=model.num_timesteps,
        selection="validation mean reward only", test_results=results,
        wins=sum(r["victory"] for r in results),
        mean_score=float(np.mean([r["score"] for r in results])))
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
