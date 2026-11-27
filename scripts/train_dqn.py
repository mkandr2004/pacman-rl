"""Train the first Pac-Man DQN locally and save reproducible metadata."""

import argparse
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from rl.evaluation import evaluate_model
from rl.pacman_env import PacmanEnv
from rl.wrappers import NormalizePositions


def make_training_env(max_steps, monitor_path=None):
    return Monitor(
        NormalizePositions(PacmanEnv(max_episode_steps=max_steps)),
        filename=str(monitor_path) if monitor_path else None,
        info_keywords=("score", "health", "victory", "game_over"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    args = parser.parse_args()

    if min(args.timesteps, args.max_steps, args.eval_episodes) <= 0:
        parser.error("timesteps, max-steps and eval-episodes must be positive")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    runs_path = Path(args.run_dir) if args.run_dir else Path("runs") / f"dqn_{run_name}_seed_{args.seed}"
    output_path = Path(args.output) if args.output else runs_path / "model"
    model_file = output_path if output_path.suffix == ".zip" else Path(f"{output_path}.zip")
    if model_file.exists():
        parser.error(f"Model already exists: {model_file}; use a new output path")
    runs_path.mkdir(parents=True, exist_ok=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_training_env(args.max_steps, runs_path / "training")
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=30_000,
        learning_starts=min(1_000, max(100, args.timesteps // 10)),
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(output_path)
    env.close()

    summary, episodes = evaluate_model(
        model,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed_start=args.eval_seed_start,
    )
    report = {
        "algorithm": "DQN",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "max_episode_steps": args.max_steps,
        "model": str(model_file),
        "evaluation": summary,
        "episodes": episodes,
        "environment_version": PacmanEnv.ENVIRONMENT_VERSION,
        "rewards": dict(PacmanEnv.REWARD_CONFIG),
        "frame_duration_ms": 10,
        "frames_per_action": 4,
        "evaluation_seed_start": args.eval_seed_start,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "stable-baselines3", "gymnasium", "numpy", "pygame-ce")},
        "hyperparameters": {
            **{key: getattr(model, key) for key in ("learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma", "gradient_steps", "target_update_interval", "exploration_fraction", "exploration_final_eps")},
            "train_freq": {"frequency": model.train_freq.frequency, "unit": model.train_freq.unit.value},
            "net_arch": [128, 128],
        },
    }
    report_path = runs_path / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
