"""Run reproducible local experiments, keeping every seed and its curves."""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from scripts.heuristic_baseline import run_episode as heuristic_episode
from scripts.random_baseline import run_episode as random_episode


def plot_results(root, reports, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for report in reports:
        path = root / f"seed_{report['seed']}" / "training.monitor.csv"
        with path.open(encoding="utf-8") as stream:
            next(stream)  # Monitor metadata header.
            rows = list(csv.DictReader(stream))
        steps, rewards, total = [], [], 0
        for row in rows:
            total += int(row["l"])
            steps.append(total)
            rewards.append(float(row["r"]))
        smoothed = [mean(rewards[max(0, i - 9):i + 1]) for i in range(len(rewards))]
        axes[0].plot(steps, smoothed, label=f"seed {report['seed']}")
    axes[0].set(xlabel="Training decisions", ylabel="Episode reward (10-episode mean)",
                title="DQN learning curves")
    axes[0].legend()
    labels = ["Random", "BFS", "DQN"]
    scores = [
        [item["score"] for item in baselines["random"]],
        [item["score"] for item in baselines["heuristic"]],
        [report["evaluation"]["mean_score"] for report in reports],
    ]
    axes[1].bar(labels, [mean(values) for values in scores],
                yerr=[pstdev(values) for values in scores], capsize=5)
    axes[1].set(ylabel="Evaluation score", title="Same evaluation seeds 1000-1004")
    axes[1].text(0.5, -0.23, "Bars: mean ± SD; DQN SD across training seeds",
                 ha="center", transform=axes[1].transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    root = Path("runs") / datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    print(f"EXPERIMENT={root}", flush=True)
    baselines = {
        "random": [random_episode(seed, args.max_steps) for seed in range(1000, 1005)],
        "heuristic": [heuristic_episode(seed, args.max_steps) for seed in range(1000, 1005)],
    }
    (root / "baselines.json").write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    for policy, results in baselines.items():
        print(f"{policy}: score={mean(r['score'] for r in results):.2f}, reward={mean(r['reward'] for r in results):.2f}", flush=True)
    reports = []
    for seed in args.seeds:
        run_dir = root / f"seed_{seed}"
        command = [
            sys.executable, "-m", "scripts.train_dqn",
            "--timesteps", str(args.timesteps), "--seed", str(seed),
            "--max-steps", str(args.max_steps), "--eval-episodes", "5",
            "--eval-seed-start", "1000", "--run-dir", str(run_dir),
            "--output", str(root / f"dqn_seed_{seed}"),
        ]
        print(f"TRAINING_SEED={seed}", flush=True)
        subprocess.run(command, check=True)
        reports.append(json.loads((run_dir / "report.json").read_text(encoding="utf-8")))
    summary = {
        "environment_version": "simulation-clock-v2",
        "timesteps_per_training_seed": args.timesteps,
        "max_episode_steps": args.max_steps,
        "training_seeds": args.seeds,
        "evaluation_seeds": list(range(1000, 1005)),
        "dqn_mean_score": mean(r["evaluation"]["mean_score"] for r in reports),
        "dqn_score_sd_across_training_seeds": pstdev(r["evaluation"]["mean_score"] for r in reports),
        "dqn_mean_reward": mean(r["evaluation"]["mean_reward"] for r in reports),
        "baselines": baselines,
        "training_runs": reports,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(root, reports, baselines)
    print(f"FINISHED={root}", flush=True)


if __name__ == "__main__":
    main()
