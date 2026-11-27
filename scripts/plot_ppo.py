"""Plot saved PPO validation measurements without rerunning evaluations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    root = Path(args.run_dir)
    with np.load(root / "evaluations.npz") as data:
        steps = data["timesteps"]
        rewards = data["results"]
        means = rewards.mean(axis=1)
        deviations = rewards.std(axis=1)
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(steps, means, marker="o", label="Validation mean reward")
    axis.fill_between(steps, means - deviations, means + deviations, alpha=0.2,
                      label="± SD across 5 validation episodes")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if (root / "report.json").exists():
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        if "baseline_validation_reward" in report:
            axis.axhline(report["baseline_validation_reward"], linestyle="--",
                         color="black", label="Original checkpoint on corrected environment")
    axis.set(xlabel="Training decisions", ylabel="Episode reward",
             title=f"MaskablePPO — {config.get('environment_version', config.get('rules', 'original level'))}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "validation_curve.png", dpi=160)
    plt.close(fig)
    print(root / "validation_curve.png")


if __name__ == "__main__":
    main()
