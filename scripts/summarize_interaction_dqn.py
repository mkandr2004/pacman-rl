"""Summarize exploration, collection and validation across DQN runs."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.train_interaction_dqn import CHECKPOINT_INTERVAL, save_json


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def analyze(directory):
    report = json.loads((directory / "report.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    progress = records(directory / "progress.jsonl")
    episodes = records(directory / "episodes.jsonl")
    phases = []
    for stop in range(CHECKPOINT_INTERVAL, report["decisions"] + 1, CHECKPOINT_INTERVAL):
        start = stop - CHECKPOINT_INTERVAL
        windows = [row for row in progress if start < row["step"] <= stop]
        games = [row for row in episodes if start < row["decision"] <= stop]
        totals = {key: sum(row[key] for row in windows) for key in (
            "decisions", "random_actions", "points", "coins", "lemons", "walls", "lives_lost",
            "oscillations", "wins", "deaths", "time_limits")}
        phases.append(dict(start_exclusive=start, end=stop, epsilon=windows[-1]["epsilon"],
                           explored_tiles_median=median(row["tiles"] for row in windows),
                           episodes_finished=len(games), mean_final_score=mean(r["score"] for r in games),
                           median_final_score=median(r["score"] for r in games), **totals))
    assert sum(row["decisions"] for row in phases) == report["decisions"]
    return dict(report=report, validation=validation, phases=phases, progress=progress)


def draw(series, path):
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained")
    collection, lemons, oscillations, scores = axes.flat
    for data in series:
        label = f"Seed {data['report']['seed']}"
        rows = data["progress"]
        x = [row["step"] / 1000 for row in rows]
        collection.plot(x, [1000*(r["points"]+r["coins"])/r["decisions"] for r in rows], label=label)
        lemons.step(x, [r["lemons"] for r in rows], where="post", label=label)
        oscillations.plot(x, [100*r["oscillations"]/r["decisions"] for r in rows], label=label)
        validated = data["validation"]["results"]
        scores.plot([int(r["checkpoint"].split("_")[1].split(".")[0])/1000 for r in validated],
                    [r["outcome"]["score"] for r in validated], marker="o", label=label)
    for ax, title in zip(axes.flat, (
        "Objets collectés par 1 000 décisions", "Citrons par fenêtre de 4 096 décisions",
        "Décisions en aller-retour (%)", "Score moyen en validation (30 parties)",
    )):
        ax.set_title(title)
        ax.set_xlabel("Décisions d’entraînement (milliers)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    collection.legend(frameon=False)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    series = [analyze(directory) for directory in args.experiments]
    args.output.mkdir(exist_ok=False, parents=True)
    draw(series, args.output / "courbes.png")
    save_json(args.output / "summary.json", series)


if __name__ == "__main__":
    main()
