"""Audit the common benchmark and plot learning and clipping measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import stat
from statistics import mean

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scripts.train_interaction_dqn import save_json


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def plot(root, benchmark, models):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    normal, recovery, curves, movement = axes.flatten()
    groups = ("scratch", "dagger42", "hybrid", "warm", "split")
    labels = ("PPO seul", "DAgger 42", "IL + PPO", "Critique préentraîné", "Clipping séparé")
    palette = ("#656d75", "#2b3545", "#2b7094", "#168471", "#b46a2d")
    for axis, test, title in ((normal, "normal", "Victoires normales"),
                              (recovery, "recovery", "Victoires en reprise")):
        for i, (group, color) in enumerate(zip(groups,palette)):
            values = [row[test]["victory"]/row[test]["episodes"]*100 for name, row in models.items()
                      if name.split("_")[0] == group]
            x = [i] if len(values) == 1 else [i-.13, i, i+.13]
            axis.scatter(x, values, color=color)
        axis.set_xticks(range(len(groups)),labels,rotation=18,ha="right")
        axis.set_ylabel("Parties gagnées (%)")
        axis.set_ylim(-3, 103)
        axis.set_title(title)
    for path in sorted(root.glob("scratch_*_stage*/result.json")):
        run = read(path)
        episodes = []
        for monitor in sorted(path.parent.glob("lane_*.monitor.csv")):
            with monitor.open(encoding="utf8") as stream:
                start = json.loads(stream.readline().removeprefix("#"))["t_start"]
                episodes.extend((start+float(row["t"]), int(row["score"])) for row in csv.DictReader(stream))
        scores = [score for _, score in sorted(episodes)]
        blocks = [scores[len(scores)*i//4:len(scores)*(i+1)//4] for i in range(4)]
        curves.plot(range(1,5), [mean(block) for block in blocks], marker="o", label=str(run["seed"]))
    curves.axhline(0, color="#58606a", linewidth=1, linestyle=":", label="Validation")
    curves.set_title("PPO seul pendant la collecte")
    curves.set_xlabel("Quartile chronologique des parties terminées")
    curves.set_xticks(range(1,5))
    curves.set_ylabel("Score moyen")
    curves.legend(title="Seed")
    for mode, color in (("global", "#168471"), ("split", "#b46a2d")):
        for i, path in enumerate(sorted(root.glob(f"{mode}_*/result.json"))):
            rows = read(path)["rollout_diagnostics"]
            movement.plot([row["steps"] for row in rows],
                          [row["actor_updates"]["total_delta_l2"] for row in rows],
                          color=color, alpha=.72, label=mode if i == 0 else None)
    movement.set_title("Déplacement des poids de l’acteur")
    movement.set_xlabel("Décisions PPO")
    movement.set_ylabel("Distance L2 à l’initialisation")
    movement.legend()
    fig.tight_layout(pad=2)
    fig.savefig(benchmark / "comparaison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--benchmark",type=Path,required=True)
    args = parser.parse_args()
    protocol = read(args.benchmark / "protocol.json")
    report = read(args.benchmark / "results.json")
    expected = {"normal": set(protocol["full_seeds"]), "t1": set(protocol["stochastic_seeds"]),
                "t05": set(protocol["stochastic_seeds"]), "recovery": {s for _, s in protocol["cases"]}}
    remaining = {(name, mode, seed) for name in protocol["models"]
                 for mode, seeds in expected.items() for seed in seeds}
    with (args.benchmark / "games.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            remaining.remove((row["model"],row["mode"],row["seed"]))
    assert not remaining
    for item in protocol["models"].values():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    archive = []
    for path in sorted(args.runs.rglob("*.zip")):
        path.chmod(stat.S_IREAD)
        archive.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    integrity = dict(games=report["games"], checkpoints=archive)
    destination = args.benchmark / "integrity.json"
    if destination.exists():
        assert read(destination) == integrity
    else:
        save_json(destination, integrity)
    plot(args.runs,args.benchmark,report["models"])
    print(f"Verified {report['games']} games and protected {len(archive)} checkpoints.",flush=True)


if __name__ == "__main__":
    main()
