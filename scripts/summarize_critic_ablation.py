"""Summarize the optimizer traces and audited games of a completed critic ablation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import stat
from statistics import mean

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scripts.critic_rollouts import TEMPERATURES
from scripts.train_critic_ablation import SEEDS
from scripts.train_interaction_dqn import save_json


def draw(output, runs, result):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    effects = result["critic_effects"]
    colors = {"t1": "#28628c", "t05": "#087c65"}
    first = axes[0, 0]
    for label in TEMPERATURES:
        for ix, seed in enumerate(SEEDS):
            effect = effects[f"{label}_{seed}"]["full"]
            point = effect["delta_points"]
            lo, hi = effect["ci95"]
            first.errorbar(point, ix+(-.11 if label=="t1" else .11), xerr=[[point-lo], [hi-point]],
                           fmt="o", color=colors[label], capsize=3,
                           label=f"T = {TEMPERATURES[label]:g}" if ix==0 else None)
    first.axvline(0, color="#575b60", lw=1)
    first.set_yticks(range(len(SEEDS)), labels=SEEDS)
    first.set_title("Effet du critique sur les victoires")
    first.set_xlabel("Écart apparié, points de pourcentage")
    first.set_ylabel("Seed d’entraînement")
    first.legend()
    for ax, key, title, ylabel in (
        (axes[0, 1], lambda r: r["fixed_probe"]["actor_kl"], "Écart à l’acteur DAgger 42", "KL sur observations fixes"),
        (axes[1, 0], lambda r: r["gradient_norm"]["critic"]["mean"], "Gradient du critique avant clipping", "Norme L2"),
        (axes[1, 1], lambda r: r["fixed_probe"]["value_mean"], "Valeur sur les mêmes observations", "Retour prédit"),
    ):
        for label in TEMPERATURES:
            for critic in ("random", "warm"):
                paths = [[key(row) for row in runs[f"{critic}_{label}_{seed}"]["rollout_diagnostics"]]
                         for seed in SEEDS]
                steps = [row["steps"] for row in runs[f"{critic}_{label}_{SEEDS[0]}"]["rollout_diagnostics"]]
                for path in paths:
                    ax.plot(steps, path, color=colors[label], alpha=.16)
                ax.plot(steps, [mean(y) for y in zip(*paths)], color=colors[label],
                        linestyle="--" if critic=="random" else "-",
                        label=f"T {TEMPERATURES[label]:g}, {'neuf' if critic=='random' else 'ajusté'}")
        ax.set_title(title)
        ax.set_xlabel("Décisions PPO")
        ax.set_ylabel(ylabel)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_yscale("log")
    fig.tight_layout(pad=2)
    fig.savefig(output, dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    root = args.runs
    benchmark = root / "benchmark"
    result = json.loads((benchmark / "results.json").read_text())
    protocol = json.loads((benchmark / "protocol.json").read_text())
    names = set(result["models"]) - {"dagger42"}
    runs = {name: json.loads((root / name / "result.json").read_text()) for name in sorted(names)}

    expected = {"full": {(seed, "full") for seed in protocol["full_seeds"]},
                "recovery": {(seed, family) for family, seed in protocol["recovery"]}}
    for label in TEMPERATURES:
        expected[f"sample_{label}"] = {(seed, "full") for seed in protocol["sampling_seeds"]}
    remaining = {(model, mode, seed, family) for model in result["models"]
                 for mode, cases in expected.items() for seed, family in cases}
    by_family = {name: Counter() for name in result["models"]}
    with (benchmark / "games.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            remaining.remove((row["model"], row["mode"], row["seed"], row["family"]))
            if row["mode"] == "recovery" and row["victory"]:
                by_family[row["model"]][row["family"]] += 1
            if row["mode"] == "full" and row["victory"]:
                assert (row["points"], row["coins"], row["score"]) == (81, 11, 136)
    assert not remaining
    for entry in protocol["models"].values():
        assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    archive = []
    for name in runs:
        for path in sorted((root / name).glob("*.zip")):
            path.chmod(stat.S_IREAD)
            archive.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(archive) == 3 * len(runs)
    integrity = dict(verified_games=result["games"], recovery_wins_by_family=by_family,
                     checkpoints=archive, reused_checkpoints={
                         name: row["historical_equivalent_checkpoint"] for name, row in runs.items()
                         if row["historical_equivalent_checkpoint"]})
    audit_path = benchmark / "integrity.json"
    if audit_path.exists():
        assert json.loads(audit_path.read_text()) == integrity
    else:
        save_json(audit_path, integrity)
    draw(benchmark / "critic_comparison.png", runs, result)
    print("Verified", result["games"], "games and protected", len(archive), "checkpoints", flush=True)


if __name__ == "__main__":
    main()
