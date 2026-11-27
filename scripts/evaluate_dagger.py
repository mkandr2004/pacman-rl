"""Locked DAgger comparison. This module neither imports nor runs the expert."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rl.imitation_eval import play_network, summary


_MODELS = {}


def trial(job):
    name, checkpoint, seed, scenario, prefix = job
    torch.set_num_threads(1)
    if checkpoint not in _MODELS:
        _MODELS[checkpoint] = MaskablePPO.load(checkpoint, device="cpu")
    try:
        row = play_network(_MODELS[checkpoint], seed, danger=False, prefix=prefix)
    except RuntimeError as error:
        if str(error) != "Scenario prefix terminated; invalid scenario":
            raise
        row = dict(seed=seed, setup_terminated=True)
    return name, scenario, row


def load_candidate(directory):
    report = json.loads((directory / "training_report.json").read_text())
    curve = json.loads((directory / "curve.json").read_text())
    assert report["critic_unchanged"] and report["ppo_num_timesteps"] == 0
    selected = directory / report["selected"]
    collected = {row["seed"] for generation in curve for row in generation["collection"]}
    return selected, report, collected


def metrics_after_wall(games):
    broken = [row for row in games if row["first_wall_step"] is not None]
    eligible = [row for row in broken if row["steps"] - row["first_wall_step"] >= 30
                or row["victory"] or row["death"]]
    return dict(episodes_with_wall=len(broken), eligible=len(eligible),
                no_life_lost_30=sum(row["first_life_loss_after_wall"] is None
                                    or row["first_life_loss_after_wall"] > 30 for row in eligible),
                reached_last_12=sum(row["reached_last_items"] for row in games))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("At least one worker is required.")
    torch.set_num_threads(1)
    frozen = Path("runs/frozen_bc_86_7/selected.zip")
    saved = json.loads(frozen.with_name("integrity.json").read_text())
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == saved["sha256"]
    models = {"bc42_frozen": frozen, "bc137": Path("runs/bc_cnn_137_blank/selected.zip")}
    used = {54000, 54005, 54012, 54026}
    seeds = tuple(range(63000,63100))
    recovery_seeds = tuple(range(63100,63106))
    for directory in args.candidates:
        checkpoint, config, collected = load_candidate(directory)
        assert tuple(config["final_test_seeds"]) == seeds
        assert tuple(config["final_test_recovery"]) == recovery_seeds
        used.update(collected)
        used.update(config["validation_seeds"])
        used.update(config["validation_recovery"])
        dataset = Path(config.get("original_data", "runs/expert_imitation_120"))
        archived = json.loads((dataset / "manifest.json").read_text())
        used.update(row["seed"] for row in archived["episodes"])
        if config.get("recovery_data"):
            branches = json.loads((Path(config["recovery_data"]) / "manifest.json").read_text())
            used.update(row["seed"] for row in branches["episodes"])
        models[directory.name] = checkpoint
    if used.intersection(seeds + recovery_seeds):
        raise RuntimeError("Final test seeds overlap the archived training or validation seeds.")
    assert len(models) == len(args.candidates)+2
    scenarios = json.loads(args.scenarios.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(full_test_seeds=seeds, recovery_seeds=recovery_seeds, excluded=sorted(used),
                    map_sha256=hashlib.sha256(Path("maps/map_2.txt").read_bytes()).hexdigest(),
                    models={name: dict(path=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                            for name,file in models.items()})
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2))
    jobs = [(name, str(file), seed, "full", []) for seed in seeds for name,file in models.items()]
    jobs += [(name,str(file),seed,scenario,prefix)
             for scenario,prefix in scenarios.items() for seed in recovery_seeds for name,file in models.items()]
    results = {name: {"full": [], **{key: [] for key in scenarios}} for name in models}
    started = perf_counter()
    with (ProcessPoolExecutor(max_workers=args.workers) as pool,
          (args.output / "games.jsonl").open("x", encoding="utf-8") as log):
        for index, (name, scenario, row) in enumerate(pool.map(trial,jobs,chunksize=4),start=1):
            results[name][scenario].append(row)
            log.write(json.dumps(dict(model=name, scenario=scenario, **row)) + "\n")
            if index % 75 == 0:
                log.flush()
                print(index, "/", len(jobs), "paired games evaluated", flush=True)
    baseline = [int(row["victory"]) for row in results["bc42_frozen"]["full"]]
    rng = np.random.default_rng(20261124)
    indices = rng.integers(0,len(baseline),size=(10000,len(baseline)))
    output = {}
    for name, values in results.items():
        games = values["full"]
        assert [row["seed"] for row in games] == list(seeds)
        assert all((row["points"],row["coins"],row["score"]) == (81,11,136)
                   for row in games if row["victory"])
        diff = np.array([int(row["victory"]) for row in games])-baseline
        by_scenario = {}
        for scenario in scenarios:
            valid = [row for row in values[scenario] if not row.get("setup_terminated")]
            by_scenario[scenario] = dict(valid=len(valid), attempts=len(recovery_seeds),
                                         wins=sum(row["victory"] for row in valid),
                                         escapes=sum(row["escape_step"] is not None for row in valid))
        output[name] = dict(summary=summary(games), wall_metrics=metrics_after_wall(games),
                            recovery=by_scenario, gained=int((diff>0).sum()), lost=int((diff<0).sum()),
                            paired_interval_95_percent_points=(np.quantile(diff[indices].mean(axis=1),
                                                                         [.025,.975])*100).tolist(),
                            games=games, recovery_games={key:values[key] for key in scenarios})
        print(name, output[name]["summary"]["victory"], "wins;", output[name]["recovery"], flush=True)
    (args.output / "results.json").write_text(json.dumps(
        dict(elapsed_seconds=perf_counter()-started, **manifest, results=output),indent=2))


if __name__=="__main__":
    main()
