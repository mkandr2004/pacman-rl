"""Roll the expert forward from real pre-collision states of the BC learner."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import torch
from sb3_contrib import MaskablePPO

from rl.imitation_eval import play_network
from scripts.generate_expert_data import capture
from scripts.train_dagger import FAILURES, FROZEN, prefixes_from_manifest, verify_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("At least one worker is required.")
    torch.set_num_threads(1)
    verify_reference()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    policy = MaskablePPO.load(FROZEN / "selected.zip", device="cpu")
    source = json.loads(Path("runs/expert_imitation_120/manifest.json").read_text())
    scenarios = {f"prefix_{key}": prefix for key, prefix in prefixes_from_manifest(source).items()}
    jobs = []
    for seed in FAILURES:
        game = play_network(policy, seed, danger=False, record=True)
        if game["victory"] or not game["death"]:
            raise RuntimeError(f"The reported BC failure on seed {seed} did not reproduce.")
        trace = game["trace"]
        actions = [record["action"] for record in trace]
        losses = [i for i, row in enumerate(trace)
                  if row["health"] < (trace[i-1]["health"] if i else 3)]
        if seed == FAILURES[0]:
            first_wall = next(i for i, row in enumerate(trace) if row["walls"])
            scenarios["post_wall"] = actions[:first_wall+6]
            scenarios["before_contact"] = actions[:losses[1]-8]
        (args.output / f"learner_seed_{seed}.json").write_text(json.dumps(game, indent=2))
        for loss_step in losses:
            for lead in (8, 24):
                stop = max(0, loss_step-lead)
                for danger_weight, anticipate in ((1.75, False), (2.5, True)):
                    location = args.output / f"branch_{len(jobs):02}"
                    (location / "recovery").mkdir(parents=True)
                    jobs.append((location, "recovery", seed, actions[:stop], danger_weight, anticipate))
    winning = json.loads(Path("runs/bc_eval_cnn_42_blank/win_seed_54001.json").read_text())
    near_end = next(i for i, row in enumerate(winning["trace"]) if row["score"] >= 124)
    scenarios["last_items"] = [row["action"] for row in winning["trace"][:near_end+1]]
    (args.output / "scenarios.json").write_text(json.dumps(scenarios, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(capture, jobs))
    for job, row in zip(jobs, rows):
        row["file"] = str((job[0] / row["file"]).relative_to(args.output))
    manifest = dict(episodes=rows, wins=sum(row["victory"] for row in rows),
                    seconds=perf_counter()-started)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(len(rows), "branches", manifest["wins"], "wins", flush=True)


if __name__ == "__main__":
    main()
