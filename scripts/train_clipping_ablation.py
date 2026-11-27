"""Restart from the exact warm-critic checkpoints and compare clipping rules."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.split_clip_ppo import WeightAuditPPO
from scripts.diagnose_dagger_robustness import verify_references
from scripts.train_critic_ablation import SEEDS, state_digest
from scripts.train_hybrid_ppo_pilot import BUDGET, digest, evaluation, rank, worker
from scripts.train_interaction_dqn import save_json


PREVIOUS = Path("runs/critic_ablation_20261126")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--clipping", choices=("global", "split"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_references()
    torch.set_num_threads(1)
    source = PREVIOUS / f"warm_t05_{args.seed}"
    previous = json.loads((source / "result.json").read_text())
    manifest = json.loads((PREVIOUS / "benchmark/integrity.json").read_text())
    initial = source / "initial.zip"
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == next(
        row["sha256"] for row in manifest["checkpoints"] if Path(row["path"]) == initial)
    cases = list({r["family"]: r for r in json.loads(
        Path("runs/dagger_targeted_data/holdout.json").read_text())["episodes"]}.values())
    args.output.mkdir(parents=True, exist_ok=False)
    env = SubprocVecEnv([partial(worker, args.seed, str(args.output / "online"))], start_method="spawn")
    try:
        model = WeightAuditPPO.load(initial, env=env, device="cpu")
        model.split_gradients = args.clipping == "split"
        model.set_random_seed(args.seed)
        assert not model.policy.optimizer.state_dict()["state"] and model.num_timesteps == 0
        assert state_digest(model.policy, critic=False) == previous["initial_actor"]
        assert state_digest(model.policy, critic=True) == previous["initial_critic"]
        config = json.loads((source / "config.json").read_text())
        config.update(clipping=args.clipping, initial=str(initial))
        save_json(args.output / "config.json", config)
        rows = []
        elapsed = 0.
        for step in (4096, BUDGET):
            started = perf_counter()
            model.learn(step-model.num_timesteps, reset_num_timesteps=False, log_interval=None)
            elapsed += perf_counter()-started
            assert model.num_timesteps==step
            file = f"ppo_{step:06}.zip"
            model.save(args.output / file)
            row = dict(file=file, steps=step, **evaluation(model, cases))
            rows.append(row)
            save_json(args.output / f"validation_{step:06}.json", row)
            print(args.clipping, args.seed, step, row["full"]["victory"], row["recovery"]["victory"], flush=True)
        first = model.rollout_diagnostics[0]["trajectory_sha256"]
        assert first == previous["rollout_diagnostics"][0]["trajectory_sha256"]
        with np.load(source / "first_rollout.npz") as archived:
            for key in ("advantages", "values", "targets"):
                np.testing.assert_array_equal(model.first_rollout[key], archived[key])
        # Loading SB3 checkpoints can reseed shared generators. Audit only after learning.
        reproduced = {}
        if args.clipping == "global":
            for row in rows:
                file = row["file"]
                reference = MaskablePPO.load(source / file, device="cpu")
                current = MaskablePPO.load(args.output / file, device="cpu")
                reproduced[row["steps"]] = digest(current.policy.state_dict().values()) == digest(
                    reference.policy.state_dict().values())
            if not all(reproduced.values()):
                raise RuntimeError(f"Instrumented global clipping changed weights: {reproduced}.")
        report = dict(**config, selected=max(rows, key=rank), checkpoints=rows, training_seconds=elapsed,
                      actual_steps=model.num_timesteps, historical_reproduction=reproduced,
                      rollout_diagnostics=model.rollout_diagnostics)
        save_json(args.output / "result.json", report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
