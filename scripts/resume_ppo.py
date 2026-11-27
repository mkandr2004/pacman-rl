"""Resume existing PPO weights, back them up, then safely update the same path."""
import argparse
import json
import os
import shutil
from datetime import datetime
from functools import partial
from hashlib import sha256
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from scripts.train_ppo import make_env
from scripts.audit_policy import audit
from rl.pacman_env import PacmanEnv
from rl.shaping import ProgressPotential


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--timesteps", type=int, default=65536)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--potential-shaping", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = Path(args.model).resolve(strict=True)
    root = Path("runs") / datetime.now().strftime("resume_%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    backup = root / "original_model.zip"
    shutil.copy2(source, backup)
    original_hash = digest(source)
    if digest(backup) != original_hash:
        raise RuntimeError("Backup verification failed")
    config = dict(source=str(source), original_sha256=original_hash,
        backup=str(backup.resolve()), additional_timesteps=args.timesteps,
        architecture="unchanged", observations="bonus-state-v3 (unchanged)",
        rules="safe-borders-v4", potential_shaping=args.potential_shaping,
        shaping_gamma=0.995, validation_seed_start=8000,
        validation_reward="base reward, no shaping",
        potential_config=dict(collection_scale=ProgressPotential.COLLECTION_SCALE,
            navigation_scale=ProgressPotential.NAVIGATION_SCALE,
            terminal_potential=0, truncate_as_continuation=True) if args.potential_shaping else None,
        comparison_seeds=[1000, 7000, 7001, 7002, 7003],
        rewards=dict(PacmanEnv.REWARD_CONFIG))
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"RUN={root}\nBACKUP_VERIFIED={backup}", flush=True)
    train_env = SubprocVecEnv([
        partial(make_env, str(root / f"worker_{index}"), potential_shaping=args.potential_shaping)
        for index in range(args.workers)], start_method="spawn")
    # Select checkpoints using task reward, never the added learning signal.
    validation = make_env(potential_shaping=False)
    model = MaskablePPO.load(source, env=train_env, device="cpu")
    if args.potential_shaping and model.gamma != 0.995:
        train_env.close()
        validation.close()
        raise ValueError("Shaping gamma must match the checkpoint gamma")
    model.learning_rate = 1e-4
    model.lr_schedule = lambda _: 1e-4
    model.ent_coef = 0.03
    validation.reset(seed=8000)
    baseline_reward, _ = evaluate_policy(model, validation,
        n_eval_episodes=5, deterministic=True)
    shutil.copy2(backup, root / "best_model.zip")
    callback = MaskableEvalCallback(validation, n_eval_episodes=5,
        eval_freq=max(1, 16384 // args.workers), log_path=str(root),
        best_model_save_path=str(root), deterministic=True)
    callback.best_mean_reward = baseline_reward
    start_steps = model.num_timesteps
    print(f"RESUMING_FROM={start_steps}; BASELINE_VALIDATION_REWARD={baseline_reward}", flush=True)
    try:
        model.learn(args.timesteps, reset_num_timesteps=False,
            callback=callback, log_interval=8)
        model.save(root / "final_model")
    finally:
        train_env.close()
        validation.close()
    candidate = root / "best_model.zip"
    results = audit(candidate, config["comparison_seeds"], root / "candidate_audit")
    safe = all(r["outside_frames"] == 0 and r["borders_broken"] == 0 for r in results)
    report = dict(config=config, start_steps=start_steps,
        actual_additional_steps=model.num_timesteps - start_steps,
        baseline_validation_reward=baseline_reward,
        selected_validation_reward=float(callback.best_mean_reward),
        candidate_sha256=digest(candidate), results=results, safe=safe)
    if not safe:
        raise RuntimeError("Safety evaluation failed; original model left unchanged")
    # Validate the saved archive before replacing the authorized target path.
    MaskablePPO.load(candidate, device="cpu")
    if digest(source) != original_hash:
        raise RuntimeError("Source changed during training; refusing to overwrite it")
    staging = source.with_name(f"{source.stem}.pending_{root.name}.zip")
    shutil.copy2(candidate, staging)
    os.replace(staging, source)
    report["updated_model"] = str(source)
    report["weights_changed"] = digest(source) != original_hash
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
