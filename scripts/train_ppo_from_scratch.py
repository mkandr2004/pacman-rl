"""Run the from-scratch MaskablePPO control or a staged, fixed-horizon campaign."""

import argparse
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.anchored_ppo import AuditedMaskablePPO
from rl.imitation_eval import play_network
from rl.imitation_policy import ImitationCNN
from rl.pacman_env import PacmanEnv
from scripts.train_hybrid_ppo_pilot import NormalStarts, digest, outcomes
from scripts.train_interaction_dqn import save_json


HORIZON = 524288
WORKERS = 8
ROLLOUT = 4096
FIRST_LR, LAST_LR = 3e-4, 5e-5
SETTINGS = dict(n_steps=512, n_epochs=4, batch_size=256, gamma=.995, gae_lambda=.95,
                clip_range=.2, target_kl=.02, ent_coef=.01, vf_coef=.5, max_grad_norm=.5)
VALIDATION = tuple(range(93600, 93612))
SEEDS = (811, 953, 1091)


def learning_rate(progress):
    return LAST_LR + (FIRST_LR - LAST_LR) * progress


class FixedHorizonPPO(AuditedMaskablePPO):
    def _update_current_progress_remaining(self, num_timesteps, total_timesteps):
        super()._update_current_progress_remaining(num_timesteps, HORIZON)

    def train(self):
        n = self.action_space.n
        actions = self.rollout_buffer.actions.astype(np.int64).reshape(-1)
        masks = self.rollout_buffer.action_masks.reshape(-1, n)
        if not masks[np.arange(len(actions)), actions].all():
            raise RuntimeError("Illegal action found in the training rollout.")
        super().train()
        self.rollout_diagnostics[-1]["learning_rate"] = self.lr_schedule(self._current_progress_remaining)


def worker(seed, lane, start, directory):
    torch.set_num_threads(1)
    env = NormalStarts(seed)
    # A worker has fewer than HORIZON / WORKERS possible resets in a complete campaign.
    env.next_seed = 20_000_000 + seed * 1_000_000 + lane * 100_000 + start // WORKERS
    assert env.unwrapped.ghost_count == 7 and env.unwrapped.REWARD_CONFIG == PacmanEnv.REWARD_CONFIG
    return Monitor(env, filename=str(directory / f"lane_{lane}"),
                   info_keywords=("victory", "game_over", "score", "health"))


def evaluate(model, cases):
    with torch.random.fork_rng(devices=[]):
        normal = [play_network(model, seed, danger=False) for seed in VALIDATION]
        recovery = [play_network(model, row["seed"], danger=False, prefix=row["prefix"]) for row in cases]
    return dict(full=outcomes(normal), recovery=outcomes(recovery))


def rank_scratch(row):
    full = row["full"]
    return full["victory"], row["recovery"]["victory"], full["reward"], full["score"], -full["max_stagnation"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", choices=(*SEEDS, 673), type=int, required=True)
    parser.add_argument("--stop-at", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.stop_at % ROLLOUT or not ROLLOUT <= args.stop_at <= HORIZON:
        parser.error("--stop-at must align to a complete rollout within the campaign horizon")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    start = 0
    if args.resume:
        model = FixedHorizonPPO.load(args.resume, device="cpu")
        start = model.num_timesteps
        if model.seed != args.seed or args.stop_at <= start:
            raise ValueError("Resume requires the same seed and a larger stopping milestone.")
    cases = list({row["family"]: row for row in json.loads(
        Path("runs/dagger_targeted_data/holdout.json").read_text())["episodes"]}.values())
    config = dict(seed=args.seed, start=start, stop_at=args.stop_at, horizon=HORIZON, workers=WORKERS,
                  temperature=1., danger=False, encoder="cnn", threads=args.threads,
                  settings=SETTINGS, lr_start=FIRST_LR, lr_end=LAST_LR, from_checkpoint=str(args.resume) if args.resume else None,
                  rewards=PacmanEnv.REWARD_CONFIG, validation_seeds=VALIDATION,
                  versions={name: version(name) for name in ("torch","stable-baselines3","sb3-contrib","gymnasium")})
    save_json(args.output / "config.json", config)
    env = SubprocVecEnv([partial(worker, args.seed, lane, start, args.output) for lane in range(WORKERS)],
                      start_method="spawn")
    try:
        if args.resume:
            model.set_env(env)
            # Resume with a reserved sampling stream and fresh simulator episodes.
            model.set_random_seed(args.seed+start)
            model.seed = args.seed
        else:
            model = FixedHorizonPPO(
                "MultiInputPolicy", env, seed=args.seed, device="cpu", learning_rate=learning_rate,
                policy_kwargs=dict(net_arch=dict(pi=[256,128], vf=[256,128]),
                                   activation_fn=torch.nn.ReLU, features_extractor_class=ImitationCNN,
                                   share_features_extractor=False), **SETTINGS)
            assert model.num_timesteps == 0 and not model.policy.optimizer.state_dict()["state"]
        initial_hash = digest(model.policy.state_dict().values())
        initial = f"ppo_{start:06}.zip"
        model.save(args.output / initial)
        checkpoints = [dict(file=initial, steps=start, **evaluate(model, cases))]
        elapsed = 0.
        milestones = [step for step in (8192, 16384, *range(32768, HORIZON, 32768))
                      if start < step < args.stop_at]
        for step in (*milestones, args.stop_at):
            clock = perf_counter()
            model.learn(step-model.num_timesteps, reset_num_timesteps=False, log_interval=None)
            elapsed += perf_counter()-clock
            assert model.num_timesteps==step
            path = f"ppo_{step:06}.zip"
            model.save(args.output / path)
            row = dict(file=path, steps=step, training_seconds=elapsed, **evaluate(model, cases))
            checkpoints.append(row)
            save_json(args.output / f"validation_{step:06}.json", row)
            print(args.seed, step, round(elapsed, 2), row["full"]["victory"], round(row["full"]["score"], 2),
                  row["recovery"]["victory"], flush=True)
        final_hash = digest(model.policy.state_dict().values())
        assert initial_hash != final_hash and all(torch.isfinite(p).all() for p in model.policy.parameters())
        report = dict(**config, actual_steps=model.num_timesteps, training_seconds=elapsed,
                      decisions_per_second=(model.num_timesteps-start)/elapsed, initial_hash=initial_hash,
                      final_hash=final_hash, selected=max(checkpoints[1:], key=rank_scratch), checkpoints=checkpoints,
                      rollout_diagnostics=model.rollout_diagnostics)
        save_json(args.output / "result.json", report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
