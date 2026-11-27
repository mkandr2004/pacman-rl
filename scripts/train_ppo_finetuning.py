"""Compare small PPO updates on the frozen DAgger policy and new policies."""

import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.anchored_ppo import AuditedMaskablePPO, masked_kl
from rl.imitation_env import make_imitation_env
from rl.imitation_eval import play_network, summary
from rl.imitation_policy import ImitationCNN
from rl.pacman_env import PacmanEnv
from scripts.pretrain_dagger_critic import FROZEN, verify_frozen


FULL_VALIDATION = tuple(range(65200,65220))
FULL_TEST = tuple(range(67000,67200))


class RecoveryStarts(gym.Wrapper):
    """Use original starts for 75% of episodes and certified resets otherwise."""

    def __init__(self, pool, run_seed):
        super().__init__(make_imitation_env(danger=False))
        self.pool = pool
        self.run_seed = run_seed
        self.episode = 0
        self.reset_rng = np.random.default_rng(run_seed)

    def reset(self, *, seed=None, options=None):
        recovery = self.episode % 4 == 3
        if recovery:
            state = self.pool[int(self.reset_rng.integers(len(self.pool)))]
            self.unwrapped.prefix = tuple(state["prefix"])
            game_seed = state["seed"]
            family = state["family"]
        else:
            self.unwrapped.prefix = ()
            game_seed = 70000 + self.run_seed * 10000 + self.episode
            family = "full"
        self.episode += 1
        obs, info = self.env.reset(seed=game_seed, options=options)
        self.family = family
        return obs, info

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        return obs, reward, terminal, truncated, {**info, "start_family": self.family}

    def action_masks(self):
        return self.unwrapped.action_masks()


def worker(pool, run_seed, path):
    torch.set_num_threads(1)
    return Monitor(RecoveryStarts(pool, run_seed), filename=path,
                   info_keywords=("score", "victory", "game_over", "start_family"))


def evaluate(model, recovery):
    full = [play_network(model, seed, danger=False) for seed in FULL_VALIDATION]
    resumes = [dict(family=case["family"], **play_network(
        model, case["seed"], danger=False, prefix=case["prefix"])) for case in recovery]
    return dict(full=summary(full), games=full, recovery=resumes,
                recovery_result=summary(resumes))


def rank(outcome):
    return (outcome["full"]["victory"], outcome["recovery_result"]["victory"],
            -outcome["full"]["max_stagnation"], outcome["full"]["score"])


def anchor_distance(model, reference, critic_directory):
    with np.load(critic_directory / "seed_65100.npz", allow_pickle=False) as data:
        obs = {key: torch.from_numpy(data[key]) for key in model.observation_space.spaces}
        mask = data["masks"]
    with torch.no_grad():
        prior = reference.policy.get_distribution(obs, action_masks=mask).distribution
        current = model.policy.get_distribution(obs, action_masks=mask).distribution
        return dict(kl=masked_kl(prior, current).mean().item(),
                    agreement=(prior.probs.argmax(-1) == current.probs.argmax(-1)).float().mean().item(),
                    states=len(mask))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("standard", "anchored", "scratch"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=32768)
    parser.add_argument("--recovery", type=Path, default=Path("runs/ppo_recovery_verified/qualified.json"))
    parser.add_argument("--critic", type=Path, default=Path("runs/ppo_dagger_critic"))
    args = parser.parse_args()
    if args.seed < 0 or args.budget <= 0 or args.budget % 8192:
        parser.error("Seed must be nonnegative and budget a positive multiple of 8192.")
    verify_frozen()
    torch.set_num_threads(1)
    rows = json.loads(args.recovery.read_text())
    training = [row for row in rows if row["qualified"] and row["split"] == "train"]
    eligible = [row for row in rows if row["qualified"] and row["split"] == "validation"]
    validation = []
    by_family = Counter()
    for row in eligible:
        if by_family[row["family"]] < 2:
            validation.append(row)
            by_family[row["family"]] += 1
    if not training or not validation:
        raise RuntimeError("Verified training and validation resets are both required.")
    if set(FULL_TEST).intersection(FULL_VALIDATION + tuple(row["seed"] for row in rows)):
        raise RuntimeError("Data split overlap.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    scratch = args.arm == "scratch"
    coefficient = 1.0 if args.arm == "anchored" else 0.
    settings = dict(n_steps=1024, n_epochs=3, batch_size=256, gamma=.995, gae_lambda=.95,
                    learning_rate=3e-4 if scratch else 1.5e-5,
                    clip_range=.20 if scratch else .08, target_kl=.02 if scratch else .006,
                    ent_coef=.001, vf_coef=.25, max_grad_norm=.5)
    config = dict(arm=args.arm, encoder="cnn", danger=False, seed=args.seed,
                  budget=args.budget, anchor_coefficient=coefficient, hyperparameters=settings,
                  full_validation=FULL_VALIDATION, reserved_full_test=FULL_TEST,
                  verified_training_starts=len(training),
                  validation_cases=[(r["family"],r["seed"]) for r in validation],
                  rewards=PacmanEnv.REWARD_CONFIG,
                  start_mix="three normal starts then one certified recovery reset")
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    reference = MaskablePPO.load(FROZEN, device="cpu")
    for parameter in reference.policy.parameters():
        parameter.requires_grad_(False)
    env = SubprocVecEnv([partial(worker, training, args.seed, str(args.output / "training"))],
                        start_method="spawn")
    try:
        model = AuditedMaskablePPO(
            "MultiInputPolicy", env, seed=args.seed, device="cpu",
            anchor_coefficient=coefficient, anchor_policy=reference.policy if coefficient else None,
            policy_kwargs=dict(net_arch=dict(pi=[256,128], vf=[256,128]),
                               share_features_extractor=False, activation_fn=torch.nn.ReLU,
                               features_extractor_class=ImitationCNN), **settings)
        if not scratch:
            warm_report = json.loads((args.critic / "report.json").read_text())
            if not warm_report["actor_bitwise_equal"] or not warm_report["ppo_optimizer_still_empty"]:
                raise RuntimeError("Critic checkpoint did not pass its integrity audit.")
            warm = MaskablePPO.load(args.critic / "critic_warm.zip", device="cpu")
            model.policy.load_state_dict(warm.policy.state_dict())
        assert not model.policy.optimizer.state_dict()["state"]
        initial = anchor_distance(model, reference, args.critic)
        if not scratch and (abs(initial["kl"]) > 1e-7 or initial["agreement"] != 1.):
            raise RuntimeError("The warm start altered the actor.")
        model.save(args.output / "initial.zip")
        selected = None
        history = []
        for goal in range(4096, args.budget + 1, 4096):
            model.learn(4096, reset_num_timesteps=False, log_interval=None)
            checkpoint = f"ppo_{goal:06}.zip"
            model.save(args.output / checkpoint)
            distance = anchor_distance(model, reference, args.critic)
            row = dict(steps=model.num_timesteps, checkpoint=checkpoint, anchor=distance,
                       mean_update_kl=mean(r["approx_kl"] for r in model.rollout_diagnostics[-4:]))
            if goal % 8192 == 0 or (not scratch and distance["kl"] > .08):
                row["validation"] = evaluate(model, validation)
                if selected is None or rank(row["validation"]) > rank(selected["validation"]):
                    selected = row
            history.append(row)
            (args.output / "curve.json").write_text(json.dumps(history, indent=2))
            (args.output / "ppo_diagnostics.json").write_text(json.dumps(model.rollout_diagnostics, indent=2))
            full = row.get("validation", {}).get("full", {})
            hard = row.get("validation", {}).get("recovery_result", {})
            print(args.arm, args.seed, goal, "anchor_kl", round(distance["kl"], 5),
                  "validation", full.get("victory"), "recovery", hard.get("victory"), flush=True)
            if not scratch and distance["kl"] > .08:
                break
        verify_frozen()
        (args.output / "result.json").write_text(json.dumps(
            dict(**config, selected=selected["checkpoint"], selected_step=selected["steps"],
                 actual_steps=model.num_timesteps, initial_anchor=initial,
                 selected_validation=selected["validation"], seconds=perf_counter()-started), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
