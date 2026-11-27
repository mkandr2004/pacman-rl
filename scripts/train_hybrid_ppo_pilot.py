"""Controlled pilot comparing scratch PPO and two DAgger 42 warm starts."""

import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import gymnasium as gym
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.anchored_ppo import AuditedMaskablePPO
from rl.imitation_env import make_imitation_env
from rl.imitation_eval import play_network, summary
from rl.imitation_policy import ImitationCNN
from rl.pacman_env import PacmanEnv
from rl.tempered_ppo import TemperedPolicy
from scripts.diagnose_dagger_robustness import REFERENCE, verify_references
from scripts.pretrain_dagger_critic import critic_name
from scripts.train_interaction_dqn import save_json


ARMS = {"scratch": 1., "hybrid": 1., "tempered": .5}
SEEDS = (337, 463)
BUDGET = 8192
VALIDATION = tuple(range(88600, 88612))
FINAL = tuple(range(89000, 89100))
PPO_SETTINGS = dict(n_steps=1024, n_epochs=3, batch_size=256, gamma=.995, gae_lambda=.95,
                    learning_rate=3e-5, clip_range=.08, target_kl=.006, ent_coef=.001,
                    vf_coef=.25, max_grad_norm=.5)


def make_ppo(env, seed, temperature, algorithm=AuditedMaskablePPO):
    return algorithm(TemperedPolicy, env, seed=seed, device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]),
                           activation_fn=torch.nn.ReLU, features_extractor_class=ImitationCNN,
                           share_features_extractor=False, temperature=temperature), **PPO_SETTINGS)


class NormalStarts(gym.Wrapper):
    def __init__(self, run_seed):
        super().__init__(make_imitation_env(danger=False))
        self.next_seed = 1_800_000 + run_seed * 1000

    def reset(self, *, seed=None, options=None):
        value = self.next_seed
        self.next_seed += 1
        return self.env.reset(seed=value, options=options)

    def action_masks(self):
        return self.unwrapped.action_masks()


def worker(seed, path):
    torch.set_num_threads(1)
    return Monitor(NormalStarts(seed), filename=path,
                   info_keywords=("victory", "game_over", "score", "health"))


def outcomes(games):
    return dict(**summary(games), reward=mean(g["reward"] for g in games))


def evaluation(model, cases):
    state = torch.get_rng_state()
    # play_network sets PyTorch's seed, which must not alter later PPO rollouts.
    with torch.random.fork_rng(devices=[]):
        normal = [play_network(model, seed, danger=False) for seed in VALIDATION]
        recovery = [play_network(model, case["seed"], danger=False, prefix=case["prefix"]) for case in cases]
    assert torch.equal(state, torch.get_rng_state())
    return dict(full=outcomes(normal), recovery=outcomes(recovery))


def digest(tensors):
    return hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in tensors)).hexdigest()


def rank(row):
    return (row["full"]["victory"], row["recovery"]["victory"], -row["full"]["max_stagnation"],
            row["full"]["reward"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_references()
    torch.set_num_threads(1)
    temperature = ARMS[args.arm]
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(arm=args.arm, seed=args.seed, danger=False, encoder="cnn", temperature=temperature,
                  budget=BUDGET, settings=PPO_SETTINGS, validation_normal=VALIDATION,
                  rewards=PacmanEnv.REWARD_CONFIG,
                  imitation_source=None if args.arm == "scratch" else str(REFERENCE["dagger42"]))
    save_json(args.output / "config.json", config)
    archived = json.loads(Path("runs/dagger_targeted_data/holdout.json").read_text())["episodes"]
    by_family = {row["family"]:row for row in archived}
    cases = list(by_family.values())
    assert len(cases) >= 5 and not (set(VALIDATION) | set(FINAL)).intersection(row["seed"] for row in cases)
    env = SubprocVecEnv([partial(worker, args.seed, str(args.output / "online"))], start_method="spawn")
    started = perf_counter()
    try:
        model = make_ppo(env, args.seed, temperature)
        state = model.policy.state_dict()
        if args.arm != "scratch":
            source = MaskablePPO.load(REFERENCE["dagger42"], device="cpu").policy.state_dict()
            state = {key: value if critic_name(key) else source[key] for key,value in state.items()}
            model.policy.load_state_dict(state)
        model.set_random_seed(args.seed)
        assert not model.policy.optimizer.state_dict()["state"]
        hashes = dict(initial_actor=digest(value for key,value in state.items() if not critic_name(key)),
                      initial_critic=digest(value for key,value in state.items() if critic_name(key)))
        model.save(args.output / "initial.zip")
        checkpoints = []
        training_seconds = []
        for step in (4096,8192):
            phase = perf_counter()
            model.learn(4096, reset_num_timesteps=False, log_interval=None)
            training_seconds.append(perf_counter()-phase)
            filename = f"ppo_{step:06}.zip"
            model.save(args.output / filename)
            row = dict(file=filename, steps=step, **evaluation(model, cases))
            checkpoints.append(row)
            save_json(args.output / f"validation_{step:06}.json", row)
            print(args.arm,args.seed,step,row["full"]["victory"],row["recovery"]["victory"],flush=True)
        with (args.output / "online.monitor.csv").open(encoding="utf-8") as stream:
            next(stream)
            logs = list(csv.DictReader(stream))
        best = max(checkpoints,key=rank)
        result = dict(**config, **hashes, selected=best, checkpoints=checkpoints,
                      actual_steps=model.num_timesteps, training_seconds=sum(training_seconds),
                      elapsed_seconds=perf_counter()-started, collected_episodes=len(logs),
                      collected_wins=sum(row["victory"]=="True" for row in logs),
                      collected_reward=mean(float(row["r"]) for row in logs),
                      rollout_diagnostics=model.rollout_diagnostics)
        save_json(args.output / "result.json", result)
    finally:
        env.close()


if __name__ == "__main__":
    main()
