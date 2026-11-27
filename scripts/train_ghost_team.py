"""Bounded shared-parameter PPO on the same physical game for all agents."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from main.arena_models import ARENA_MODELS, load_model
from rl.anchored_ppo import AuditedMaskablePPO
from rl.ghost_team_env import REWARD_PROFILES, TEAM_SIZES, GhostTeamEnv, TeamVecEnv
from rl.ghost_team_evaluation import play_team, summarize
from scripts.train_interaction_dqn import save_json
from scripts.train_slow_ghost import policy_digest


BUDGET = 24576
CHECKPOINT = 8192
CONTROL = 2048
SEEDS = (301, 509)
VALIDATION = tuple(range(95600, 95608))
TEST = tuple(range(95800, 95830))
REWARD_TEST = tuple(range(125600, 125630))
SETTINGS = dict(n_steps=512, batch_size=128, n_epochs=3, learning_rate=3e-4, gamma=.995,
                gae_lambda=.95, ent_coef=.02, clip_range=.2, target_kl=.025)


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


class Collection(BaseCallback):
    def __init__(self, count):
        super().__init__()
        self.personal = np.zeros(count, dtype=int)
        self.team_lives = self.wins = self.pacman_wins = self.timeouts = 0
        self.shared = 0.

    def _on_step(self):
        info = self.locals["infos"][0]
        self.personal += info["credit"]
        self.team_lives += len(info["capture_events"])
        self.wins += info["ghost_win"]
        self.pacman_wins += info["victory"]
        self.timeouts += info["timeout"]
        self.shared += info["shared_reward"]
        return True

    def snapshot(self):
        return dict(agent_captures=self.personal.tolist(), team_lives=self.team_lives,
                    ghost_wins=self.wins, pacman_wins=self.pacman_wins, timeouts=self.timeouts,
                    shared_reward=self.shared)


def technical(directory, count, profile):
    opponent = load_model(ARENA_MODELS[0])
    before = policy_digest(opponent)
    match = GhostTeamEnv(opponent, count, reward_profile=profile)
    vec = TeamVecEnv(match, 95500)
    rng = np.random.default_rng(1191)
    clock = perf_counter()
    try:
        observation = vec.reset()
        actions = np.zeros((count, 5), dtype=int)
        credit = np.zeros(count, dtype=int)
        wins = timeouts = captures = 0
        for _ in range(CONTROL):
            assert all(match.observation_space.contains({k:v[i] for k,v in observation.items()})
                       for i in range(count))
            masks = np.stack(vec.env_method("action_masks"))
            assert masks.shape == actions.shape and masks.any(axis=1).all()
            decisions = np.array([rng.choice(np.flatnonzero(mask)) for mask in masks])
            actions[np.arange(count), decisions] += 1
            observation, rewards, dones, infos = vec.step(decisions)
            info = infos[0]
            individual = match.reward["individual_life"]
            weights = [match.reward.get("fast_individual_life", individual) if slot in match.fast_slots else individual
                       for slot in match.slots]
            shared_count = (sum(info["credit"]) if match.reward.get("team_life_controlled_only")
                            else len(info["capture_events"]))
            expected = (np.asarray(info["credit"]) * weights + match.reward["team_life"] * shared_count
                        + match.reward["team_win"] * info["ghost_win"] + match.reward["pacman_win"] * info["victory"])
            np.testing.assert_allclose(rewards, expected, atol=1e-7)
            assert np.isfinite(rewards).all() and np.all(dones == dones[0])
            assert not (info["timeout"] and info["ghost_win"]) and not info["window_closed"]
            credit += info["credit"]
            captures += len(info["capture_events"])
            wins += info["ghost_win"]
            timeouts += info["timeout"]
    finally:
        vec.close()
    assert before == policy_digest(opponent)
    output = dict(count=count, reward_profile=profile, team_steps=CONTROL, agent_steps=CONTROL*count, passed=True,
                  action_counts=actions.tolist(), agent_captures=credit.tolist(), team_lives=captures,
                  ghost_wins=wins, timeouts=timeouts, seconds=perf_counter()-clock)
    save_json(directory/"control.json",output)
    print(output,flush=True)


def rank(row):
    return min(row["per_agent"]) > 0, row["controlled_lives"], row["ghost_win"], -row["timeout"]


def train(directory, count, seed, gate, parent, profile):
    control = json.loads(gate.read_text(encoding="utf-8"))
    assert control["passed"] and (control["count"], control["team_steps"]) == (count, CONTROL)
    assert control.get("reward_profile", "baseline") == profile
    opponent = load_model(ARENA_MODELS[0])
    source = policy_digest(opponent)
    world_seed = 20_000_000 + count * 300_000 + seed*100_000
    match = GhostTeamEnv(opponent, count, reward_profile=profile)
    vec = TeamVecEnv(match, world_seed)
    config = dict(count=count, seed=seed, team_budget=BUDGET, agent_budget=BUDGET*count,
                  world_seed=world_seed, policy_sha256=ARENA_MODELS[0].sha256,
                  held_out_opponent=ARENA_MODELS[1].name, validation_seeds=VALIDATION,
                  reward=REWARD_PROFILES[profile], reward_profile=profile, settings=SETTINGS, network=[128,64],
                  environment=GhostTeamEnv.version, continued_from=str(parent) if parent else None)
    save_json(directory/"config.json",config)
    checkpoints = []
    collection = Collection(count)
    try:
        model = AuditedMaskablePPO("MultiInputPolicy", vec, seed=seed, device="cpu",
                       policy_kwargs=dict(net_arch=dict(pi=[128,64], vf=[128,64])), **SETTINGS)
        if parent:
            prior = MaskablePPO.load(parent,device="cpu")
            model.policy.load_state_dict(prior.policy.state_dict())
            model.set_random_seed(seed)
        initial = policy_digest(model)
        assert model.num_timesteps==0 and not model.policy.optimizer.state_dict()["state"]
        model.save(directory/"initial.zip")
        started = perf_counter()
        for stop in range(CHECKPOINT,BUDGET+1,CHECKPOINT):
            model.learn(CHECKPOINT*count,reset_num_timesteps=False,callback=collection,log_interval=None)
            assert model.num_timesteps == stop*count and all(torch.isfinite(p).all() for p in model.policy.parameters())
            name = f"ppo_{stop:06}.zip"
            model.save(directory/name)
            row = dict(team_steps=stop, path=name, elapsed=perf_counter()-started, **collection.snapshot())
            checkpoints.append(row)
            save_json(directory/f"collection_{stop:06}.json",row)
            print(seed,stop,round(row["elapsed"],1),row["agent_captures"],row["ghost_wins"],flush=True)
        assert source == policy_digest(opponent) and policy_digest(model) != initial
        save_json(directory/"training.json",dict(checkpoints=checkpoints,diagnostics=model.rollout_diagnostics))
    finally:
        vec.close()
    try:
        validation = []
        for entry in checkpoints:
            policy = MaskablePPO.load(directory/entry["path"],device="cpu")
            result = summarize([play_team(match, val_seed, policy) for val_seed in VALIDATION])
            validation.append(dict(team_steps=entry["team_steps"],path=entry["path"],**result))
            print("Validation",seed,entry["team_steps"],result["ghost_win"],result["per_agent"],flush=True)
    finally:
        match.close()
    assert source == policy_digest(opponent)
    save_json(directory/"results.json",dict(**config,initial_sha256=initial,collection=checkpoints,
              validation=validation,selected=max(validation,key=rank),frozen_verified=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--count",type=int,choices=TEAM_SIZES,required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--control",action="store_true")
    modes.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--gate",type=Path)
    parser.add_argument("--parent",type=Path)
    parser.add_argument("--reward",choices=REWARD_PROFILES,default="baseline")
    args = parser.parse_args()
    if args.seed is not None and args.gate is None:
        parser.error("--gate required for training")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    if args.control:
        technical(args.output,args.count,args.reward)
    else:
        train(args.output,args.count,args.seed,args.gate,args.parent,args.reward)


if __name__=="__main__":
    main()
