"""Compare initial critic weights under paired, otherwise identical PPO runs."""

import argparse
import csv
from functools import partial
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.nn.functional import smooth_l1_loss

from rl.critic_ppo_audit import CriticAuditPPO
from rl.pacman_env import PacmanEnv
from scripts.critic_rollouts import SPLITS, TEMPERATURES
from scripts.diagnose_dagger_robustness import REFERENCE, verify_references
from scripts.pretrain_dagger_critic import critic_name, describe, load_returns
from scripts.train_hybrid_ppo_pilot import (BUDGET, PPO_SETTINGS, VALIDATION, digest,
                                          evaluation, make_ppo, rank, worker)
from scripts.train_interaction_dqn import save_json


SEEDS = (337, 463, 719)
HISTORICAL = Path("runs/hybrid_pilot_20261126")
EPOCHS = 12


def state_digest(policy, critic):
    return digest(value for key, value in policy.state_dict().items() if critic_name(key) == critic)


def predict_dataset(policy, data, keys, *, probabilities=False):
    values, probs = [], []
    with torch.no_grad():
        for batch in torch.arange(len(data["mc_returns"])).split(256):
            obs = {key: data[key][batch] for key in keys}
            values.append(policy.predict_values(obs).flatten())
            if probabilities:
                dist = policy.get_distribution(obs, action_masks=data["masks"][batch])
                torch.testing.assert_close(dist.log_prob(data["actions"][batch]), data["log_probs"][batch],
                                           atol=2e-5, rtol=2e-5)
                probs.append(dist.distribution.probs.clone())
    return torch.cat(values), torch.cat(probs) if probabilities else None


def summarize_returns(values, target):
    residual = values - target
    return dict(rmse=residual.square().mean().sqrt().item(), mae=residual.abs().mean().item(),
                explained_variance=1 - residual.var().item() / target.var().item(),
                predicted_mean=values.mean().item(), predicted_sd=values.std().item(),
                target_mean=target.mean().item(), target_sd=target.std().item())


def warm_critic(model, root, label, seed):
    started = perf_counter()
    policy = model.policy
    policy.set_training_mode(False)
    keys = tuple(model.observation_space.spaces)
    manifest = json.loads((root / "episodes.json").read_text())
    assert manifest["gamma"] == PPO_SETTINGS["gamma"]
    episodes = [row for row in manifest["episodes"] if row["temperature"] == label]
    data = {}
    for split, seeds in SPLITS.items():
        rows = [r for r in episodes if r["split"] == split]
        assert [r["seed"] for r in rows] == list(seeds)
        data[split] = load_returns(root / label, rows, (*keys, "actions", "log_probs"), "{seed}.npz")
    train, val, held = (data[split] for split in SPLITS)
    actor, critic, _ = describe(model)
    frozen = {key: val.clone() for key, val in policy.state_dict().items() if not critic_name(key)}
    held_before, probs_before = predict_dataset(policy, held, keys, probabilities=True)
    initial_val, _ = predict_dataset(policy, val, keys)
    history = [dict(epoch=0, **summarize_returns(initial_val, val["mc_returns"]))]
    best = history[0]
    best_state = None
    for parameter in actor:
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(critic, lr=3e-4)
    generator = torch.Generator().manual_seed(seed)
    try:
        for epoch in range(1, EPOCHS + 1):
            for batch in torch.randperm(len(train["mc_returns"]), generator=generator).split(256):
                obs = {key: train[key][batch] for key in keys}
                values = policy.predict_values(obs).flatten()
                loss = smooth_l1_loss(values / 50, train["mc_returns"][batch] / 50)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic, 1.)
                optimizer.step()
            values, _ = predict_dataset(policy, val, keys)
            row = dict(epoch=epoch, **summarize_returns(values, val["mc_returns"]))
            history.append(row)
            if row["rmse"] < best["rmse"]:
                best = row
                best_state = {key: value.clone() for key, value in policy.state_dict().items() if critic_name(key)}
    finally:
        for parameter in actor:
            parameter.requires_grad_(True)
    if best_state is not None:
        policy.load_state_dict(best_state, strict=False)
    held_after, probs_after = predict_dataset(policy, held, keys, probabilities=True)
    actor_identical = all(torch.equal(value, policy.state_dict()[key]) for key, value in frozen.items())
    assert actor_identical and torch.equal(probs_before, probs_after)
    before = summarize_returns(held_before, held["mc_returns"])
    after = summarize_returns(held_after, held["mc_returns"])
    return dict(history=history, selected=best, holdout_before=before, holdout_after=after,
                validation_improved=best["epoch"] > 0,
                holdout_improved=after["rmse"] < before["rmse"],
                actor_identical=actor_identical, probability_checks=len(probs_before),
                log_probability_temperature_checked=label,
                interactions={split: sum(row["steps"] for row in episodes if row["split"] == split)
                              for split in SPLITS},
                states={split: len(rows["mc_returns"]) for split, rows in data.items()},
                seconds=perf_counter()-started)


def historical_checkpoint(args, config, selected, path):
    arm = {"t1": "hybrid", "t05": "tempered"}[args.temperature]
    directory = HISTORICAL / f"{arm}_{args.seed}"
    if args.critic != "random" or not (directory / "result.json").exists():
        return None
    historical = json.loads((directory / "result.json").read_text())
    assert all(config[key] == historical[key] for key in
               ("seed", "temperature", "settings", "budget", "initial_actor", "initial_critic", "rewards"))
    archived = directory / historical["selected"]["file"]
    current = MaskablePPO.load(path, device="cpu")
    old = MaskablePPO.load(archived, device="cpu")
    identical = selected["steps"] == historical["selected"]["steps"] and all(
        torch.equal(value, old.policy.state_dict()[key]) for key, value in current.policy.state_dict().items())
    return str(archived) if identical else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic", choices=("random", "warm"), required=True)
    parser.add_argument("--temperature", choices=TEMPERATURES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_references()
    torch.set_num_threads(1)
    temperature = TEMPERATURES[args.temperature]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = list({r["family"]: r for r in json.loads(
        Path("runs/dagger_targeted_data/holdout.json").read_text())["episodes"]}.values())
    env = SubprocVecEnv([partial(worker, args.seed, str(args.output / "online"))], start_method="spawn")
    started = perf_counter()
    try:
        model = make_ppo(env, args.seed, temperature, algorithm=CriticAuditPPO)
        source = MaskablePPO.load(REFERENCE["dagger42"], device="cpu").policy.state_dict()
        initial = model.policy.state_dict()
        model.policy.load_state_dict({k: v if critic_name(k) else source[k] for k, v in initial.items()})
        actor = state_digest(model.policy, critic=False)
        random_critic = state_digest(model.policy, critic=True)
        assert actor == digest(value for key, value in source.items() if not critic_name(key))
        pretrain = None
        if args.critic == "warm":
            pretrain = warm_critic(model, args.rollouts, args.temperature, args.seed)
            save_json(args.output / "pretrain.json", pretrain)
            print("pretrain", args.seed, args.temperature, pretrain["holdout_before"]["rmse"],
                  pretrain["holdout_after"]["rmse"], "holdout improved", pretrain["holdout_improved"], flush=True)
            # Keep the four factorial arms even when selection does not generalize to holdout.
            if not pretrain["validation_improved"]:
                raise RuntimeError("No fitted critic improved the RMSE on the selection trajectories.")
        assert actor == state_digest(model.policy, critic=False)
        assert not model.policy.optimizer.state_dict()["state"]
        model.set_random_seed(args.seed)
        config = dict(critic=args.critic, seed=args.seed, temperature=temperature, danger=False, encoder="cnn",
                      budget=BUDGET, settings=PPO_SETTINGS, validation_normal=VALIDATION,
                      rewards=PacmanEnv.REWARD_CONFIG, initial_actor=actor, base_critic=random_critic,
                      initial_critic=state_digest(model.policy, critic=True))
        save_json(args.output / "config.json", config)
        model.save(args.output / "initial.zip")
        checkpoints = []
        training_seconds = 0.
        for step in (4096, 8192):
            phase = perf_counter()
            model.learn(4096, reset_num_timesteps=False, log_interval=None)
            training_seconds += perf_counter() - phase
            filename = f"ppo_{step:06}.zip"
            model.save(args.output / filename)
            row = dict(file=filename, steps=step, **evaluation(model, cases))
            checkpoints.append(row)
            save_json(args.output / f"validation_{step:06}.json", row)
            print(args.critic, args.temperature, args.seed, step, row["full"]["victory"],
                  row["recovery"]["victory"], flush=True)
        with (args.output / "online.monitor.csv").open(encoding="utf8") as stream:
            next(stream)
            logs = list(csv.DictReader(stream))
        np.savez_compressed(args.output / "first_rollout.npz", **model.first_rollout)
        best = max(checkpoints, key=rank)
        archived = historical_checkpoint(args, config, best, args.output / best["file"])
        report = dict(**config, selected=best, checkpoints=checkpoints, actual_steps=model.num_timesteps,
                      historical_equivalent_checkpoint=archived, training_seconds=training_seconds,
                      pretrain_seconds=pretrain["seconds"] if pretrain else 0.,
                      elapsed_seconds=perf_counter() - started, collected_episodes=len(logs),
                      collected_wins=sum(row["victory"]=="True" for row in logs),
                      collected_reward=mean(float(row["r"]) for row in logs),
                      rollout_diagnostics=model.rollout_diagnostics)
        save_json(args.output / "result.json", report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
