"""Fit V(s) to returns observed under frozen DAgger action probabilities."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch.nn.functional import smooth_l1_loss

from rl.imitation_env import make_imitation_env


FROZEN_ROOT = Path("runs/frozen_dagger_97")
FROZEN = FROZEN_ROOT / "fraction_65.zip"
GAMMA = .995
_COLLECTOR = None


def verify_frozen():
    manifest = json.loads((FROZEN_ROOT / "integrity.json").read_text())
    for path in (FROZEN, Path(manifest["source"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise RuntimeError(f"Protected reference checksum mismatch: {path}")


def worker_init():
    global _COLLECTOR
    torch.set_num_threads(1)
    _COLLECTOR = MaskablePPO.load(FROZEN, device="cpu")


def collect(job):
    seed, directory = job
    torch.manual_seed(seed)
    env = make_imitation_env(danger=False)
    try:
        obs, _ = env.reset(seed=seed)
        columns = {key: [] for key in obs}
        rewards, masks, actions = [], [], []
        for _ in range(3000):
            mask = env.unwrapped.action_masks()
            action, _ = _COLLECTOR.predict(obs, deterministic=False, action_masks=mask)
            for key, value in obs.items():
                columns[key].append(value)
            masks.append(mask)
            actions.append(int(action))
            obs, reward, terminal, truncated, info = env.step(action)
            rewards.append(reward)
            if terminal or truncated:
                break
        returns = np.zeros(len(rewards), dtype=np.float32)
        total = 0.
        for index in reversed(range(len(rewards))):
            total = rewards[index] + GAMMA * total
            returns[index] = total
        # Preserve the data even when an episode is censored. Such episodes
        # are excluded from the Monte Carlo fit to avoid zero bootstrap.
        np.savez_compressed(directory / f"seed_{seed}.npz",
                            **{key: np.stack(values) for key, values in columns.items()},
                            rewards=rewards, mc_returns=returns, masks=np.stack(masks),
                            actions=actions)
        return dict(seed=seed, steps=len(rewards), terminal=terminal,
                    truncated=truncated, score=info["score"], victory=info["victory"])
    finally:
        env.close()


def load_returns(directory, episodes, keys, name_pattern="seed_{seed}.npz"):
    data = {key: [] for key in (*keys, "mc_returns", "masks")}
    for episode in episodes:
        if not episode["terminal"]:
            continue
        with np.load(directory / name_pattern.format(seed=episode["seed"]), allow_pickle=False) as stream:
            for key in data:
                data[key].append(stream[key])
    if not data["mc_returns"]:
        raise RuntimeError("No terminal rollout available for critic fit.")
    return {key: torch.from_numpy(np.concatenate(values)) for key, values in data.items()}


def critic_name(key):
    return key.startswith(("vf_features_extractor.", "mlp_extractor.value_net.", "value_net."))


def describe(model):
    policy = model.policy
    groups = (
        (policy.pi_features_extractor, policy.mlp_extractor.policy_net, policy.action_net),
        (policy.vf_features_extractor, policy.mlp_extractor.value_net, policy.value_net),
    )
    actor, critic = [[parameter for module in group for parameter in module.parameters()] for group in groups]
    if {id(p) for p in actor}.intersection(id(p) for p in critic):
        raise RuntimeError("The critic and actor are not disjoint.")
    optimizer = policy.optimizer.state_dict()
    report = dict(actor_parameters=sum(p.numel() for p in actor),
                  critic_parameters=sum(p.numel() for p in critic),
                  shared_features=policy.share_features_extractor,
                  ppo_steps=model.num_timesteps, ppo_updates=model._n_updates,
                  optimizer_class=type(policy.optimizer).__name__,
                  optimizer_moment_entries=len(optimizer["state"]),
                  optimizer_learning_rates=[g["lr"] for g in optimizer["param_groups"]])
    return actor, critic, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    verify_frozen()
    torch.set_num_threads(2)
    torch.manual_seed(241)
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    model = MaskablePPO.load(FROZEN, device="cpu")
    actor, critic, audit = describe(model)
    (args.output / "initial_audit.json").write_text(json.dumps(audit, indent=2))
    print(audit, flush=True)
    fit_seeds = list(range(65000, 65032))
    check_seeds = list(range(65100, 65112))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        episodes = list(pool.map(collect, ((seed, args.output) for seed in fit_seeds + check_seeds)))
    policy = model.policy
    keys = tuple(model.observation_space.spaces)
    train = load_returns(args.output, episodes[:len(fit_seeds)], keys)
    validation = load_returns(args.output, episodes[len(fit_seeds):], keys)
    expected = {k: v.clone() for k, v in policy.state_dict().items() if not critic_name(k)}
    probe = {k: validation[k][:2048] for k in keys}
    probe_masks = validation["masks"][:2048]
    optimizer = torch.optim.Adam(critic, lr=3e-4)
    for parameter in actor:
        parameter.requires_grad_(False)

    @torch.no_grad()
    def score():
        values = torch.cat([policy.predict_values({k: validation[k][i:i+256] for k in keys}).flatten()
                            for i in range(0, len(validation["mc_returns"]), 256)])
        target = validation["mc_returns"]
        return dict(rmse=((values-target).square().mean().sqrt().item()),
                    mae=(values-target).abs().mean().item(),
                    explained_variance=1 - (values-target).var().item()/target.var().item())

    with torch.no_grad():
        before = policy.get_distribution(probe, action_masks=probe_masks).distribution.probs.clone()
    history = [dict(epoch=0, **score())]
    best = history[0]
    for epoch in range(1, 21):
        for index in torch.randperm(len(train["mc_returns"])).split(256):
            values = policy.predict_values({key: train[key][index] for key in keys}).flatten()
            loss = smooth_l1_loss(values / 50, train["mc_returns"][index] / 50)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic, 1.0)
            optimizer.step()
        current = dict(epoch=epoch, **score())
        history.append(current)
        if epoch % 5 == 0:
            model.save(args.output / f"critic_epoch_{epoch:02}.zip")
        if current["rmse"] < best["rmse"]:
            best = current
            model.save(args.output / "critic_warm.zip")
    if best["epoch"] == 0:
        raise RuntimeError("Critic pretraining did not improve held-out RMSE.")
    warm = MaskablePPO.load(args.output / "critic_warm.zip", device="cpu")
    with torch.no_grad():
        after = warm.policy.get_distribution(probe, action_masks=probe_masks).distribution.probs
    if not torch.equal(before, after) or any(
        not torch.equal(value, warm.policy.state_dict()[key]) for key, value in expected.items()
    ):
        raise RuntimeError("Actor changed during critic pretraining.")
    verify_frozen()
    report = dict(initial=audit, fit_seeds=fit_seeds, validation_seeds=check_seeds,
                  episodes=episodes, gamma=GAMMA, history=history, best=best,
                  validation_states=len(validation["mc_returns"]), train_states=len(train["mc_returns"]),
                  actor_bitwise_equal=True, identical_probability_states=len(before),
                  ppo_optimizer_still_empty=not warm.policy.optimizer.state_dict()["state"],
                  seconds=perf_counter()-started)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("Initial:", history[0], "warm:", best, "actor unchanged on", len(before), "states", flush=True)


if __name__ == "__main__":
    main()
