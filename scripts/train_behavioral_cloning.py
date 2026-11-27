"""Supervised maximum likelihood on the legal directions of expert replays."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from sb3_contrib import MaskablePPO

from rl.imitation_env import make_imitation_env
from rl.imitation_eval import play_network, summary
from rl.imitation_policy import ImitationCNN


def load_demo_split(root, manifest, split, observation_keys):
    rows = [row for row in manifest["episodes"] if row["split"] == split and row["victory"]]
    if not rows:
        raise ValueError(f"{split} has no winning expert trajectories")
    observations = {key: [] for key in observation_keys}
    actions, masks = [], []
    for row in rows:
        with np.load(root / row["file"], allow_pickle=False) as archive:
            chosen = archive["actions"]
            mask = archive["masks"]
            if len(chosen) != row["steps"] or not mask[np.arange(len(chosen)), chosen].all():
                raise ValueError(f"Corrupt action labels in seed {row['seed']}")
            for key, chunks in observations.items():
                values = archive[f"obs_{key}"]
                if len(values) != len(chosen) + 1:
                    raise ValueError(f"Missing successor observation in seed {row['seed']}")
                chunks.append(values[:-1])
            actions.append(chosen)
            masks.append(mask)
    return ({key: torch.from_numpy(np.concatenate(chunks)) for key, chunks in observations.items()},
            torch.tensor(np.concatenate(actions), dtype=torch.long),
            torch.from_numpy(np.concatenate(masks)))


def epoch_pass(policy, dataset, batch_size, danger, order, optimizer=None):
    obs, actions, masks = dataset
    loss_sum = correct = choice_correct = choice_count = 0
    for indices in order.split(batch_size):
        batch_obs = {key: value[indices] for key, value in obs.items()}
        if not danger:
            batch_obs["danger"].zero_()
        batch_actions = actions[indices]
        batch_masks = masks[indices]
        with torch.set_grad_enabled(optimizer is not None):
            dist = policy.get_distribution(batch_obs, action_masks=batch_masks)
            loss = -dist.log_prob(batch_actions).mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
        hits = dist.distribution.probs.argmax(dim=1) == batch_actions
        ambiguous = batch_masks.sum(dim=1) > 1
        loss_sum += loss.item() * len(indices)
        correct += hits.sum().item()
        choice_correct += (ambiguous & hits).sum().item()
        choice_count += ambiguous.sum().item()
    return dict(loss=loss_sum / len(actions), accuracy=correct / len(actions),
                choice_accuracy=choice_correct / max(choice_count, 1), n=len(actions),
                choice_decisions=choice_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("dense", "cnn"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--blank-danger", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Use positive epochs and batch size.")
    torch.set_num_threads(2)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    seeds = [row["seed"] for row in manifest["episodes"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("An episode seed occurs more than once across the dataset.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    danger = not args.blank_danger
    env = make_imitation_env(danger=danger)
    try:
        settings = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]),
                        activation_fn=nn.ReLU, share_features_extractor=False)
        if args.encoder == "cnn":
            settings["features_extractor_class"] = ImitationCNN
        model = MaskablePPO("MultiInputPolicy", env, policy_kwargs=settings, device="cpu",
                            seed=args.seed, n_steps=1024, batch_size=256, gamma=.995)
        policy = model.policy
        actor_parts = (policy.pi_features_extractor, policy.mlp_extractor.policy_net, policy.action_net)
        critic_parts = (policy.vf_features_extractor, policy.mlp_extractor.value_net, policy.value_net)
        actor_parameters = [param for part in actor_parts for param in part.parameters()]
        critic_parameters = [param for part in critic_parts for param in part.parameters()]
        initial_critic = [param.detach().clone() for param in critic_parameters]
        optimizer = torch.optim.AdamW(actor_parameters, lr=3e-4, weight_decay=1e-4)
        training, validation, test = [
            load_demo_split(args.dataset, manifest, split, env.observation_space.spaces)
            for split in ("train", "validation", "test")
        ]
        rng = torch.Generator().manual_seed(args.seed)
        config = dict(encoder=args.encoder, seed=args.seed, danger=danger, dataset=str(args.dataset),
                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=3e-4,
                      actor_parameters=sum(p.numel() for p in actor_parameters),
                      critic_parameters=sum(p.numel() for p in critic_parameters),
                      supervised_episodes="expert wins only", validation_seeds=list(range(55000, 55005)))
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
        model.save(args.output / "initial.zip")
        history = []
        best = None
        best_epoch = None
        optimization_seconds = 0.0
        for epoch in range(1, args.epochs + 1):
            epoch_start = perf_counter()
            policy.set_training_mode(True)
            train_metrics = epoch_pass(policy, training, args.batch_size, danger,
                                       torch.randperm(len(training[1]), generator=rng), optimizer)
            policy.set_training_mode(False)
            validation_metrics = epoch_pass(policy, validation, args.batch_size, danger,
                                            torch.arange(len(validation[1])))
            optimization_seconds += perf_counter()-epoch_start
            row = dict(epoch=epoch, train=train_metrics, validation=validation_metrics)
            if epoch % 4 == 0 or epoch == args.epochs:
                model.save(args.output / f"epoch_{epoch:02}.zip")
                games = [play_network(model, seed, danger=danger) for seed in config["validation_seeds"]]
                row["games"] = games
                row["autonomous"] = summary(games)
                performance = row["autonomous"]
                key = (performance["victory"], performance["points"] + performance["coins"],
                       -performance["oscillation_fraction"], -validation_metrics["loss"])
                if best is None or key > best:
                    best, best_epoch = key, epoch
                    model.save(args.output / "selected.zip")
            history.append(row)
            (args.output / "curve.json").write_text(json.dumps(history, indent=2))
            print(f"epoch={epoch} val={validation_metrics['choice_accuracy']:.3f} "
                  f"autonomous={row.get('autonomous',{}).get('victory','-')}", flush=True)
        if any(not torch.equal(param, original) for param, original in zip(critic_parameters, initial_critic)):
            raise RuntimeError("An independently initialized critic changed during imitation training.")
        chosen = MaskablePPO.load(args.output / "selected.zip", device="cpu")
        test_metrics = epoch_pass(chosen.policy, test, args.batch_size, danger, torch.arange(len(test[1])))
        report = dict(**config, selected_epoch=best_epoch, held_out_classification=test_metrics,
                      optimization_seconds=optimization_seconds, total_seconds=perf_counter()-started,
                      critic_unchanged=True, ppo_num_timesteps=chosen.num_timesteps)
        (args.output / "training_report.json").write_text(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
