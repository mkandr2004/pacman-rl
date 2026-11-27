"""Small supervised ablation for robust recovery with frozen-policy rehearsal."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rl.anchored_ppo import masked_kl
from rl.imitation_eval import play_network, summary
from scripts.diagnose_dagger_robustness import REFERENCE, verify_references
from scripts.pretrain_dagger_critic import critic_name


ARMS = ("uniform", "weighted", "weighted_set")
UPDATES = 120
VALIDATE_AT = (40, 80, 120)


def load_arrays(folder, records):
    parts = []
    for record in records:
        with np.load(folder / record["archive"], allow_pickle=False) as data:
            parts.append({key: data[key] for key in data.files})
    return {key: torch.from_numpy(np.concatenate([part[key] for part in parts]))
            for key in parts[0]}


def batch(data, indices):
    return {key: value[indices] for key, value in data.items()}


def distribution(policy, data):
    obs = {key: data[key] for key in policy.observation_space.spaces}
    return policy.get_distribution(obs, action_masks=data["masks"]).distribution


def acceptable_loss(logits, acceptable, weights):
    # Reward probability mass assigned to the demonstrated set without forcing
    # equal probabilities between multiple winning actions.
    losses = -logits.masked_fill(~acceptable, -torch.inf).logsumexp(-1)
    return (losses * weights).sum() / weights.sum()


def evaluate(model, full_seeds, recovery):
    before = torch.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        games = [play_network(model, seed, danger=False) for seed in full_seeds]
        resumes = [dict(family=c["family"], **play_network(
            model, c["seed"], danger=False, prefix=c["prefix"])) for c in recovery]
    assert torch.equal(before, torch.get_rng_state())
    families = Counter(row["family"] for row in resumes if row["victory"])
    critical = families["prefix_160"] + families["prefix_161"]
    return dict(full=summary(games), recovery=summary(resumes), families=families, critical=critical)


def validation_rank(result, reference):
    full = result["full"]["victory"]
    compliant = full >= reference["full"]["victory"] - 1
    return (compliant, *(
        (result["recovery"]["victory"], result["critical"], full)
        if compliant else (full, result["recovery"]["victory"], result["critical"])
    ), -result["full"]["max_stagnation"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    verify_references()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    train = [row for row in manifest["episodes"] if row["split"] == "train"]
    val = [row for row in manifest["episodes"] if row["split"] == "validation"]
    val_seeds = set(manifest["full_validation"]) | {row["seed"] for row in val}
    training_seeds = {row["seed"] for row in train + manifest["anchor"]}
    test_seeds = set(manifest["full_test"]) | {row["seed"] for row in manifest["episodes"]
                                                 if row["split"] == "test"}
    assert not (val_seeds & training_seeds or test_seeds & (val_seeds | training_seeds))
    recovery = load_arrays(args.data, train)
    anchors = load_arrays(args.data, [row for row in manifest["anchor"] if row["victory"]])
    if (recovery["acceptable"] & ~recovery["masks"]).any() or not recovery["acceptable"].any(1).all():
        raise ValueError("The targets must be nonempty subsets of the physical action mask.")
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(arm=args.arm, seed=args.seed, danger=False, encoder="cnn", updates=UPDATES,
                  learning_rate=.000025, recovery_batch=48, initial_state_batch=8,
                  anchor_batch=96, anchor_weight=3.,
                  data=str(args.data), validation_rng_isolated=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    ref = MaskablePPO.load(REFERENCE["dagger42"], device="cpu")
    model = MaskablePPO.load(REFERENCE["dagger42"], device="cpu")
    for name, parameter in model.policy.named_parameters():
        parameter.requires_grad_(not critic_name(name))
    trainable = [p for p in model.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config["learning_rate"])
    started = perf_counter()
    reference = evaluate(ref, manifest["full_validation"], val)
    print("reference", reference["full"]["victory"], reference["recovery"]["victory"], flush=True)
    curves, losses = [], []
    starts = torch.nonzero(recovery["relative_step"] == 0).flatten().numpy()
    for step in range(1, UPDATES+1):
        start_count = config["initial_state_batch"]
        indices = np.concatenate((rng.choice(starts, size=start_count),
                                  rng.integers(len(recovery["masks"]),
                                               size=config["recovery_batch"] - start_count)))
        difficult = batch(recovery, indices)
        rehearsal = batch(anchors, rng.integers(len(anchors["masks"]), size=config["anchor_batch"]))
        logits = distribution(model.policy, difficult).logits
        labels = torch.nn.functional.one_hot(difficult["actions"].long(), 4).bool()
        if args.arm == "weighted_set":
            labels = difficult["acceptable"]
        steps = difficult["relative_step"]
        weights = ((1. + (steps < 24) + 2. * (steps < 8))
                   if args.arm != "uniform" else torch.ones_like(steps, dtype=logits.dtype))
        with torch.no_grad():
            old = distribution(ref.policy, rehearsal)
        kl = masked_kl(old, distribution(model.policy, rehearsal)).mean()
        target = acceptable_loss(logits, labels, weights)
        loss = target + config["anchor_weight"] * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.)
        optimizer.step()
        losses.append(dict(step=step, target_loss=target.item(), anchor_kl=kl.item()))
        if step in VALIDATE_AT:
            filename = f"actor_{step:03}.zip"
            model.save(args.output / filename)
            result = evaluate(model, manifest["full_validation"], val)
            curves.append(dict(step=step, checkpoint=filename, **result))
            (args.output / "curve.json").write_text(json.dumps(curves, indent=2))
            print(args.arm, args.seed, step, result["full"]["victory"],
                  result["recovery"]["victory"], result["critical"], flush=True)
    selected = max(curves, key=lambda row: validation_rank(row, reference))
    same_critic = all(torch.equal(val, ref.policy.state_dict()[name])
                      for name, val in model.policy.state_dict().items() if critic_name(name))
    assert same_critic
    (args.output / "losses.json").write_text(json.dumps(losses, indent=2))
    (args.output / "result.json").write_text(json.dumps(
        dict(**config, selected=selected, baseline_validation=reference,
             critic_unchanged=same_critic, seconds=perf_counter()-started), indent=2))
    verify_references()


if __name__ == "__main__":
    main()
