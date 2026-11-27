"""DAgger with learner-only rollouts and rehearsal of successful trajectories."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch import nn

from rl.imitation_env import make_imitation_env
from rl.imitation_eval import play_network, summary
from rl.planning_reference import PlanningReference
from scripts.train_behavioral_cloning import load_demo_split


FROZEN = Path("runs/frozen_bc_86_7")
FAILURES = (54000, 54005, 54012, 54026)
VALIDATION_SEEDS = tuple(range(62000, 62024))
VALIDATION_RECOVERY = (62100, 62101)
TEST_SEEDS = tuple(range(63000, 63100))
TEST_RECOVERY = tuple(range(63100, 63106))


def verify_reference():
    expected = json.loads((FROZEN / "integrity.json").read_text())["sha256"]
    actual = hashlib.sha256((FROZEN / "selected.zip").read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("The protected BC checkpoint does not match its recorded SHA-256.")


def prefixes_from_manifest(manifest):
    return {len(row["prefix"]): row["prefix"] for row in manifest["episodes"] if row["prefix"]}


def collect(model, seed, prefix, stochastic, destination, teacher_seed):
    """Sample learner states before their real action, then attach expert labels."""
    torch.manual_seed(seed)
    env = make_imitation_env(danger=False, prefix=prefix)
    base = env.unwrapped
    teacher = PlanningReference(teacher_seed, local_avoidance=True)
    try:
        try:
            obs, _ = env.reset(seed=seed)
        except RuntimeError as error:
            if str(error) != "Scenario prefix terminated; invalid scenario":
                raise
            return dict(seed=seed, prefix_length=len(prefix), setup_terminated=True)
        initial = base._last_raw_observation
        columns = {key: [] for key in obs}
        records = []
        flags = []
        last_pickup = 0
        last_break = -100
        for step in range(3000):
            before = base._last_raw_observation
            mask = base.action_masks()
            action, _ = model.predict(obs, deterministic=not stochastic, action_masks=mask)
            actual = int(action)
            expert = teacher.predict(base)
            if not mask[actual] or not mask[expert]:
                raise RuntimeError(f"Illegal label or learner action on seed {seed}.")
            player = base._game._screen_field_mapper.get_pacman().get_rect()
            ghosts = base._game._screen_field_mapper.get_ghosts()
            gaps = [teacher._clearance(player, ghost.get_rect()) for ghost in ghosts]
            state_tags = set()
            if sum(gap <= 60 for gap in gaps) >= 2:
                state_tags.add("two_threats")
            if min(gaps) <= 24 and before["effects"][0] < .10:
                state_tags.add("imminent_contact")
            remaining = before["points_remaining"] + before["coins_remaining"]
            if remaining <= 12:
                state_tags.add("last_items")
            if step - last_pickup >= 32:
                state_tags.add("stagnation")
            if before["effects"][2] > 0:
                state_tags.add("breaker_active")
            if step - last_break <= 30:
                state_tags.add("post_wall")
            for key, value in obs.items():
                columns[key].append(value)
            obs, reward, terminated, truncated, info = env.step(actual)
            after = base._last_raw_observation
            if expert != actual:
                # A queued expert continuation is not valid after a learner disagreement.
                teacher.path.clear()
                teacher.detours.clear()
                teacher.expected = None
            removed_walls = sorted(set(before["walls"]) - set(after["walls"]))
            if removed_walls:
                last_break = step
                state_tags.add("wall_destroyed")
            if len(before["bonus_positions"][2]) > len(after["bonus_positions"][2]):
                state_tags.add("lemon_collected")
            if remaining > after["points_remaining"] + after["coins_remaining"]:
                last_pickup = step
            if after["health"] < before["health"]:
                state_tags.add("life_lost")
                for preceding in flags[-12:]:
                    preceding.add("before_loss")
            flags.append(state_tags)
            records.append(dict(learner=actual, expert=expert, mask=mask.tolist(),
                                position=before["pacman"], ghosts=before["ghosts"],
                                walls_removed=removed_walls, health=before["health"],
                                next_health=after["health"], reward=reward))
            if terminated or truncated:
                break
        tags = Counter(tag for states in flags for tag in states)
        priorities = np.array([
            1 + min(4, sum(tag in states for tag in (
                "before_loss", "post_wall", "last_items", "stagnation", "two_threats", "breaker_active"
            ))) + 2 * ("imminent_contact" in states)
            for states in flags], dtype=np.float32)
        arrays = {f"obs_{key}": np.stack(values) for key, values in columns.items()}
        np.savez_compressed(destination, **arrays,
                           expert_actions=np.array([row["expert"] for row in records], dtype=np.uint8),
                           learner_actions=np.array([row["learner"] for row in records], dtype=np.uint8),
                           masks=np.array([row["mask"] for row in records], dtype=bool),
                           priorities=priorities)
        report = dict(seed=seed, expert_seed=teacher_seed, prefix=list(prefix), prefix_length=len(prefix),
                      start=initial["pacman"], file=destination.name, stochastic=stochastic,
                      winner=info["victory"], score=info["score"], health=info["health"],
                      steps=len(records), tags=dict(tags),
                      disagreement=sum(row["expert"] != row["learner"] for row in records))
        destination.with_suffix(".json").write_text(json.dumps(dict(
            **report, trace=[dict(**row, tags=sorted(states)) for row, states in zip(records, flags)]
        ), indent=2))
        return report
    finally:
        env.close()


def load_labeled(files, keys, action_key):
    obs = {key: [] for key in keys}
    actions, masks, priorities = [], [], []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            for key in obs:
                obs[key].append(data[f"obs_{key}"][:len(data["masks"])])
            actions.append(data[action_key])
            masks.append(data["masks"])
            priorities.append(data["priorities"] if "priorities" in data else
                              np.full(len(data["masks"]), 2., dtype=np.float32))
    return ({key: torch.from_numpy(np.concatenate(values)) for key, values in obs.items()},
            torch.tensor(np.concatenate(actions), dtype=torch.long),
            torch.from_numpy(np.concatenate(masks)), torch.from_numpy(np.concatenate(priorities)))


def sample_batch(dataset, size, rng, probabilities=None):
    obs, actions, masks = dataset[:3]
    indices = rng.choice(len(actions), size, p=probabilities)
    return {key: values[indices] for key, values in obs.items()}, actions[indices], masks[indices]


def update(model, original, anchors, annotated, seed, updates, learning_rate, recovery=None):
    policy = model.policy
    actor = [*policy.pi_features_extractor.parameters(),
             *policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters()]
    optimizer = torch.optim.AdamW(actor, lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    priorities = annotated[3].numpy().astype(np.float64)
    probabilities = priorities / priorities.sum()
    losses = []
    policy.set_training_mode(True)
    for _ in range(updates):
        batches = (sample_batch(original, 96, rng),
                   sample_batch(anchors, 96, rng),
                   sample_batch(annotated, 64 if recovery is None else 48, rng, probabilities))
        if recovery is not None:
            batches += (sample_batch(recovery, 16, rng),)
        obs = {key: torch.cat([batch[0][key] for batch in batches]) for key in batches[0][0]}
        obs["danger"].zero_()
        actions = torch.cat([batch[1] for batch in batches])
        masks = torch.cat([batch[2] for batch in batches])
        loss = -policy.get_distribution(obs, action_masks=masks).log_prob(actions).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor, 1.0)
        optimizer.step()
        losses.append(loss.item())
    policy.set_training_mode(False)
    return sum(losses) / len(losses)


def validate(model, prefixes):
    games = [play_network(model, seed, danger=False) for seed in VALIDATION_SEEDS]
    recovery = []
    for prefix in prefixes.values():
        for seed in VALIDATION_RECOVERY:
            try:
                recovery.append(play_network(model, seed, danger=False, prefix=prefix))
            except RuntimeError as error:
                if str(error) != "Scenario prefix terminated; invalid scenario":
                    raise
                recovery.append(dict(seed=seed, prefix_length=len(prefix), setup_terminated=True))
    valid = [row for row in recovery if not row.get("setup_terminated")]
    return dict(games=games, full=summary(games), recovery=recovery,
                recovery_wins=sum(row["victory"] for row in valid),
                recovery_valid=len(valid))


def ranking(validation):
    full = validation["full"]
    return (full["victory"], validation["recovery_wins"], full["score"],
            -full["oscillation_fraction"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("runs/expert_imitation_120"))
    parser.add_argument("--initial", type=Path, default=FROZEN / "selected.zip")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--recovery-data", type=Path, help="Optional winning expert continuations of old learner failures.")
    args = parser.parse_args()
    if args.rounds < 1 or args.updates < 1 or args.learning_rate <= 0:
        parser.error("Training requires a positive budget and learning rate.")
    torch.set_num_threads(2)
    verify_reference()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config = dict(encoder="cnn", danger=False, seed=args.seed, initial=str(args.initial),
                  initial_sha256=hashlib.sha256(args.initial.read_bytes()).hexdigest(),
                  rounds=args.rounds, updates=args.updates, learning_rate=args.learning_rate,
                  validation_seeds=VALIDATION_SEEDS, validation_recovery=VALIDATION_RECOVERY,
                  final_test_seeds=TEST_SEEDS, final_test_recovery=TEST_RECOVERY,
                  learner_probability=1, batch_mix=dict(original=96, successful_learner=96,
                  dagger=48 if args.recovery_data else 64, expert_recovery=16 if args.recovery_data else 0),
                  recovery_data=str(args.recovery_data) if args.recovery_data else None,
                  original_data=str(args.dataset))
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    reserved = set(VALIDATION_SEEDS + VALIDATION_RECOVERY + TEST_SEEDS + TEST_RECOVERY)
    if reserved.intersection(row["seed"] for row in manifest["episodes"]):
        raise ValueError("The demonstration dataset contains a reserved validation or test seed.")
    prefixes = prefixes_from_manifest(manifest)
    model = MaskablePPO.load(args.initial, device="cpu")
    critic = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()
              if name.startswith(("vf_features_extractor.", "mlp_extractor.value_net.", "value_net."))}
    keys = model.observation_space.spaces.keys()
    original = load_demo_split(args.dataset, manifest, "train", keys)
    recovery = None
    if args.recovery_data:
        branches = json.loads((args.recovery_data / "manifest.json").read_text())
        if reserved.intersection(row["seed"] for row in branches["episodes"]):
            raise ValueError("The recovery dataset contains a reserved validation or test seed.")
        winning = [args.recovery_data / row["file"] for row in branches["episodes"] if row["victory"]]
        if not winning:
            raise ValueError("Recovery dataset has no winning continuation.")
        recovery = load_labeled(winning, keys, "actions")
    history, label_files, anchor_files = [], [], []
    selected = None
    for round_index in range(1, args.rounds + 1):
        location = args.output / f"round_{round_index:02}"
        location.mkdir()
        first_seed = 61000 + ((args.seed + 31 * round_index) % 800)
        jobs = [(seed, [], False) for seed in FAILURES]
        jobs += [(first_seed+i, [], i >= 8) for i in range(10)]
        jobs += [(first_seed+10+i, prefix, False) for i, prefix in enumerate(prefixes.values())]
        # Original successful BC episodes are eligible for rehearsal only,
        # and are excluded from the new validation and test seed sets.
        if round_index == 1:
            jobs += [(seed, [], False) for seed in (54001, 54002, 54003, 54006, 54007, 54010)]
        reports = []
        for index, (seed, prefix, stochastic) in enumerate(jobs):
            archive = location / f"episode_{index:02}_seed_{seed}.npz"
            row = collect(model, seed, prefix, stochastic, archive, args.seed + seed)
            reports.append(row)
            if not row.get("setup_terminated"):
                label_files.append(archive)
                if round_index == 1 and row["winner"]:
                    anchor_files.append(archive)
        if not anchor_files:
            raise RuntimeError("No successful initial rollouts are available for rehearsal.")
        anchors = load_labeled(anchor_files, keys, "learner_actions")
        annotated = load_labeled(label_files, keys, "expert_actions")
        nll = update(model, original, anchors, annotated, args.seed+round_index,
                     args.updates, args.learning_rate, recovery)
        checkpoint = args.output / f"dagger_{round_index:02}.zip"
        model.save(checkpoint)
        validation = validate(model, prefixes)
        row = dict(round=round_index, checkpoint=checkpoint.name, nll=nll,
                   annotation_count=len(annotated[1]), anchor_count=len(anchors[1]),
                   collection=reports, validation=validation)
        history.append(row)
        if selected is None or ranking(validation) > ranking(selected["validation"]):
            selected = row
        (args.output / "curve.json").write_text(json.dumps(history, indent=2))
        print(f"round={round_index} annotations={len(annotated[1])} "
              f"validation={validation['full']['victory']}/{len(VALIDATION_SEEDS)} "
              f"recovery={validation['recovery_wins']}/{validation['recovery_valid']}", flush=True)
        verify_reference()
    current = model.policy.state_dict()
    if any(not torch.equal(value, current[key]) for key, value in critic.items()):
        raise RuntimeError("The value function changed during DAgger.")
    result = dict(**config, selected=selected["checkpoint"], training_seconds=perf_counter()-started,
                  critic_unchanged=True, ppo_num_timesteps=model.num_timesteps)
    (args.output / "training_report.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
