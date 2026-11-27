"""Paired diagnostics of frozen DAgger policies on the unchanged game."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rl.imitation_env import make_imitation_env


REFERENCE = {
    "dagger42": Path("runs/frozen_dagger_95_5/full_update.zip"),
    "dagger241": Path("runs/frozen_dagger_97/fraction_65.zip"),
}
TEMPERATURES = (0., .25, .50, .75, 1., 1.25)
DIAGNOSTIC_SEEDS = (*range(65000, 65032), *range(65100, 65112))
MODELS = {}


def verify_references():
    for path in REFERENCE.values():
        record = json.loads(path.with_name("integrity.json").read_text())
        for filename in (path, Path(record["source"])):
            if hashlib.sha256(filename.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {filename}")


def model_for(name):
    if name not in MODELS:
        torch.set_num_threads(1)
        MODELS[name] = MaskablePPO.load(REFERENCE[name], device="cpu")
    return MODELS[name]


@torch.no_grad()
def policy_logits(model, obs, mask):
    tensors, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(tensors, action_masks=mask)
    return distribution.distribution.logits.flatten()


def sample(logits, mask, temperature, rng):
    if temperature == 0:
        return int(logits.argmax())
    if temperature == 1:
        # Matches Categorical.sample's softmax and multinomial with a private RNG.
        probabilities = logits.softmax(-1)
    else:
        probabilities = (logits / temperature).masked_fill(~torch.as_tensor(mask), -torch.inf).softmax(-1)
    return int(torch.multinomial(probabilities, 1, generator=rng))


def describe_state(base):
    raw = base._last_raw_observation
    mapper = base._game._screen_field_mapper
    pacman = mapper.get_pacman().get_rect()
    ghosts = []
    for ghost in mapper.get_ghosts():
        rect = ghost.get_rect()
        clearance = (max(pacman.left - rect.right, rect.left - pacman.right, 0)
                     + max(pacman.top - rect.bottom, rect.top - pacman.bottom, 0))
        direction = ghost.moving_direction
        previous = getattr(ghost, "_prev_step", None)
        ghosts.append(dict(type=type(ghost).__name__, position=ghost.get_sprite_pos(),
                           direction=direction.name if direction is not None else None,
                           patrol_index=getattr(ghost, "_instruction_index", None),
                           forbidden_reverse=previous.name if previous is not None else None,
                           patrol_progress=[ghost._passed_horizontal, ghost._passed_vertical],
                           clearance=clearance))
    return dict(position=raw["pacman"], score=raw["score"], health=raw["health"],
                effects=raw["effects"], ghosts=ghosts,
                remaining_items=raw["points_remaining"] + raw["coins_remaining"])


def run_episode(name, seed, temperature=0., *, prefix=(), force=(), state_steps=()):
    model = model_for(name)
    rng = torch.Generator().manual_seed(seed)
    env = make_imitation_env(danger=False, prefix=prefix)
    base = env.unwrapped
    steps, states = [], {}
    try:
        obs, _ = env.reset(seed=seed)
        first = base._last_raw_observation
        for index in range(3000):
            mask = base.action_masks()
            logits = policy_logits(model, obs, mask)
            action = int(force[index]) if index < len(force) else sample(logits, mask, temperature, rng)
            if not mask[action]:
                raise ValueError(f"Invalid recorded action at decision {index + 1}.")
            if index in state_steps:
                states[index] = describe_state(base)
            before = base._last_raw_observation
            obs, _, terminal, truncated, info = env.step(action)
            steps.append((*logits.softmax(-1).tolist(), action, int(logits.argmax()),
                          *before["pacman"], before["health"], before["score"],
                          info["health"], info["score"], info["points_remaining"],
                          info["coins_remaining"]))
            if terminal or truncated:
                break
        outcome = dict(model=name, seed=seed, temperature=temperature, steps=len(steps),
                       victory=bool(info["victory"]), death=bool(info["game_over"]),
                       timeout=bool(truncated and not terminal), score=info["score"], health=info["health"],
                       points=first["points_remaining"] - info["points_remaining"],
                       coins=first["coins_remaining"] - info["coins_remaining"])
        return outcome, np.array(steps, dtype=np.float32), states
    finally:
        env.close()


def trial(job):
    name, seed, temperature, directory = job
    result, trace, _ = run_episode(name, seed, temperature)
    np.savez_compressed(directory / f"{name}_{seed}_{temperature:g}.npz", trace=trace)
    return result


def branch(job):
    model, seed, baseline, sampled = job
    differences = np.flatnonzero(baseline[:len(sampled), 4] != sampled[:len(baseline), 4])
    if len(differences) == 0:
        return None
    index = int(differences[0])
    forced = [*baseline[:index, 4].astype(int), int(sampled[index, 4])]
    result, trace, details = run_episode(model, seed, force=forced, state_steps=(index,))
    opponent = baseline[index, :4].argsort()[-2]
    result.update(decision=index + 1, greedy_action=int(baseline[index, 4]),
                  sampled_action=int(sampled[index, 4]), probabilities=baseline[index, :4].tolist(),
                  margin=float(baseline[index, int(baseline[index, 4])] - baseline[index, opponent]),
                  state=details[index], greedy_win=bool(baseline[-1, 12:14].sum() == 0),
                  greedy_health=int(baseline[-1, 10]), greedy_score=int(baseline[-1, 11]),
                  delta_curves={
                      str(span): (trace[min(len(trace) - 1, index + span - 1), 10:12]
                                  - baseline[min(len(baseline) - 1, index + span - 1), 10:12]).tolist()
                      for span in (1, 10, 30, 90)
                  })
    return result


def analyze_trials(rows, directory):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["temperature"]].append(row)
    out = []
    for (model, temperature), group in groups.items():
        traces = [np.load(directory / f"{model}_{row['seed']}_{temperature:g}.npz")["trace"]
                  for row in group]
        trace = np.concatenate(traces)
        raw = trace[:, :4]
        p = raw
        if temperature not in (0, 1):
            p = p.astype(np.float64) ** (1 / temperature)
            p = p / p.sum(axis=1, keepdims=True)
        if temperature == 0:
            p = np.eye(4)[trace[:, 5].astype(int)]
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-30)), axis=1)
        ordered = np.sort(p, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        multi = (raw > 0).sum(axis=1) > 1
        raw_gap = np.sort(raw, axis=1)
        out.append(dict(model=model, temperature=temperature, games=len(group),
                        wins=sum(row["victory"] for row in group), decisions=len(trace),
                        score=np.mean([row["score"] for row in group]),
                        points=np.mean([row["points"] for row in group]),
                        coins=np.mean([row["coins"] for row in group]),
                        health=np.mean([row["health"] for row in group]),
                        multiaction_steps=int(multi.sum()), entropy=float(entropy[multi].mean()),
                        raw_entropy=float((-raw * np.log(np.maximum(raw, 1e-30))).sum(1)[multi].mean()),
                        raw_margin_median=float(np.median((raw_gap[:, -1] - raw_gap[:, -2])[multi])),
                        margin_median=float(np.median(margin[multi])),
                        margin_under_10pct=float((margin[multi] < .1).mean()),
                        sampled_non_greedy=float((trace[:, 4] != trace[:, 5]).mean())))
    return out


def read_trace(directory, model, seed, temperature):
    filename = directory / f"{model}_{seed}_{temperature:g}.npz"
    with np.load(filename, allow_pickle=False) as data:
        return data["trace"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    verify_references()
    args.output.mkdir(parents=True, exist_ok=False)
    traces = args.output / "traces"
    traces.mkdir()
    jobs = [(model, seed, t, traces) for seed in DIAGNOSTIC_SEEDS
            for t in TEMPERATURES for model in REFERENCE]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        with (args.output / "games.jsonl").open("x") as stream:
            for index, row in enumerate(pool.map(trial, jobs, chunksize=2), 1):
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                if index % 48 == 0:
                    stream.flush()
                    print(index, "/", len(jobs), flush=True)
        report = analyze_trials(rows, traces)
        (args.output / "temperatures.json").write_text(json.dumps(report, indent=2))
        # A one-action intervention isolates the first sample from repeated exploration.
        jobs = [(model, seed, read_trace(traces, model, seed, 0), read_trace(traces, model, seed, 1))
                for seed in DIAGNOSTIC_SEEDS for model in REFERENCE]
        branches = [row for row in pool.map(branch, jobs, chunksize=2) if row is not None]
    (args.output / "first_branches.json").write_text(json.dumps(branches, indent=2))
    verify_references()
    for row in report:
        print(row["model"], row["temperature"], row["wins"], "/", row["games"], flush=True)


if __name__ == "__main__":
    main()
