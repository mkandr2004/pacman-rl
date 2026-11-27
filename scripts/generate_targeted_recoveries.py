"""Mine real learner states, certify winning continuations, and capture observations."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from rl.imitation_env import make_imitation_env
from scripts.diagnose_dagger_robustness import (
    describe_state, model_for, policy_logits, sample, verify_references,
)
from scripts.ppo_recovery_states import certify, sources


FAMILIES = ("one_life", "lemon_active", "two_threats", "after_error", "prefix_160", "prefix_161")
RANGES = {"train": range(68000, 68032), "validation": range(69000, 69020), "test": range(69500, 69526)}
QUOTAS = {"train": 10, "validation": 4, "test": 6}
FULL_VALIDATION = tuple(range(69070, 69094))
FULL_TEST = tuple(range(69700, 69800))


def mine(job):
    split, seed = job
    name = "dagger42" if seed % 2 else "dagger241"
    temperature = .6 if seed % 4 in (1, 2) else 0.
    model = model_for(name)
    env = make_imitation_env(danger=False)
    rng = torch.Generator().manual_seed(seed + 130000)
    actions, records, found = [], [], {}
    base = env.unwrapped
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(3000):
            state = describe_state(base)
            mask = base.action_masks()
            legal_choice = sum(mask) > 1
            facts = {
                "one_life": state["health"] == 1 and state["effects"][0] < .25,
                "lemon_active": state["effects"][2] >= .60,
                "two_threats": sum(g["clearance"] <= 60 for g in state["ghosts"]) >= 2,
                "prefix_160": state["position"] == (630, 630),
                "prefix_161": state["position"] == (618, 630),
            }
            for family, condition in facts.items():
                if condition and legal_choice and actions and state["remaining_items"] >= 2:
                    found.setdefault(family, actions.copy())
            logits = policy_logits(model, obs, mask)
            action = sample(logits, mask, temperature, rng)
            actions.append(action)
            records.append(action != int(logits.argmax()))
            obs, _, terminal, truncated, info = env.step(action)
            if info["health"] < state["health"] and info["health"] > 0 and not (terminal or truncated):
                deviated = [i for i in range(max(0, len(actions) - 9), len(actions)) if records[i]]
                if deviated:
                    cut = deviated[-1] + 1
                    found.setdefault("after_error", actions[:cut])
            if terminal or truncated:
                break
        return [dict(split=split, seed=seed, family=family, prefix=prefix, source=name,
                     source_temperature=temperature, source_win=info["victory"])
                for family, prefix in found.items()]
    finally:
        env.close()


def certify_candidate(candidate):
    family, split, seed, prefix = (candidate[key] for key in ("family", "split", "seed", "prefix"))
    result = certify((family, split, seed, prefix))
    result.update({key: val for key, val in candidate.items() if key not in result})
    if result["qualified"] and family in ("prefix_160", "prefix_161"):
        target = [630 if family == "prefix_160" else 618, 630]
        if list(result["position"]) != target:
            result.update(qualified=False, reason="other_position")
    return result


def early_lives_lost(seed, prefix, actions):
    env = make_imitation_env(danger=False, prefix=prefix)
    try:
        _, info = env.reset(seed=seed)
        initial = info["health"]
        for action in actions[:12]:
            assert env.unwrapped.action_masks()[action]
            _, _, terminal, truncated, info = env.step(action)
            if terminal or truncated:
                break
        return max(0, initial - info["health"])
    finally:
        env.close()


def capture(job):
    case, folder = job
    seed, prefix = case["seed"], case["prefix"]
    expert = case["winning_actions"]
    start = make_imitation_env(danger=False, prefix=prefix)
    try:
        start.reset(seed=seed)
        options = np.flatnonzero(start.unwrapped.action_masks()).tolist()
    finally:
        start.close()
    accepted = {expert[0]}
    added = {}
    reference_losses = early_lives_lost(seed, prefix, expert)
    for action in options:
        if action == expert[0]:
            continue
        other = certify(("branch", "train", seed, prefix + [action]))
        if other["qualified"]:
            continuation = [action, *other["winning_actions"]]
            losses = early_lives_lost(seed, prefix, continuation)
            if losses <= reference_losses:
                accepted.add(action)
            added[str(action)] = dict(actions=continuation, early_losses=losses)
    env = make_imitation_env(danger=False, prefix=prefix)
    base = env.unwrapped
    cols = defaultdict(list)
    try:
        obs, _ = env.reset(seed=seed)
        for index, action in enumerate(expert):
            mask = base.action_masks()
            assert mask[action]
            if index < 48 or index % 9 == 0:
                for key, value in obs.items():
                    cols[key].append(value)
                labels = accepted if index == 0 else {action}
                cols["actions"].append(action)
                cols["acceptable"].append([direction in labels for direction in range(4)])
                cols["masks"].append(mask)
                cols["relative_step"].append(index)
            obs, _, terminal, truncated, info = env.step(action)
            assert not truncated and terminal == (index + 1 == len(expert))
        assert info["victory"]
    finally:
        env.close()
    archive = folder / f"{case['family']}_{seed}.npz"
    np.savez_compressed(archive, **{k: np.stack(v) for k, v in cols.items()})
    return dict(archive=archive.name, first_acceptable=sorted(accepted), alternatives=added,
                initial_reference_losses=reference_losses, rows=len(cols["actions"]))


def normal_anchor(job):
    seed, folder = job
    model = model_for("dagger42")
    env = make_imitation_env(danger=False)
    cols = defaultdict(list)
    try:
        obs, _ = env.reset(seed=seed)
        for index in range(3000):
            mask = env.unwrapped.action_masks()
            logits = policy_logits(model, obs, mask)
            action = int(logits.argmax())
            if index % 4 == 0:
                for key, value in obs.items():
                    cols[key].append(value)
                cols["masks"].append(mask)
            obs, _, terminal, truncated, info = env.step(action)
            if terminal or truncated:
                break
    finally:
        env.close()
    if not info["victory"]:
        return dict(seed=seed, victory=False)
    archive = f"anchor_{seed}.npz"
    np.savez_compressed(folder / archive, **{k: np.stack(v) for k, v in cols.items()})
    return dict(seed=seed, victory=True, archive=archive, rows=len(cols["masks"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    verify_references()
    args.output.mkdir(parents=True, exist_ok=False)
    templates = sources()
    all_jobs = [(split, seed) for split, seeds in RANGES.items() for seed in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        mined = [row for group in pool.map(mine, all_jobs) for row in group]
        for split, base_seed in (("train", 68160), ("validation", 69160), ("test", 69560)):
            for offset, family in enumerate(("prefix_160", "prefix_161")):
                mined.extend(dict(family=family, split=split, seed=base_seed + 20 * offset + i,
                                  prefix=templates[family], source="recorded_critical_prefix",
                                  source_temperature=None) for i in range(QUOTAS[split] + 2))
        # Reuse certified pre-failure learner states from the previous independent test as training only.
        older = Path("runs/dagger_recovery_diagnostics/critical.jsonl")
        if older.exists():
            old_cases = json.loads(Path("runs/ppo_recovery_verified/qualified.json").read_text())
            previous = {entry["seed"]: entry["prefix"] for entry in old_cases if entry["family"] == "prefix_161"}
            for line in older.read_text().splitlines():
                if not line.endswith("}"):
                    continue
                row = json.loads(line)
                if row["model"] != "dagger42" or "rescue" not in row:
                    continue
                after = row["first_divergence"]
                if after and row["rescue"][0]["qualified"]:
                    prefix = previous[row["seed"]] + [int(s[4]) for s in row["trace_neural"][:after]]
                    mined.append(dict(family="after_error", split="train", seed=row["seed"], prefix=prefix,
                                      source="previous_critical_failure", source_temperature=0.))
        batches = defaultdict(list)
        for row in mined:
            batches[row["split"], row["family"]].append(row)
        rng = np.random.default_rng(618630)
        cert_jobs = []
        for (split, family), rows in sorted(batches.items()):
            rng.shuffle(rows)
            cert_jobs.extend(rows[: 2 * QUOTAS[split] + 4])
        (args.output / "mined.json").write_text(json.dumps(mined, indent=2))
        print("Mined", len(mined), "certifying", len(cert_jobs), flush=True)
        verified = []
        with (args.output / "certifications.jsonl").open("x") as stream:
            for index, row in enumerate(pool.map(certify_candidate, cert_jobs), 1):
                verified.append(row)
                stream.write(json.dumps(row) + "\n")
                if index % 20 == 0:
                    stream.flush()
                    print("Certified attempts", index, "/", len(cert_jobs), flush=True)
        caps = Counter()
        selected = []
        for row in verified:
            key = row["split"], row["family"]
            if row["qualified"] and caps[key] < QUOTAS[row["split"]]:
                selected.append(row)
                caps[key] += 1
        training = [row for row in selected if row["split"] == "train"]
        for row, data in zip(training, pool.map(capture, [(row, args.output) for row in training])):
            row.update(data)
        anchors = list(pool.map(normal_anchor, [(seed, args.output) for seed in range(68120, 68132)]))
    sets = {split: {row["seed"] for row in selected if row["split"] == split} for split in RANGES}
    sets["full_validation"] = set(FULL_VALIDATION)
    sets["full_test"] = set(FULL_TEST)
    sets["anchor"] = {row["seed"] for row in anchors}
    assert sum(map(len, sets.values())) == len(set().union(*sets.values()))
    report = dict(episodes=selected, anchor=anchors, count={"/".join(key): count for key, count in caps.items()},
                  full_validation=FULL_VALIDATION, full_test=FULL_TEST)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2))
    print(report["count"], flush=True)


if __name__ == "__main__":
    main()
