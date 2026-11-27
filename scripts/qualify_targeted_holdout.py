"""Ensure the held-out states contain novel observations and actual winning replays."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from rl.imitation_env import make_imitation_env
from scripts.generate_targeted_recoveries import FAMILIES


def signature(observation):
    contents = (np.asarray(value).tobytes() for _, value in sorted(observation.items()))
    return hashlib.sha256(b"".join(contents)).hexdigest()


def inspect(case):
    env = make_imitation_env(danger=False, prefix=case["prefix"])
    try:
        obs, _ = env.reset(seed=case["seed"])
        return signature(obs)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source = json.loads((args.data / "manifest.json").read_text())
    train = [row for row in source["episodes"] if row["split"] == "train"]
    validation = [row for row in source["episodes"] if row["split"] == "validation"]
    frozen_exclusions = {row["seed"] for row in train + validation + source["anchor"]}
    frozen_exclusions |= set(source["full_validation"]) | set(source["full_test"])
    training_signatures = set()
    for row in train + [r for r in source["anchor"] if r["victory"]]:
        with np.load(args.data / row["archive"], allow_pickle=False) as data:
            keys = {k: data[k] for k in data.files if k not in ("actions","masks","acceptable","relative_step")}
        for index in range(row["rows"]):
            training_signatures.add(signature({key: value[index] for key, value in keys.items()}))
    selected = [r for r in source["episodes"] if r["split"] == "test"]
    additional = [json.loads(line) for line in (args.data / "certifications.jsonl").read_text().splitlines()]
    variants = selected + [r for r in additional if r["split"] == "test" and r["qualified"]]
    unique = {(r["seed"], r["family"], tuple(r["prefix"])): r for r in variants}
    novelty, games = [], []
    counts = Counter()
    blocked = set(training_signatures)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        blocked.update(pool.map(inspect, validation))
        for row, state in zip(unique.values(), pool.map(inspect, unique.values(), chunksize=2)):
            novel = state not in blocked
            novelty.append(dict(seed=row["seed"], family=row["family"], novel=novel))
            if novel and counts[row["family"]] < 6:
                counts[row["family"]] += 1
                games.append(row)
                blocked.add(state)
    assert not ({r["seed"] for r in games} & frozen_exclusions)
    assert all(counts[family] == 6 for family in FAMILIES), counts
    path = args.data / "holdout.json"
    with path.open("x") as stream:
        json.dump(dict(episodes=games, distinct_training_observations=len(training_signatures),
                       observation_uniqueness_audit=novelty), stream, indent=2)
    print(counts, flush=True)


if __name__ == "__main__":
    main()
