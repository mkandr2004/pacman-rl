"""Collect complete, independently seeded expert trajectories locally."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from rl.imitation_env import make_imitation_env
from rl.planning_reference import PlanningReference


SPLIT_STARTS = {"train": 51000, "validation": 52000, "test": 53000}
TRACE_ROOT = Path("runs/navigation_study/current_diagnostic")


def recorded_prefixes():
    prefixes = json.loads((TRACE_ROOT / "prefixes.json").read_text())
    trace = json.loads((TRACE_ROOT / "natural_trace.json").read_text())
    return {str(len(prefix)): prefix for prefix in [
        *prefixes, [step["action"] for step in trace["decisions"][:161]]
    ]}


def capture(job):
    root, split, seed, prefix, danger_weight, avoidance = job
    env = make_imitation_env(prefix=prefix)
    base = env.unwrapped
    expert = PlanningReference(seed + 70000, danger_weight, local_avoidance=avoidance)
    elapsed = perf_counter()
    try:
        obs, _ = env.reset(seed=seed)
        start = base._last_raw_observation
        game = base._game
        mapper = game._screen_field_mapper
        columns = {key: [value] for key, value in obs.items()}
        raw_positions, ghost_positions = [start["pacman"]], [start["ghosts"]]
        actions, masks, rewards, terminals, truncations = [], [], [], [], []
        clearances, threats, targets, events = [], [], [], []
        for step in range(3000):
            raw = base._last_raw_observation
            player = mapper.get_pacman().get_rect()
            distances = [expert._clearance(player, g.get_rect()) for g in mapper.get_ghosts()]
            clearances.append(min(distances))
            threats.append(sum(distance <= 60 for distance in distances))
            mask = base.action_masks()
            action = expert.predict(base)
            if not mask[action]:
                raise RuntimeError(f"Expert chose masked action on seed {seed}, decision {step}")
            obs, reward, terminal, truncated, info = env.step(action)
            after = base._last_raw_observation
            for key, value in obs.items():
                columns[key].append(value)
            actions.append(action)
            masks.append(mask)
            rewards.append(reward)
            terminals.append(terminal)
            truncations.append(truncated)
            raw_positions.append(after["pacman"])
            ghost_positions.append(after["ghosts"])
            targets.append(expert.target_kind or "none")
            event = {}
            for key in ("points", "coins", "walls", "bonuses"):
                removed = sorted(set(raw[key]) - set(after[key]))
                if removed:
                    event[key] = removed
            if raw["health"] != after["health"]:
                event["health_change"] = after["health"] - raw["health"]
            if event:
                events.append({"decision": step, **event})
            if terminal or truncated:
                break
        relative = f"{split}/seed_{seed}.npz"
        arrays = {f"obs_{key}": np.stack(values) for key, values in columns.items()}
        np.savez_compressed(root / relative, **arrays, actions=np.asarray(actions, dtype=np.uint8),
                            masks=np.stack(masks), rewards=np.asarray(rewards, dtype=np.float32),
                            terminated=np.asarray(terminals), truncated=np.asarray(truncations),
                            pacman_pixels=np.asarray(raw_positions, dtype=np.int16),
                            ghost_pixels=np.asarray(ghost_positions, dtype=np.int16),
                            ghost_clearance=np.asarray(clearances, dtype=np.int16),
                            threats_within_60=np.asarray(threats, dtype=np.uint8),
                            expert_target=np.asarray(targets, dtype="U5"))
        summary = dict(seed=seed, expert_seed=seed + 70000, split=split, file=relative,
                       prefix=list(prefix), prefix_length=len(prefix), start=start["pacman"],
                       start_score=start["score"], danger_weight=danger_weight,
                       local_avoidance=avoidance, steps=len(actions), victory=info["victory"],
                       game_over=info["game_over"], timeout=bool(truncated and not terminal),
                       score=info["score"], health=info["health"],
                       points=start["points_remaining"] - info["points_remaining"],
                       coins=start["coins_remaining"] - info["coins_remaining"],
                       walls=sum(len(event.get("walls", ())) for event in events),
                       close_threat_decisions=sum(distance <= 30 for distance in clearances),
                       two_threat_decisions=sum(n >= 2 for n in threats),
                       bonus_decisions=int((arrays["obs_effects"][:-1, :3] > 0).any(axis=1).sum()),
                       breaker_decisions=int((arrays["obs_effects"][:-1, 2] > 0).sum()),
                       seconds=perf_counter() - elapsed)
        (root / relative).with_suffix(".json").write_text(json.dumps(
            {**summary, "events": events}, indent=2), encoding="utf-8")
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train", type=int, default=84)
    parser.add_argument("--validation", type=int, default=18)
    parser.add_argument("--test", type=int, default=18)
    args = parser.parse_args()
    counts = {split: getattr(args, split) for split in SPLIT_STARTS}
    if any(not 1 <= n <= 300 for n in counts.values()) or args.workers < 1:
        parser.error("Each split needs 1–300 episodes and at least one worker.")
    args.output.mkdir(parents=True, exist_ok=False)
    prefixes = list(recorded_prefixes().values())
    jobs = []
    for split, first_seed in SPLIT_STARTS.items():
        (args.output / split).mkdir()
        for i in range(counts[split]):
            # 1 in 4 replays starts midgame with the seven original ghosts.
            prefix = prefixes[(i // 4) % len(prefixes)] if i % 4 == 3 else []
            jobs.append((args.output, split, first_seed+i, prefix,
                         (2.5, 1.75, 3.25)[i % 3], i % 3 == 0))
    started = perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(capture, jobs, chunksize=1):
            results.append(result)
            print(result["split"], result["seed"], result["steps"], result["victory"], flush=True)
    outcomes = {split: dict(Counter("win" if row["victory"] else
                                  "death" if row["game_over"] else "timeout"
                                  for row in results if row["split"] == split))
                for split in counts}
    manifest = dict(version=1, map="maps/map_2.txt",
                    map_hash=hashlib.sha256(Path("maps/map_2.txt").read_bytes()).hexdigest(),
                    frames_per_decision=4, ghosts=7, split_counts=counts, outcomes=outcomes,
                    transitions=sum(row["steps"] for row in results),
                    seconds=perf_counter()-started, episodes=results,
                    versions={package: version(package) for package in ("numpy", "pygame-ce", "gymnasium")})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
