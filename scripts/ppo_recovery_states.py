"""Qualify complete-game recovery states before they enter an RL reset pool."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from rl.imitation_env import make_imitation_env
from rl.pacman_env import PacmanEnv
from rl.planning_reference import PlanningReference


def sources():
    runs = Path("runs")
    archived = json.loads((runs / "dagger_recovery_branches/scenarios.json").read_text())
    definitions = {key: value for key, value in archived.items()
                   if key in ("prefix_40", "prefix_160", "prefix_161", "prefix_240", "last_items")}
    winning = json.loads((runs / "bc_eval_cnn_42_blank/win_seed_54001.json").read_text())["trace"]
    first_wall = next(i for i, step in enumerate(winning) if step["walls"])
    definitions["post_wall"] = [step["action"] for step in winning[:first_wall + 1]]
    definitions["lemon_active"] = [step["action"] for step in winning[:first_wall - 12]]
    failing = json.loads((runs / "dagger_recovery_branches/learner_seed_54000.json").read_text())["trace"]
    first_single_life = next(i for i, step in enumerate(failing) if step["health"] == 1)
    definitions["one_life"] = [step["action"] for step in failing[:first_single_life + 12]]
    assessed = json.loads((runs / "dagger_bc42_run42/round_01/episode_00_seed_54000.json").read_text())["trace"]
    conflict = next(i for i, step in enumerate(assessed)
                    if i > 20 and "two_threats" in step["tags"] and "imminent_contact" in step["tags"])
    definitions["two_threats"] = [step["learner"] for step in assessed[:conflict]]
    return definitions


def certify(job):
    family, split, seed, prefix = job
    base_row = dict(family=family, split=split, seed=seed, prefix=prefix)
    env = make_imitation_env(danger=False, prefix=prefix)
    base = env.unwrapped
    trials = []
    try:
        for weight, local in ((2.5, True), (1.75, False)):
            try:
                env.reset(seed=seed)
            except RuntimeError as error:
                if str(error) != "Scenario prefix terminated; invalid scenario":
                    raise
                return {**base_row, "qualified": False, "reason": "prefix_terminal"}
            raw = base._last_raw_observation
            mapper = base._game._screen_field_mapper
            player = mapper.get_pacman().get_rect()
            threats = sum(PlanningReference._clearance(player, g.get_rect()) <= 60
                          for g in mapper.get_ghosts())
            facts = dict(position=raw["pacman"], health=raw["health"], threats_within_60=threats,
                         objects_remaining=raw["points_remaining"] + raw["coins_remaining"],
                         breaker_active=bool(raw["effects"][2]),
                         opened=len(base.scenario_initial_raw["walls"]) - len(raw["walls"]))
            meets_condition = {
                "one_life": facts["health"] == 1,
                "two_threats": threats >= 2,
                "post_wall": facts["opened"] > 0,
                "lemon_active": facts["breaker_active"],
                "last_items": 0 < facts["objects_remaining"] <= 12,
            }.get(family, True)
            if not meets_condition:
                return {**base_row, **facts, "qualified": False, "reason": "condition_absent"}
            teacher = PlanningReference(seed + 180000, danger_weight=weight, local_avoidance=local)
            actions = []
            for _ in range(3000):
                action = teacher.predict(base)
                if not base.action_masks()[action]:
                    raise RuntimeError("Invalid expert action in recovery certification.")
                _, _, terminal, truncated, info = env.step(action)
                actions.append(action)
                if terminal or truncated:
                    break
            won = bool(info["victory"] and info["points_remaining"] == info["coins_remaining"] == 0)
            trials.append(dict(weight=weight, local_avoidance=local, won=won,
                               score=info["score"], steps=len(actions)))
            if won:
                return {**base_row, **facts, "qualified": True,
                        "trials": trials, "winning_actions": actions}
        return {**base_row, **facts, "qualified": False, "reason": "expert_failed", "trials": trials}
    finally:
        env.close()


def supplemental_jobs():
    """Discover additional held-out prefixes from previously won games."""
    paths = (
        Path("runs/bc_eval_cnn_42_blank/win_seed_54001.json"),
        Path("runs/bc_first_network_victory/win_seed_55003.json"),
        Path("runs/dagger_final_100/win_dagger_select_bc42_run241_seed_63000.json"),
    )
    candidates = {"one_life": [], "lemon_active": []}
    for path in paths:
        replay = json.loads(path.read_text())
        actions = []
        env = PacmanEnv(max_episode_steps=3000)
        seen = set()
        try:
            env.reset(seed=replay["seed"])
            for row in replay["trace"]:
                action = row["action"]
                actions.append(action)
                env.step(action)
                raw = env._last_raw_observation
                remaining = raw["points_remaining"] + raw["coins_remaining"]
                if raw["effects"][2] >= .60 and "lemon_active" not in seen:
                    candidates["lemon_active"].append(actions.copy())
                    seen.add("lemon_active")
                if raw["health"] == 1 and raw["effects"][0] < .25 and remaining >= 2 and "one_life" not in seen:
                    candidates["one_life"].append(actions.copy())
                    seen.add("one_life")
        finally:
            env.close()
    indexed = [(family,prefix) for family,prefixes in candidates.items() for prefix in prefixes]
    return [(family, "test", 64300 + index * 20 + i, prefix)
            for index,(family,prefix) in enumerate(indexed) for i in range(12)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    prefixes = sources()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(family, split, base + index * 10 + i, prefix)
            for index, (family, prefix) in enumerate(prefixes.items())
            for split, base, count in (("train", 64000, 6), ("validation", 64100, 4), ("test", 64200, 6))
            for i in range(count)]
    if args.supplement:
        jobs = supplemental_jobs()
        prefixes = dict.fromkeys(job[0] for job in jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(certify, jobs, chunksize=1))
    (args.output / "qualified.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    for family in prefixes:
        print(family, {split: sum(r["qualified"] for r in rows if r["family"] == family and r["split"] == split)
                        for split in ("train", "validation", "test")}, flush=True)


if __name__ == "__main__":
    main()
