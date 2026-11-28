"""Replay paired lobby matches and report movement, captures and collection gaps."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

import numpy as np

from main.arena_models import ARENA_MODELS, GHOST_MODEL, TEAM_MODEL, load_model
from rl.ghost_env import DIRECTIONS, WAIT, SlowGhostEnv
from rl.ghost_evaluation import seed_sampling
from rl.ghost_interception import interception_actions
from rl.ghost_team_env import GhostTeamEnv


MODES = ("classic", "slow_ppo", "team_ppo", "interception")


def replay(opponent, mode, controller, seed):
    seed_sampling(seed)
    env = (SlowGhostEnv(opponent, track_classic=True) if mode == "slow_ppo"
           else GhostTeamEnv(opponent, 2, track_classic=True))
    try:
        observation, info = env.reset(seed=seed)
        slots = (env.slow_id,) if mode == "slow_ppo" else env.slots
        if mode == "classic":
            slots = ()
        ghosts = tuple(env.base._game._screen_field_mapper.get_ghosts())
        stats = [dict(index=index, kind=type(ghost).__name__, controlled=index in slots,
                      immobile=Counter(), stalled=0, longest=None, reversals=0,
                      blocked_direction_frames=0)
                 for index, ghost in enumerate(ghosts)]
        initial_health = info["health"]
        captures = Counter()
        collection_gap = longest_gap = 0
        opening = []
        previous = env.base._last_raw_observation["ghosts"]
        two_steps_back = None
        while True:
            before = info
            commands = {}
            if mode == "classic":
                observation, _, terminated, truncated, info = env.step_classic()
            else:
                masks = env.action_masks()
                if mode == "interception":
                    actions = interception_actions(env, masks)
                else:
                    actions, _ = controller.predict(observation, deterministic=False, action_masks=masks)
                mask_rows = np.atleast_2d(masks)
                for slot, action, mask in zip(slots, np.atleast_1d(actions), mask_rows, strict=True):
                    commands[slot] = (
                        "WAIT" if action == WAIT else DIRECTIONS[action].name,
                        [direction.name for direction, allowed in zip(DIRECTIONS, mask[:WAIT]) if allowed],
                    )
                observation, _, terminated, truncated, info = env.step(actions)
            captures.update(index for index, _ in info["capture_events"])
            collected = any(info[key] < before[key] for key in ("points_remaining", "coins_remaining"))
            collection_gap = 0 if collected else collection_gap + 1
            longest_gap = max(longest_gap, collection_gap)
            raw = env.base._last_raw_observation
            if len(opening) < 30:
                opening.append(raw["pacman"])
            for index, ghost in enumerate(stats):
                moved_frames = info["ghost_moved_frames"][index]
                action, legal = commands.get(index, ("classic", None))
                if ghost["controlled"] and action != "WAIT":
                    ghost["blocked_direction_frames"] += info["active_frames"] - moved_frames
                if two_steps_back is not None and raw["ghosts"][index] != previous[index]:
                    ghost["reversals"] += int(raw["ghosts"][index] == two_steps_back[index])
                if moved_frames:
                    ghost["stalled"] = 0
                    continue
                reason = {"WAIT": "no_legal_direction", "classic": "classic_no_movement"}.get(
                    action, "direction_without_movement")
                ghost["immobile"][reason] += 1
                ghost["stalled"] += 1
                if ghost["longest"] is None or ghost["stalled"] > ghost["longest"]["decisions"]:
                    if not ghost["controlled"]:
                        direction = ghosts[index].moving_direction
                        action = direction.name if direction is not None else None
                        legal = [direction.name for direction in env.base._game.ghost_directions(index)]
                    ghost["longest"] = dict(decisions=ghost["stalled"], end_step=info["elapsed_steps"],
                                            position=list(env.base._last_raw_observation["ghosts"][index]),
                                            last_action=action, legal=legal)
            two_steps_back, previous = previous, raw["ghosts"]
            if terminated or truncated:
                break
        assert sum(bool(info[key]) for key in ("victory", "ghost_win", "timeout", "window_closed")) == 1
        for index, ghost in enumerate(stats):
            del ghost["stalled"]
            ghost["captures"] = captures[index]
        return dict(seed=seed, mode=mode, victory=info["victory"], ghost_win=info["ghost_win"],
                    timeout=info["timeout"], score=info["score"], final_health=info["health"],
                    starting_health=initial_health, lives_lost=sum(captures.values()),
                    controlled_captures=sum(captures[index] for index in slots),
                    longest_collection_gap=longest_gap, decisions=info["elapsed_steps"],
                    opening=opening, ghosts=stats)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[92000, 95207, 95817])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    controllers = {}
    for mode, entry in (("slow_ppo", GHOST_MODEL), ("team_ppo", TEAM_MODEL)):
        if mode in args.modes:
            controllers[mode] = load_model(entry)
    games = []
    groups = defaultdict(list)
    for entry in ARENA_MODELS:
        opponent = load_model(entry)
        for mode in args.modes:
            for seed in args.seeds:
                result = replay(opponent, mode, controllers.get(mode), seed)
                result["pacman"] = entry.name
                games.append(result)
                groups[(entry.name, mode)].append(result)
    summary = [
        dict(pacman=pacman, mode=mode, games=len(rows),
             wins=sum(row["victory"] for row in rows), losses=sum(row["ghost_win"] for row in rows),
             timeouts=sum(row["timeout"] for row in rows),
             mean_lives_lost=mean(row["lives_lost"] for row in rows),
             mean_controlled_captures=mean(row["controlled_captures"] for row in rows),
             distinct_openings=len({tuple(row["opening"]) for row in rows}),
             mean_longest_collection_gap=mean(row["longest_collection_gap"] for row in rows))
        for (pacman, mode), rows in groups.items()
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(dict(summary=summary, games=games), stream, indent=2)
            stream.write("\n")
    print(json.dumps(summary if args.output else dict(summary=summary, games=games), indent=2))


if __name__ == "__main__":
    main()
