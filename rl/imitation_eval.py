"""Full episodes driven solely by a neural policy and the physical mask."""

from statistics import mean
from time import perf_counter

import torch

from rl.imitation_env import make_imitation_env
from rl.pacman_env import PacmanEnv
from rl.wrappers import NormalizePositions


def play_network(model, seed, *, danger=True, stochastic=False, prefix=(), render=False,
                 legacy=False, record=False, measure_topology=False):
    if measure_topology:
        from scripts.navigation_study import reachable_collectibles

    started = perf_counter()
    torch.manual_seed(seed)
    if legacy and prefix:
        raise ValueError("Legacy evaluation expects a normal game start.")
    render_mode = "human" if render else None
    env = (NormalizePositions(PacmanEnv(max_episode_steps=3000, render_mode=render_mode))
           if legacy else make_imitation_env(danger=danger, prefix=prefix, render_mode=render_mode))
    base = env.unwrapped
    try:
        obs, _ = env.reset(seed=seed)
        first = base._last_raw_observation
        positions = [first["pacman"]]
        simulation_start = base._game._simulation_ms
        oscillations = last_pickup = max_stagnation = 0
        escape_step = None
        first_wall_step = None
        first_life_loss_after_wall = None
        reached_last_items = False
        useful_wall_events = newly_reachable_items = wall_events = 0
        reward_sum = 0.0
        trace = []
        for step in range(1, 3001):
            mask = base.action_masks()
            action, _ = model.predict(obs, deterministic=not stochastic, action_masks=mask)
            if not mask[int(action)]:
                raise RuntimeError("The network selected an invalid direction.")
            before = base._last_raw_observation
            obs, reward, terminal, truncated, info = env.step(action)
            after = base._last_raw_observation
            position = after["pacman"]
            if len(positions) >= 2 and position == positions[-2] != positions[-1]:
                oscillations += 1
            positions.append(position)
            if before["points_remaining"] + before["coins_remaining"] > info["points_remaining"] + info["coins_remaining"]:
                last_pickup = step
            max_stagnation = max(step - last_pickup, max_stagnation)
            if escape_step is None and last_pickup and sum(abs(a - b) for a, b in zip(position, first["pacman"])) >= 120:
                escape_step = step
            if len(before["walls"]) > len(after["walls"]):
                wall_events += 1
                if first_wall_step is None:
                    first_wall_step = step
                if measure_topology:
                    opened = reachable_collectibles({**before, "walls": after["walls"]}) - reachable_collectibles(before)
                    useful_wall_events += int(opened > 0)
                    newly_reachable_items += max(0, opened)
            if first_wall_step is not None and first_life_loss_after_wall is None and info["reward_terms"]["health"] < 0:
                first_life_loss_after_wall = step - first_wall_step
            reached_last_items |= info["points_remaining"] + info["coins_remaining"] <= 12
            if record:
                trace.append(dict(action=int(action), before=before["pacman"], after=position,
                                  score=info["score"], health=info["health"], reward=reward,
                                  walls=sorted(set(before["walls"]) - set(after["walls"]))))
            reward_sum += reward
            if terminal or truncated:
                break
        result = dict(seed=seed, prefix_length=len(prefix), start=first["pacman"], steps=step,
                      victory=info["victory"], death=info["game_over"],
                      timeout=bool(truncated and not terminal and not info["window_closed"]),
                      window_closed=info["window_closed"], score=info["score"],
                      points=first["points_remaining"] - info["points_remaining"],
                      coins=first["coins_remaining"] - info["coins_remaining"],
                      health=info["health"], oscillations=oscillations,
                      oscillation_fraction=oscillations / max(1, step - 1), max_stagnation=max_stagnation,
                      escape_step=escape_step, walls=len(set(first["walls"]) - set(after["walls"])),
                      lemons=len(first["bonus_positions"][2]) - len(after["bonus_positions"][2]),
                      reward=reward_sum, simulated_seconds=(base._game._simulation_ms-simulation_start)/1000,
                      first_wall_step=first_wall_step,
                      first_life_loss_after_wall=first_life_loss_after_wall,
                      reached_last_items=reached_last_items,
                      seconds=perf_counter()-started)
        if record:
            result["trace"] = trace
        if measure_topology:
            result.update(wall_events=wall_events, useful_wall_events=useful_wall_events,
                          newly_reachable_items=newly_reachable_items)
        return result
    finally:
        env.close()


def summary(games):
    return {
        "episodes": len(games),
        **{key: sum(game[key] for game in games) for key in ("victory", "death", "timeout")},
        **{key: mean(game[key] for game in games) for key in (
            "score", "points", "coins", "health", "steps", "oscillations",
            "oscillation_fraction", "max_stagnation", "lemons", "walls", "simulated_seconds", "seconds")},
    }
