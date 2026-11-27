"""Trace an existing PPO without learning, including physical safety metrics."""
import argparse
import json
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO
from rl.pacman_env import PacmanEnv
from rl.wrappers import NormalizePositions


def audit(model_path, seeds, output, stop_on_exit=False):
    torch.set_num_threads(1)
    model = MaskablePPO.load(model_path, device="cpu")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in seeds:
        env = NormalizePositions(PacmanEnv(max_episode_steps=3000))
        obs, _ = env.reset(seed=seed)
        raw = env.unwrapped._last_raw_observation
        initial_points, initial_coins = raw["points_remaining"], raw["coins_remaining"]
        previous_walls = set(raw["walls"])
        events, exits, broken, border_broken, lost_health, stalled = [], 0, 0, 0, 0, 0
        original_advance = env.unwrapped._game.advance
        frame = 0
        width, height = raw["screen_size"]
        def traced_advance(*args, **kwargs):
            nonlocal previous_walls, frame, exits, broken, border_broken
            state, terminal, info = original_advance(*args, **kwargs)
            frame += 1
            removed = previous_walls - set(state["walls"])
            previous_walls = set(state["walls"])
            border = [p for p in removed if p[0] == 0 or p[1] == 0
                      or p[0] + 60 == width or p[1] + 60 == height]
            player = env.unwrapped._game._screen_field_mapper.get_pacman().get_rect()
            outside = not (player.left >= 0 and player.top >= 0
                           and player.right <= width and player.bottom <= height)
            exits += int(outside)
            broken += len(removed)
            border_broken += len(border)
            if removed or (outside and exits == 1):
                events.append(dict(frame=frame, position=state["pacman"],
                    action=getattr(kwargs.get("action"), "name", None),
                    effects=state["effects"], removed_walls=sorted(removed),
                    removed_borders=border, outside=outside, terminated=terminal,
                    terminal_info=info))
            return state, terminal, info
        env.unwrapped._game.advance = traced_advance
        reward_sum = 0.0
        terminated = truncated = False
        positions = []
        for step in range(3000):
            before = env.unwrapped._last_raw_observation
            action, _ = model.predict(obs, deterministic=True, action_masks=env.unwrapped.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            after = env.unwrapped._last_raw_observation
            positions.append(after["pacman"])
            stalled += int(before["pacman"] == after["pacman"])
            lost_health += max(0, before["health"] - after["health"])
            reward_sum += reward
            if terminated or truncated or (stop_on_exit and exits):
                break
        record = dict(seed=seed, steps=step + 1, score=info["score"],
            points_collected=initial_points - info["points_remaining"],
            coins_collected=initial_coins - info["coins_remaining"],
            lives_lost=lost_health, outside_frames=exits, walls_broken=broken,
            borders_broken=border_broken, stationary_decisions=stalled,
            unique_positions=len(set(positions)), victory=info["victory"],
            game_over=info["game_over"], terminated=terminated, truncated=truncated,
            reward=reward_sum, events=events)
        record["two_step_returns"] = sum(
            positions[i] == positions[i-2] != positions[i-1]
            for i in range(2, len(positions)))
        record["collection_fraction"] = (
            record["points_collected"] + record["coins_collected"]
        ) / max(1, initial_points + initial_coins)
        record["termination_reason"] = info.get("termination_reason")
        results.append(record)
        print(json.dumps(record), flush=True)
        env.close()
    (output / "audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000])
    parser.add_argument("--stop-on-exit", action="store_true")
    args = parser.parse_args()
    audit(args.model, args.seeds, args.output, args.stop_on_exit)
