"""Detailed evidence and paired benchmarks; never changes model files."""
import argparse
import json
from pathlib import Path
from statistics import mean
from unittest.mock import Mock

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from rl.navigation_lab import ScenarioEnv, HistoryEnv
from rl.wrappers import NormalizePositions
from rl.shaping import ProgressPotential


def reachable_collectibles(raw):
    """Static tile-connectivity METRIC, never consulted by the controller."""
    rows,cols = raw['screen_size'][1]//60,raw['screen_size'][0]//60
    walls={(y//60,x//60) for x,y in raw['walls']}
    start=(raw['pacman'][1]//60,raw['pacman'][0]//60)
    seen=set() if start in walls else {start}
    pending=list(seen)
    while pending:
        r,c=pending.pop()
        for cell in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=cell[0]<rows and 0<=cell[1]<cols and cell not in walls and cell not in seen:
                seen.add(cell)
                pending.append(cell)
    return sum((y//60,x//60) in seen for x,y in raw['points']+raw['coins'])


def rollout(model, seed, *, prefix=(), ghosts=True, stochastic=False,
            limit=3000, detailed=False, render_path=False, history=False, observation_path=None,
            decision_frames=4, fixture=None):
    torch.manual_seed(seed)
    base = (HistoryEnv if history else ScenarioEnv)(
        prefix=prefix, ghosts=ghosts, fixture=fixture, max_episode_steps=limit)
    env = NormalizePositions(base)
    obs, _ = env.reset(seed=seed)
    base.frames_per_action = decision_frames  # Diagnostic ablation only, AFTER identical setup.
    if render_path:
        base.render_mode = "human"
        base._clock = Mock()  # Draw identical frames, skip wall-clock throttling.
    initial = base._last_raw_observation
    origin = initial["pacman"]
    initial_items = initial["points_remaining"] + initial["coins_remaining"]
    positions, records, frames = [], [], []
    wall_events=[]
    observation_samples = {key: [] for key in obs} if observation_path else None
    potential = ProgressPotential()
    if detailed:
        potential(base.scenario_initial_raw)
    phi = potential(initial) if detailed else 0
    last_collect = stagnation_max = 0
    blocked_frames = total_frames = outside = border_breaks = walls_broken = 0
    lost = 0
    escaped = False
    original = base._game.advance
    def instrumented(*args, **kwargs):
        nonlocal blocked_frames, total_frames, outside, border_breaks, walls_broken
        before_position = tuple(base._game._screen_field_mapper.get_pacman().get_rect().topleft)
        before = base._game.observation() if detailed else None
        state, term, info = original(*args, **kwargs)
        total_frames += 1
        blocked_frames += int(before_position == state["pacman"])
        player = base._game._screen_field_mapper.get_pacman().get_rect()
        width, height = state["screen_size"]
        outside += int(player.left < 0 or player.top < 0 or player.right > width or player.bottom > height)
        if detailed:
            removed = set(before["walls"]) - set(state["walls"])
            frames.append(dict(frame=total_frames, action=kwargs["action"].name,
                before=before["pacman"], after=state["pacman"],
                blocked=before["pacman"] == state["pacman"], removed_walls=sorted(removed)))
        return state, term, info
    base._game.advance = instrumented
    try:
        for step in range(limit):
            raw = base._last_raw_observation
            mask = base.action_masks()
            if detailed:
                with torch.no_grad():
                    tensor, _ = model.policy.obs_to_tensor(obs)
                    distribution = model.policy.get_distribution(tensor, action_masks=mask)
                    probabilities = distribution.distribution.probs.cpu().numpy()[0]
                    unmasked = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]
            if observation_samples is not None:
                for key, value in obs.items():
                    observation_samples[key].append(np.array(value, copy=True))
            action, _ = model.predict(obs, deterministic=not stochastic, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            after = base._last_raw_observation
            removed = set(raw["walls"]) - set(after["walls"])
            walls_broken += len(removed)
            width, height = after["screen_size"]
            border_breaks += sum(x == 0 or y == 0 or x+60 == width or y+60 == height for x,y in removed)
            if detailed and removed:
                wall_events.append(dict(step=step+1,removed_walls=sorted(removed),
                    reachable_items_before=reachable_collectibles(raw),
                    reachable_items_after=reachable_collectibles(after),
                    newly_reachable_items_from_wall=(reachable_collectibles({**raw,'walls':after['walls']})-
                                                     reachable_collectibles(raw)),
                    collected_total=initial_items-after['points_remaining']-after['coins_remaining']))
            positions.append(tuple(after["pacman"]))
            before_count = raw["points_remaining"] + raw["coins_remaining"]
            count = after["points_remaining"] + after["coins_remaining"]
            if count < before_count:
                last_collect = step + 1
            stagnation_max = max(stagnation_max, step + 1-last_collect)
            lost += max(0, raw["health"] - after["health"])
            escaped |= abs(after["pacman"][0]-origin[0])+abs(after["pacman"][1]-origin[1]) >= 120
            next_phi = (0 if terminated else potential(after)) if detailed else 0
            if detailed:
                records.append(dict(step=step+1, action=int(action),
                    probabilities=probabilities.tolist(), mask=mask.tolist(),
                    probabilities_without_mask=unmasked.tolist(),
                    before=raw["pacman"], after=after["pacman"],
                    grid_changed=not np.array_equal(previous_obs["grid"], obs["grid"]) if step else None,
                    remaining_points=info["points_remaining"], remaining_coins=info["coins_remaining"],
                    ghosts=raw["ghosts"], effects=raw["effects"],
                    reward_terms=info["reward_terms"], potential_before=phi,
                    potential_after=next_phi, potential_term=.995*next_phi-phi))
            previous_obs = obs
            phi = next_phi
            if terminated or truncated:
                break
        if observation_samples is not None:
            np.savez_compressed(observation_path, **{k:np.stack(v) for k,v in observation_samples.items()})
        for event in wall_events:
            start=event['step']-1
            event['items_collected_next_50_decisions']=sum(
                max(0,records[j-1]['remaining_points']+records[j-1]['remaining_coins']-
                    records[j]['remaining_points']-records[j]['remaining_coins'])
                for j in range(start+1,min(len(records),start+51)))
        return dict(seed=seed, steps=step+1, simulation_frames=total_frames,
            survival_sim_seconds=total_frames*.01,
            score=info["score"], collection=initial_items-count,
            points_collected=initial["points_remaining"]-after["points_remaining"],
            coins_collected=initial["coins_remaining"]-after["coins_remaining"],
            collection_fraction=(initial_items-count)/max(1, initial_items),
            lives_lost=lost, victory=info["victory"], game_over=info["game_over"],
            truncated=truncated, escaped_zone=escaped,
            two_step_returns=sum(positions[i] == positions[i-2] != positions[i-1] for i in range(2,len(positions))),
            stagnation_max=stagnation_max, distinct_positions=len(set(positions)),
            blocked_frames=blocked_frames, outside_frames=outside, borders_broken=border_breaks,
            walls_broken=walls_broken, wall_events=wall_events, decisions=records, physical_frames=frames)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--scenario-prefixes", help="JSON with shared lists of recorded action prefixes")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = MaskablePPO.load(args.model, device="cpu")
    if args.diagnostic:
        base = rollout(model, 1000, detailed=True, history=args.history,
                       observation_path=root/"natural_observations.npz")
        (root/"natural_trace.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
        actions = [r["action"] for r in base["decisions"]]
        snapshot_indices = [40, min(160,len(actions)-2), min(240,len(actions)-2)]
        prefixes = ([actions[:i] for i in snapshot_indices] if not args.scenario_prefixes
                    else json.loads(Path(args.scenario_prefixes).read_text()))
        (root/"prefixes.json").write_text(json.dumps(prefixes), encoding="utf-8")
        cases = []
        for prefix in prefixes:
            index = len(prefix)
            for ghosts in (False, True):
                for stochastic in (False, True):
                    name = f"zone_{index}_ghosts_{ghosts}_stochastic_{stochastic}"
                    result = rollout(model, 1000, prefix=prefix, ghosts=ghosts,
                        stochastic=stochastic, limit=300, detailed=True, history=args.history,
                        observation_path=root/f"{name}_observations.npz")
                    (root/f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                    cases.append({"case":name, **{k:v for k,v in result.items() if k not in ("decisions","physical_frames")}})
                    print(cases[-1], flush=True)
        for fixture in ('lemon_upper', 'lemon_lower'):
            for ghosts in (False, True):
                for stochastic in (False, True):
                    name = f'{fixture}_ghosts_{ghosts}_stochastic_{stochastic}'
                    result = rollout(model,1000,fixture=fixture,ghosts=ghosts,stochastic=stochastic,
                                     limit=300,detailed=True,history=args.history,
                                     observation_path=root/f'{name}_observations.npz')
                    (root/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                    cases.append({'case':name, **{k:v for k,v in result.items()
                                                  if k not in ('decisions','physical_frames')}})
                    print(cases[-1],flush=True)
        # Exercise the rendering path with the same seed/actions, not image-only inference.
        rendered = rollout(model, 1000, detailed=True, limit=40, render_path=True, history=args.history)
        parity = [(r["after"],r["reward_terms"]) for r in base["decisions"][:40]] == [
            (r["after"],r["reward_terms"]) for r in rendered["decisions"]]
        (root/"scenarios.json").write_text(json.dumps(dict(cases=cases,render_parity=parity),indent=2),encoding="utf-8")
        example = NormalizePositions((HistoryEnv if args.history else ScenarioEnv)())
        try:
            observation, _ = example.reset(seed=1000)
            np.savez_compressed(root/"observation_example.npz", **observation)
        finally:
            example.close()
    else:
        results = []
        for seed in range(args.seed_start, args.seed_start+args.episodes):
            result = rollout(model, seed, stochastic=args.stochastic, history=args.history)
            results.append(result)
            print(f"seed={seed} score={result['score']} wins={result['victory']}", flush=True)
        summary = {key:mean(r[key] for r in results) for key in (
            "score","collection","collection_fraction","two_step_returns","stagnation_max","steps","lives_lost",
            "simulation_frames","blocked_frames","outside_frames","borders_broken","walls_broken",
            "survival_sim_seconds","points_collected","coins_collected")}
        summary.update(wins=sum(r["victory"] for r in results),
            game_overs=sum(r["game_over"] for r in results),
            timeouts=sum(r["truncated"] for r in results), episodes=len(results))
        (root/"results.json").write_text(json.dumps(dict(model=args.model,stochastic=args.stochastic,
            summary=summary,results=results),indent=2),encoding="utf-8")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
