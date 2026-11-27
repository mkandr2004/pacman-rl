"""Inspect the neural policy on the unchanged zero-ghost stage."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from rl.curriculum import CurriculumEnv
from rl.wrappers import NormalizePositions
from scripts.train_curriculum import PREFIXES


def audit(model, seed, prefix, stochastic=False, render=False):
    torch.manual_seed(seed)
    env = NormalizePositions(CurriculumEnv(
        0,max_episode_steps=3000,render_mode="human" if render else None))
    try:
        obs,_ = env.reset(seed=seed)
        for action in prefix:
            obs,_,_,_,_ = env.step(action)
        base = env.unwrapped
        raw = base._last_raw_observation
        start = raw["pacman"]
        start_item_count = raw["points_remaining"]+raw["coins_remaining"]
        original_walls = set(raw["walls"])
        initial_lemons = len(raw["bonus_positions"][2])
        position_counts = Counter()
        trace = []
        escaped = False
        items = 0
        max_stagnation = 0
        stagnation = 0
        for step in range(300 if prefix else 3000):
            mask = base.action_masks()
            action,_ = model.predict(obs,deterministic=not stochastic,action_masks=mask)
            obs,_,terminal,truncated,info = env.step(action)
            next_state = base._last_raw_observation
            position = next_state["pacman"]
            position_counts[position] += 1
            collected = start_item_count-info["points_remaining"]-info["coins_remaining"]
            stagnation = stagnation+1 if collected==items else 0
            max_stagnation = max(stagnation,max_stagnation)
            items = collected
            escaped |= sum(abs(x-y) for x,y in zip(start,position))>=120
            trace.append(dict(before=raw["pacman"],after=position,mask=mask.tolist(),
                              action=int(action),items=items))
            raw = next_state
            if terminal or truncated:
                break
        return dict(seed=seed,setup_length=len(prefix),start=start,
                    stochastic=stochastic,steps=step+1,score=info["score"],items=items,
                    wins=int(info["victory"]),escaped=escaped,
                    max_stagnation=max_stagnation,frequent_positions=position_counts.most_common(4),
                    lemons=initial_lemons-len(raw["bonus_positions"][2]),
                    walls=sorted(original_walls-set(raw["walls"])),
                    last_decisions=trace[-20:])
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    model=MaskablePPO.load(args.model,device="cpu")
    latest=json.loads((PREFIXES.parent/"natural_trace.json").read_text())
    prefixes=json.loads(PREFIXES.read_text())
    prefixes.append([row["action"] for row in latest["decisions"][:161]])
    cases=[audit(model,31000,[],render=args.render)]
    if not args.render:
        for prefix in prefixes:
            cases.append(audit(model,31000,prefix))
        cases.extend(audit(model,seed,[],stochastic=True) for seed in range(31000,31003))
    (args.output/"report.json").write_text(json.dumps(cases,indent=2))
    for row in cases:
        print({key:value for key,value in row.items() if key!="last_decisions"})


if __name__=="__main__":
    main()
