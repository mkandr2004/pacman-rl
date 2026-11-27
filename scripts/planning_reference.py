"""Evaluate the named classical controller against the real engine."""

import argparse
import json
from pathlib import Path

from rl.pacman_env import PacmanEnv
from rl.planning_reference import PlanningReference
from rl.curriculum import CurriculumEnv


def play(seed, ghosts=7, render=False, danger_weight=2.5, anticipate=True, local_avoidance=True):
    env = (PacmanEnv if ghosts == 7 else lambda **kwargs: CurriculumEnv(ghosts, **kwargs))(
        max_episode_steps=3000, render_mode="human" if render else None)
    policy = PlanningReference(seed,danger_weight,anticipate,local_avoidance)
    try:
        env.reset(seed=seed)
        initial = env._last_raw_observation
        trace = []
        for step in range(3000):
            before = env._last_raw_observation
            action = policy.predict(env)
            _, _, terminal, truncated, info = env.step(action)
            after = env._last_raw_observation
            rect = env._game._screen_field_mapper.get_pacman().get_rect()
            width,height = after["screen_size"]
            if not (0 <= rect.left and rect.right <= width and 0 <= rect.top and rect.bottom <= height):
                raise RuntimeError("Reference went outside the game")
            trace.append(dict(action=action, before=before["pacman"], after=after["pacman"],
                              target=policy.target_kind, health=info["health"], score=info["score"],
                              walls=sorted(set(before["walls"])-set(after["walls"]))))
            if terminal or truncated:
                break
        return dict(seed=seed,ghosts=ghosts,steps=step+1,victory=info["victory"],
                    score=info["score"],game_over=info["game_over"],timeout=truncated,
                    health=info["health"],points=initial["points_remaining"]-info["points_remaining"],
                    coins=initial["coins_remaining"]-info["coins_remaining"],trace=trace)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes",type=int,default=10)
    parser.add_argument("--seed",type=int,default=20000)
    parser.add_argument("--ghosts",type=int,choices=(0,1,2,4,7),default=7)
    parser.add_argument("--danger-weight",type=float,default=2.5)
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--static",action="store_true",help="Earlier reference without short path forecasts")
    parser.add_argument("--no-local-avoidance",action="store_true",
                        help="Keep the same A* targets and risk map but remove short trajectory comparison")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for seed in range(args.seed,args.seed+args.episodes):
        result=play(seed,args.ghosts,args.render,args.danger_weight,not args.static,
                    not args.no_local_avoidance)
        (args.output/f"seed_{seed}.json").write_text(json.dumps(result,indent=2))
        results.append({key:value for key,value in result.items() if key!="trace"})
        print(results[-1],flush=True)
    (args.output/"results.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
