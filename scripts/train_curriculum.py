"""Train and evaluate one explicit curriculum stage, with a promotion gate."""

import argparse
import json
from pathlib import Path
from functools import partial

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.curriculum import CurriculumEnv, STAGES
from rl.wrappers import NormalizePositions


PREFIXES = Path("runs/navigation_study/current_diagnostic/prefixes.json")
EVAL_SEEDS = tuple(range(30000,30010))
WIN_GATE = .8
SCENARIO_GATE = .8


def environment(ghosts, danger, logfile=None):
    torch.set_num_threads(1)
    return Monitor(NormalizePositions(CurriculumEnv(ghosts,danger_channel=danger,max_episode_steps=3000)),
                   filename=logfile,info_keywords=("score","victory","game_over"))


def evaluate(model, ghosts, danger, prefixes, *, render=False):
    env = NormalizePositions(CurriculumEnv(ghosts,danger_channel=danger,max_episode_steps=3000,
                                            render_mode="human" if render else None))
    def run(seed, prefix):
        obs,_ = env.reset(seed=seed)
        for action in prefix:
            obs,_,terminal,truncated,_ = env.step(action)
            if terminal or truncated:
                return dict(seed=seed,setup_survived=False,success=False,steps=0)
        raw = env.unwrapped._last_raw_observation
        x,y = raw["pacman"]
        original_items = raw["points_remaining"]+raw["coins_remaining"]
        escaped = False
        for step in range(300 if prefix else 3000):
            action,_ = model.predict(obs,deterministic=True,action_masks=env.unwrapped.action_masks())
            obs,_,terminal,truncated,info = env.step(action)
            raw = env.unwrapped._last_raw_observation
            items = original_items-info["points_remaining"]-info["coins_remaining"]
            px,py = raw["pacman"]
            escaped |= abs(px-x)+abs(py-y)>=120
            if terminal or truncated or (prefix and escaped and items):
                break
        return dict(seed=seed,setup_survived=True,victory=info["victory"],score=info["score"],
                    health=info["health"],steps=step+1,items=items,position=(px,py),
                    start=(x,y),game_over=info["game_over"],timeout=truncated,
                    success=bool(escaped and items and not info["game_over"]) if prefix else info["victory"])
    try:
        global_results = [run(seed,[]) for seed in EVAL_SEEDS]
        scenario_results = {str(len(prefix)):[run(seed,prefix) for seed in EVAL_SEEDS[:3]] for prefix in prefixes}
    finally:
        env.close()
    wins = sum(r["victory"] for r in global_results)
    scenario_rates = {key:np.mean([r["success"] for r in results]).item()
                      for key,results in scenario_results.items()}
    return dict(wins=wins,episodes=len(global_results),
                average_items=np.mean([r["items"] for r in global_results]).item(),
                average_score=np.mean([r["score"] for r in global_results]).item(),
                scenario_rates=scenario_rates, games=global_results, scenarios=scenario_results,
                gate_passed=wins/len(global_results)>=WIN_GATE and all(x>=SCENARIO_GATE for x in scenario_rates.values()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ghosts",type=int,choices=STAGES,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--budget",type=int,default=65536)
    parser.add_argument("--interval",type=int,default=16384)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--peer-previous",type=Path,
                        help="Independent successful run at the preceding stage")
    parser.add_argument("--danger",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.interval<=0 or args.interval%1024 or args.budget<=0 or args.budget%args.interval:
        parser.error("Use a positive budget divisible by an interval that is a multiple of 1024.")
    if args.ghosts:
        if not args.previous or not args.peer_previous:
            parser.error("Supply two mastered runs with independent seeds at the preceding ghost count.")
        previous=json.loads((args.previous/"result.json").read_text())
        peer=json.loads((args.peer_previous/"result.json").read_text())
        if (not previous["gate_passed"] or previous["ghosts"]!=STAGES[STAGES.index(args.ghosts)-1]
                or previous["danger"]!=args.danger or previous["seed"]!=args.seed):
            parser.error("The previous run must pass its gate at the prior stage with matching seed and observations.")
        if (not peer["gate_passed"] or peer["ghosts"]!=previous["ghosts"]
                or peer["danger"]!=args.danger or peer["seed"]==args.seed):
            parser.error("The peer must pass the same stage and use a different seed.")
    elif args.previous or args.peer_previous:
        parser.error("Stage zero starts from a new policy.")
    args.output.mkdir(parents=True,exist_ok=False)
    prefixes=json.loads(PREFIXES.read_text())
    natural=json.loads((PREFIXES.parent/"natural_trace.json").read_text())
    prefixes.append([entry["action"] for entry in natural["decisions"][:161]])
    env=SubprocVecEnv([partial(environment,args.ghosts,args.danger,str(args.output/f"worker_{i}"))
                      for i in range(2)],start_method="spawn")
    try:
        env.seed(args.seed)
        if args.previous:
            model=MaskablePPO.load(args.previous/"passed.zip",env,device="cpu")
        else:
            model=MaskablePPO("MultiInputPolicy",env,policy_kwargs={"net_arch":[128,128]},
                              seed=args.seed,device="cpu",n_steps=512,batch_size=256,n_epochs=5,
                              gamma=.995,ent_coef=.02,learning_rate=3e-4)
        model.save(args.output/"initial.zip")
        curve=[]
        for steps in range(args.interval,args.budget+1,args.interval):
            model.learn(args.interval,reset_num_timesteps=False)
            model.save(args.output/f"step_{steps}.zip")
            evaluation=evaluate(model,args.ghosts,args.danger,prefixes)
            curve.append(dict(steps=steps,**evaluation))
            (args.output/"curve.json").write_text(json.dumps(curve,indent=2))
            print(dict(steps=steps,wins=evaluation["wins"],items=evaluation["average_items"],
                       scenarios=evaluation["scenario_rates"]),flush=True)
            if evaluation["gate_passed"]:
                model.save(args.output/"passed.zip")
                break
        result=dict(ghosts=args.ghosts,seed=args.seed,danger=args.danger,budget=args.budget,
                    interval=args.interval,**curve[-1])
        (args.output/"result.json").write_text(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__=="__main__":
    main()
