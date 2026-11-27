"""Explicit viewer for separately trained experimental policies."""
import argparse
import torch
from sb3_contrib import MaskablePPO
from rl.navigation_lab import HistoryEnv
from rl.wrappers import NormalizePositions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--seed',type=int,default=20000)
    parser.add_argument('--stochastic',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model=MaskablePPO.load(args.model,device='cpu')
    env=NormalizePositions(HistoryEnv(render_mode='human',max_episode_steps=3000))
    try:
        obs,_=env.reset(seed=args.seed)
        while True:
            action,_=model.predict(obs,deterministic=not args.stochastic,
                                  action_masks=env.unwrapped.action_masks())
            obs,_,terminal,truncated,info=env.step(action)
            if terminal or truncated:
                print(info)
                break
    finally:
        env.close()


if __name__=='__main__':
    main()
