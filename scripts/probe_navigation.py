"""Counterfactual tests, not an agent: probe every direction equally."""
import argparse
import json
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO
from rl.navigation_lab import ScenarioEnv
from scripts.navigation_study import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prefixes', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prefixes = json.loads(Path(args.prefixes).read_text())
    prefix = prefixes[1]
    branches = []
    for action in range(4):
        env = ScenarioEnv(prefix=prefix, ghosts=False)
        try:
            _, _ = env.reset(seed=1000)
            before = env._last_raw_observation
            mask = env.action_masks().tolist()
            trajectory = []
            for _ in range(10):
                _, reward, _, _, info = env.step(action)
                trajectory.append(dict(position=env._last_raw_observation['pacman'], reward=reward))
            branches.append(dict(action=action, mask=mask, before=before['pacman'],
                                 trajectory=trajectory))
        finally:
            env.close()
    model = MaskablePPO.load(args.model, device='cpu')
    ablations=[]
    for frames in (1,4,10,20):
        r = rollout(model,1000,prefix=prefix,ghosts=False,limit=1200//frames,
                    decision_frames=frames)
        ablations.append(dict(frames_per_decision=frames, **r))
    Path(args.output).write_text(json.dumps(dict(
        warning='Counterfactual mechanics tests only. Changed frequency is out-of-distribution for this policy.',
        branches=branches, frequency_ablations=ablations),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
