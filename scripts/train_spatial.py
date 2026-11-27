"""Train isolated experiments. Never promotes or overwrites an existing model."""
import argparse
import json
from functools import partial
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl.navigation_lab import HistoryEnv
from rl.pacman_env import PacmanEnv
from rl.spatial_policy import SpatialHistoryExtractor
from rl.wrappers import NormalizePositions


def make_env(logfile=None):
    torch.set_num_threads(1)
    return Monitor(NormalizePositions(HistoryEnv(max_episode_steps=3000)), filename=logfile,
                   info_keywords=('score', 'victory', 'game_over'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--timesteps', type=int, default=131072)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--encoder', choices=('cnn', 'mlp'), default='cnn')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = dict(algorithm='MaskablePPO', encoder=args.encoder, seed=args.seed,
                  timesteps=args.timesteps, workers=args.workers,
                  observation='unchanged base + visit counts + past 4 displacements + ghost velocity',
                  environment=PacmanEnv.ENVIRONMENT_VERSION, frames_per_action=4,
                  rewards=PacmanEnv.REWARD_CONFIG, shaping=False, curriculum=False,
                  selection='final at predeclared budget; test seeds not used for selection')
    (root/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    env = SubprocVecEnv([partial(make_env, str(root/f'worker_{i}')) for i in range(args.workers)],
                        start_method='spawn')
    env.seed(args.seed)
    kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    if args.encoder == 'cnn':
        kwargs['features_extractor_class'] = SpatialHistoryExtractor
    model = MaskablePPO('MultiInputPolicy', env, policy_kwargs=kwargs, seed=args.seed,
                        learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
                        gamma=.995, gae_lambda=.95, ent_coef=.02, clip_range=.2,
                        device='cpu', verbose=1)
    config['parameters'] = sum(p.numel() for p in model.policy.parameters())
    (root/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    try:
        model.save(root/'initial_model')
        model.learn(args.timesteps, log_interval=8,
                    callback=CheckpointCallback(save_freq=max(1,32768//args.workers),
                        save_path=str(root/'checkpoints'), name_prefix='candidate'))
        model.save(root/'final_model')
    finally:
        env.close()


if __name__ == '__main__':
    main()
