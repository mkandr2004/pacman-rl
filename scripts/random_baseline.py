"""Run a random policy to validate the Gymnasium environment end to end."""

import argparse

from rl.pacman_env import PacmanEnv


def run_episode(seed, max_steps, render_mode=None):
    env = PacmanEnv(render_mode=render_mode, max_episode_steps=max_steps)
    observation, info = env.reset(seed=seed)
    del observation

    episode_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

    env.close()
    return {
        "seed": seed,
        "steps": info["elapsed_steps"],
        "score": info["score"],
        "reward": round(episode_reward, 2),
        "victory": info["victory"],
        "game_over": info["game_over"],
        "truncated": truncated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    for seed in range(args.episodes):
        print(
            run_episode(
                seed=seed,
                max_steps=args.max_steps,
                render_mode="human" if args.render else None,
            )
        )


if __name__ == "__main__":
    main()
