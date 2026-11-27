"""Shared evaluation helpers for learned policies."""

from statistics import mean

from rl.pacman_env import PacmanEnv
from rl.wrappers import NormalizePositions


def evaluate_model(model, episodes=5, max_steps=500, render_mode=None, seed_start=1000):
    results = []
    for seed in range(seed_start, seed_start + episodes):
        env = NormalizePositions(
            PacmanEnv(
                render_mode=render_mode,
                max_episode_steps=max_steps,
            )
        )
        observation, info = env.reset(seed=seed)
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        results.append(
            {
                "seed": seed,
                "score": info["score"],
                "health": info["health"],
                "reward": round(episode_reward, 2),
                "steps": info["elapsed_steps"],
                "victory": info["victory"],
                "game_over": info["game_over"],
            }
        )
        env.close()

    summary = {
        "episodes": episodes,
        "mean_score": round(mean(item["score"] for item in results), 2),
        "mean_reward": round(mean(item["reward"] for item in results), 2),
        "victories": sum(item["victory"] for item in results),
        "game_overs": sum(item["game_over"] for item in results),
    }
    return summary, results
