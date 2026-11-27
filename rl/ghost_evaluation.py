"""Episode metrics shared by the ghost control, validation and benchmark."""

from statistics import mean

import numpy as np


def seed_sampling(seed):
    import torch

    torch.manual_seed(seed + 1_000_000)


def play_episode(env, seed, controller, *, stochastic=False):
    if stochastic:
        seed_sampling(seed)
    observation, _ = env.reset(seed=seed)
    random_policy = np.random.default_rng([seed, 419])
    choices = np.zeros(env.action_space.n, dtype=np.int64)
    captures = []
    blocked = 0
    cumulative_reward = 0.
    while True:
        if controller == "classic":
            observation, reward, terminated, truncated, info = env.step_classic()
        else:
            mask = env.action_masks()
            if controller == "random":
                action = random_policy.choice(np.flatnonzero(mask))
            else:
                action, _ = controller.predict(observation, deterministic=not stochastic, action_masks=mask)
            index = int(action)
            assert mask[index], "A ghost policy selected a masked action."
            choices[index] += 1
            observation, reward, terminated, truncated, info = env.step(action)
        captures.extend(info["capture_events"])
        blocked += info["ghost_blocked_frames"]
        cumulative_reward += reward
        if terminated or truncated:
            break
    assert sum(map(bool, (info["victory"], info["ghost_win"], info["timeout"], info["window_closed"]))) == 1
    own_times = [ms / 1000 for index, ms in captures if index == env.slow_id]
    return dict(seed=seed, ghost_win=info["ghost_win"], pacman_win=info["victory"],
                timeout=info["timeout"], window_closed=info["window_closed"], score=info["score"],
                lives_taken=len(captures), slow_lives_taken=len(own_times),
                first_capture=captures[0][1]/1000 if captures else None,
                first_slow_capture=own_times[0] if own_times else None,
                decisions=info["elapsed_steps"], seconds=info["simulated_ms"]/1000, reward=cumulative_reward,
                action_counts=choices.tolist(), blocked_frames=blocked, capture_events=captures)


def summarize(games):
    first = [row["first_capture"] for row in games if row["first_capture"] is not None]
    slow = [row["first_slow_capture"] for row in games if row["first_slow_capture"] is not None]
    return dict(
        games=len(games),
        **{key: sum(row[key] for row in games) for key in ("ghost_win", "pacman_win", "timeout", "window_closed")},
        **{key: mean(row[key] for row in games) for key in
           ("lives_taken", "slow_lives_taken", "score", "decisions", "seconds", "reward", "blocked_frames")},
        games_with_capture=len(first), games_with_slow_capture=len(slow),
        mean_first_capture=mean(first) if first else None, mean_first_slow_capture=mean(slow) if slow else None,
        action_counts=np.sum([row["action_counts"] for row in games], axis=0).tolist(),
    )
