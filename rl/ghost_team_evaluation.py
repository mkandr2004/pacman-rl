"""Compare ghost teams using games, rather than individual steps, as replicates."""

from collections import Counter
from statistics import mean

import numpy as np

from rl.ghost_evaluation import seed_sampling


def play_team(env, seed, controller):
    seed_sampling(seed)
    rng = np.random.default_rng([seed, 9231])
    observation, _ = env.reset(seed=seed)
    slots = env.slots
    moves, reversals, blocked = np.zeros((3, len(slots)), dtype=np.int32)
    actions_count = np.zeros((len(slots), env.action_space.n), dtype=np.int32)
    previous = np.array(env.base._last_raw_observation["ghosts"])[list(slots)]
    two_steps_back = None
    captures = []
    returns = np.zeros(len(slots), dtype=np.float64)
    while True:
        if controller == "classic":
            observation, reward, terminated, truncated, info = env.step_classic()
        else:
            masks = env.action_masks()
            if controller == "random":
                actions = np.array([rng.choice(np.flatnonzero(mask)) for mask in masks])
            else:
                actions, _ = controller.predict(observation, deterministic=False, action_masks=masks)
            actions_count[np.arange(len(slots)), actions] += 1
            observation, reward, terminated, truncated, info = env.step(actions)
            for i, index in enumerate(slots):
                if actions[i] < 4:
                    blocked[i] += info["active_frames"]-info["ghost_moved_frames"][index]
        returns += reward
        captures.extend(info["capture_events"])
        position = np.asarray(info["positions"])[list(slots)]
        moved = np.any(previous != position, axis=1)
        moves += moved
        if two_steps_back is not None:
            reversals += moved & np.all(two_steps_back == position, axis=1)
        two_steps_back, previous = previous, position
        if terminated or truncated:
            break
    assert sum(bool(info[key]) for key in ("ghost_win","victory","timeout","window_closed")) == 1
    credit = Counter(index for index, _ in captures)
    steps = info["elapsed_steps"]
    return dict(seed=seed, ghost_win=info["ghost_win"], pacman_win=info["victory"],
                timeout=info["timeout"], closed=info["window_closed"], score=info["score"],
                lives=len(captures), controlled_lives=sum(credit[index] for index in slots),
                slots=list(slots), per_agent=[credit[index] for index in slots],
                first_capture=captures[0][1]/1000 if captures else None,
                first_controlled=next((ms/1000 for index, ms in captures if index in slots), None),
                seconds=info["simulated_ms"]/1000, decisions=steps, moves=(moves/steps).tolist(),
                reversals=(reversals/max(steps-1,1)).tolist(), blocked_frames=blocked.tolist(),
                reward=returns.tolist(), action_counts=actions_count.tolist(), capture_events=captures)


def summarize(rows):
    out = {k: sum(row[k] for row in rows) for k in ("ghost_win", "pacman_win", "timeout", "closed")}
    out.update({k: mean(row[k] for row in rows) for k in ("lives", "controlled_lives", "score", "seconds")})
    for key in ("first_capture","first_controlled"):
        sample = [row[key] for row in rows if row[key] is not None]
        out[key] = mean(sample) if sample else None
        out[key+"_n"] = len(sample)
    out.update({k: np.asarray([row[k] for row in rows]).mean(axis=0).tolist()
                for k in ("per_agent", "moves", "reversals", "blocked_frames", "reward")})
    return dict(games=len(rows), slots=rows[0]["slots"], **out)
