"""A synchronous team match with shared public state and per-ghost credit."""

from collections import Counter

import numpy as np
from gymnasium import spaces
from model.Objects.Sprites.Sprite import FastGhost, MutantGhost, SleepingGhost, SlowGhost
from stable_baselines3.common.vec_env import VecEnv

from rl.ghost_env import DIRECTIONS, WAIT, SlowGhostEnv


GHOST_TYPES = (SlowGhost, FastGhost, MutantGhost, SleepingGhost)
TEAM_SIZES = (2, 4, 7)
TEAM_REWARDS = dict(individual_life=.60, team_life=.05, team_win=.40, pacman_win=-.40)
REWARD_PROFILES = {
    "baseline": TEAM_REWARDS,
    "balanced": dict(TEAM_REWARDS, fast_individual_life=.90, team_life_controlled_only=True),
}


def ghost_rewards(slots, captures, ghost_win, pacman_win, *, config=TEAM_REWARDS, fast_slots=()):
    credit = Counter(ghost for ghost, _ in captures)
    personal = [credit[index] for index in slots]
    team_count = sum(personal) if config.get("team_life_controlled_only") else len(captures)
    shared = config["team_life"] * team_count + config["team_win"] * ghost_win + config["pacman_win"] * pacman_win
    individual = config["individual_life"]
    weights = np.asarray([config.get("fast_individual_life", individual) if index in fast_slots else individual
                          for index in slots], dtype=np.float32)
    rewards = np.asarray(personal, dtype=np.float32) * weights + shared
    return rewards, personal, shared


class GhostTeamEnv:
    """Batch axis is the controlled ghosts of one game, at one instant."""
    version = "ghost-team-v1"

    def __init__(self, opponent, count, *, reward_profile="baseline", **kwargs):
        if count not in TEAM_SIZES:
            raise ValueError(f"Unsupported ghost count {count}.")
        self.count = count
        self.reward = REWARD_PROFILES[reward_profile]
        self.match = SlowGhostEnv(opponent, **kwargs)
        self.base = self.match.base
        self.render_mode = self.match.render_mode
        if self.base.ghost_count != 7:
            raise ValueError("The team interface expects the seven-ghost arena.")
        self.action_space = self.match.action_space
        # Four map planes, per-sprite positions/state/types, motion, active slots and game status.
        cells = self.base.rows * self.base.cols
        motion = self.match.pacman.observation_space["motion"].shape[0]
        self.observation_space = spaces.Dict({
            "scene": spaces.Box(-1., 1., (4*cells + 8*(2+11+5) + motion + 7 + 6,), dtype=np.float32),
            "agent": spaces.Box(-1., 1., (7 + 2*8 + 25,), dtype=np.float32),
        })
        self.slots = ()

    def reset(self, *, seed=None):
        _, info = self.match.reset(seed=seed)
        ghosts = tuple(self.match.game._screen_field_mapper.get_ghosts())
        first_by_type = tuple(next(i for i, g in enumerate(ghosts) if type(g) is kind) for kind in GHOST_TYPES)
        self.slots = tuple(range(len(ghosts))) if self.count == 7 else first_by_type[:self.count]
        self.fast_slots = tuple(i for i in self.slots if type(ghosts[i]) is FastGhost)
        self.types = np.eye(5, dtype=np.float32)[[0, *(GHOST_TYPES.index(type(g))+1 for g in ghosts)]]
        self.active = np.zeros(7, dtype=np.float32)
        self.active[list(self.slots)] = 1.
        return self._observation(), info

    def _observation(self):
        current = self.match._pacman_observation
        positions, actors = self.match._actors_state()
        scene = np.concatenate((current["grid"][:4].ravel(), positions.ravel(), actors.ravel(), self.types.ravel(),
                                 current["motion"], self.active, current["effects"], [current["health"]/3]),
                                dtype=np.float32)
        wall_grid = np.pad(current["grid"][0], 2, constant_values=1)
        identities = np.eye(7, dtype=np.float32)
        agents = []
        for ghost_id in self.slots:
            y, x = self.base._grid_position(self.base._last_raw_observation["ghosts"][ghost_id])
            local = wall_grid[y:y+5, x:x+5].ravel()
            agents.append(np.concatenate((identities[ghost_id], (positions-positions[ghost_id+1]).ravel(), local),
                                         dtype=np.float32))
        return dict(scene=np.repeat(scene[None, :], self.count, axis=0), agent=np.stack(agents))

    def action_masks(self):
        masks = []
        for ghost_id in self.slots:
            legal = self.match.game.ghost_directions(ghost_id)
            masks.append([direction in legal for direction in DIRECTIONS] + [not legal])
        return np.asarray(masks, dtype=bool)

    def step(self, actions):
        actions = np.asarray(actions)
        if (actions.shape != (self.count,) or not np.issubdtype(actions.dtype, np.integer)
                or not self.slots or np.any((actions < 0) | (actions > WAIT))):
            raise ValueError("Supply one integer action per controlled ghost after reset.")
        if not self.action_masks()[np.arange(self.count), actions].all():
            raise ValueError("A team action failed its physical mask.")
        commands = {index: None if action == WAIT else DIRECTIONS[action]
                    for index, action in zip(self.slots, actions, strict=True)}
        return self._step(commands)

    def step_classic(self):
        return self._step(None)

    def _step(self, commands):
        terminated, truncated, info = self.match._advance_match(commands)
        rewards, credit, shared = ghost_rewards(self.slots, info["capture_events"],
            info["ghost_win"], info["victory"], config=self.reward, fast_slots=self.fast_slots)
        info.update(credit=credit, shared_reward=shared, slots=self.slots,
                    positions=self.base._last_raw_observation["ghosts"])
        return self._observation(), rewards, terminated, truncated, info

    def close(self):
        self.match.close()
        self.slots = ()


class TeamVecEnv(VecEnv):
    """Use PPO's vector slots for individual trajectories in one joint episode."""

    def __init__(self, match, first_seed):
        self.match = match
        self.next_seed = first_seed
        self.actions = None
        super().__init__(match.count, match.observation_space, match.action_space)

    def reset(self):
        self.actions = None
        observation, info = self.match.reset(seed=self.next_seed)
        self.next_seed += 1
        self.reset_infos = [dict(info) for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        self.episode_reward = np.zeros(self.num_envs, dtype=np.float32)
        return observation

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        observation, reward, terminated, truncated, info = self.match.step(self.actions)
        self.actions = None
        self.episode_reward += reward
        finished = terminated or truncated
        infos = [dict(info, **{"TimeLimit.truncated": truncated and not terminated}) for _ in range(self.num_envs)]
        if finished:
            for i, row in enumerate(infos):
                row["terminal_observation"] = {key: value[i].copy() for key, value in observation.items()}
                row["episode"] = {"r": float(self.episode_reward[i]), "l": info["elapsed_steps"]}
            observation = self.reset()
        return observation, reward, np.full(self.num_envs, finished, dtype=bool), infos

    def close(self):
        self.match.close()

    def get_attr(self, attr_name, indices=None):
        value = getattr(self.match, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        if set(self._get_indices(indices)) != set(range(self.num_envs)):
            raise ValueError("The ghost slots share one environment.")
        setattr(self.match, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        slots = self._get_indices(indices)
        result = getattr(self.match, method_name)(*args, **kwargs)
        if method_name == "action_masks":
            return [result[i] for i in slots]
        return [result for _ in slots]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
