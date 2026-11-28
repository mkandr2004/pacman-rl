"""Train or play the slow ghost against one fixed Pac-Man policy."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

import consts
from enums import Direction
from model.Objects.Sprites.Sprite import SlowGhost
from rl.imitation_env import make_imitation_env
from rl.pacman_env import PacmanEnv


DIRECTIONS = (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)
WAIT = len(DIRECTIONS)
REWARDS = {"team_life": .10, "slow_life": .30, "ghost_win": 1., "pacman_win": -1.}


class SlowGhostEnv(gym.Env):
    metadata = PacmanEnv.metadata
    version = "slow-ghost-v1"

    def __init__(self, opponent, *, render_mode=None, max_episode_steps=3000, keyboard=False,
                 track_classic=False):
        if keyboard != (opponent is None) or (keyboard and render_mode != "human"):
            raise ValueError("Use a frozen model or the keyboard in a rendered match.")
        self.opponent = opponent
        self.keyboard = keyboard
        self.track_classic = track_classic
        self.render_mode = render_mode
        self.pacman = make_imitation_env(danger=False, render_mode=render_mode)
        self.base = self.pacman.unwrapped
        self.base.max_episode_steps = max_episode_steps
        if max_episode_steps < 1:
            raise ValueError("The match needs at least one decision.")
        self.action_space = spaces.Discrete(WAIT + 1)
        self.public_keys = ("grid", "bonus_types", "health", "effects",
                            "pacman_position", "ghost_positions")
        count = 1 + self.base.ghost_count
        self.observation_space = spaces.Dict({
            **{key: self.pacman.observation_space[key] for key in self.public_keys},
            "relative": spaces.Box(-1., 1., (count, 2), dtype=np.float32),
            "actors": spaces.Box(0., 1., (count, 11), dtype=np.float32),
        })
        self.slow_id = None
        self._pacman_observation = None

    @property
    def game(self):
        return self.base._game

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pacman_observation, info = self.pacman.reset(seed=seed, options=options)
        ghosts = self.game._screen_field_mapper.get_ghosts()
        self.slow_id, = (i for i, ghost in enumerate(ghosts) if type(ghost) is SlowGhost)
        return self._observation(), info

    def _actors_state(self):
        current = self._pacman_observation
        mapper = self.game._screen_field_mapper
        sprites = (mapper.get_pacman(), *mapper.get_ghosts())
        actors = np.zeros((len(sprites), 11), dtype=np.float32)
        for i, sprite in enumerate(sprites):
            if sprite.moving_direction in DIRECTIONS:
                actors[i, DIRECTIONS.index(sprite.moving_direction)] = 1.
            actors[i, 4:] = (
                sprite.speed_horizontal / (2 * consts.PACMAN_SPEED),
                sprite.speed_vertical / (2 * consts.PACMAN_SPEED),
                sprite.get_width() / consts.BLOCK_WIDTH, sprite.get_height() / consts.BLOCK_HEIGHT,
                getattr(sprite, "_instruction_index", 0) / len(getattr(sprite, "_instructions", None) or (0,)),
                min(sprite._passed_horizontal / consts.BLOCK_WIDTH, 1.),
                min(sprite._passed_vertical / consts.BLOCK_HEIGHT, 1.),
            )
        positions = np.vstack((current["pacman_position"], current["ghost_positions"].reshape(-1, 2)))
        return positions, actors

    def _observation(self):
        current = self._pacman_observation
        positions, actors = self._actors_state()
        return {**{key: current[key] for key in self.public_keys},
                "relative": positions - positions[self.slow_id + 1], "actors": actors}

    def action_masks(self):
        if self.slow_id is None:
            raise ResetNeeded("Reset the match before reading masks.")
        legal = self.game.ghost_directions(self.slow_id)
        return np.array([direction in legal for direction in DIRECTIONS] + [not legal], dtype=bool)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid slow ghost action: {action!r}")
        direction = None if int(action) == WAIT else DIRECTIONS[int(action)]
        return self._advance({self.slow_id: direction})

    def step_classic(self):
        return self._advance(None)

    def _advance(self, ghost_actions):
        terminated, truncated, info = self._advance_match(ghost_actions)
        captures = info["capture_events"]
        own_captures = sum(index == self.slow_id for index, _ in captures)
        terms = dict(team_life=REWARDS["team_life"] * len(captures),
                     slow_life=REWARDS["slow_life"] * own_captures,
                     terminal=REWARDS["ghost_win"] if info["ghost_win"] else REWARDS["pacman_win"] if info["victory"] else 0.)
        moved = info["ghost_moved_frames"].get(self.slow_id, 0)
        tried = bool(ghost_actions and ghost_actions[self.slow_id] is not None)
        info.update(slow_captures=own_captures, ghost_reward_terms=terms,
                    ghost_blocked_frames=info["active_frames"] - moved if tried else 0)
        return self._observation(), sum(terms.values()), terminated, truncated, info

    def _advance_match(self, ghost_actions):
        if self.slow_id is None:
            raise ResetNeeded("Reset the match before stepping.")
        # Both policies see the state before any move, including the Pac-Man history
        # expected by the protected checkpoints. The old frame order is preserved.
        action = None
        if not self.keyboard:
            action, _ = self.opponent.predict(self._pacman_observation,
                deterministic=True, action_masks=self.base.action_masks())
        observation, _, terminated, truncated, info = self.base.step(
            action, ghost_actions=ghost_actions, track_ghosts=True,
            track_classic=self.track_classic, keyboard=self.keyboard)
        self._pacman_observation = self.pacman.observation(observation)
        ghost_win = info["game_over"] and not info["victory"] and not info["window_closed"]
        truncated = bool(truncated and not terminated)
        info.update(ghost_win=ghost_win, timeout=truncated and not info["window_closed"],
                    simulated_ms=self.game._simulation_ms)
        return terminated, truncated, info

    def close(self):
        self.pacman.close()
        self.slow_id = None
        self._pacman_observation = None
