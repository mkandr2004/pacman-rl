"""Gymnasium adapter for controlling Pac-Man in the visual game engine."""

import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.error import ResetNeeded

import consts
from enums import Direction
from main.run import create_game


class PacmanEnv(gym.Env):
    """Single-agent environment: learned Pac-Man versus scripted ghosts."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    WALLS = 0
    COINS = 1
    POINTS = 2
    BONUSES = 3
    PACMAN = 4
    GHOSTS = 5
    CHANNEL_COUNT = 6

    _ACTION_TO_DIRECTION = {
        0: Direction.LEFT,
        1: Direction.RIGHT,
        2: Direction.UP,
        3: Direction.DOWN,
    }

    def __init__(
        self,
        render_mode=None,
        frames_per_action=4,
        max_episode_steps=2_000,
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if frames_per_action < 1:
            raise ValueError("frames_per_action must be at least 1")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        self.render_mode = render_mode
        self.frames_per_action = frames_per_action
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Discrete(len(self._ACTION_TO_DIRECTION))

        field_path = Path(__file__).resolve().parents[1] / consts.FIELD_NAME
        with field_path.open(encoding="utf-8") as field_file:
            self.rows, self.cols = map(int, field_file.readline().split())
            field_lines = [field_file.readline().rstrip("\n") for _ in range(self.rows)]
        ghost_symbols = {"G", "F", "S", "Z", "M"}
        self.ghost_count = sum(
            cell in ghost_symbols
            for line in field_lines
            for cell in line
        )
        max_position = np.array(
            [
                self.cols * consts.BLOCK_WIDTH - 1,
                self.rows * consts.BLOCK_HEIGHT - 1,
            ],
            dtype=np.int32,
        )

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.MultiBinary(
                    (self.CHANNEL_COUNT, self.rows, self.cols)
                ),
                "pacman_position": spaces.Box(
                    low=np.zeros(2, dtype=np.int32),
                    high=max_position,
                    dtype=np.int32,
                ),
                "ghost_positions": spaces.Box(
                    low=np.zeros((self.ghost_count, 2), dtype=np.int32),
                    high=np.tile(max_position, (self.ghost_count, 1)),
                    dtype=np.int32,
                ),
                "health": spaces.Discrete(consts.HEALTH + 1),
            }
        )

        self._game = None
        self._last_raw_observation = None
        self._elapsed_steps = 0
        self._episode_finished = False
        self._clock = pygame.time.Clock()

    def _configure_headless_display(self):
        if self.render_mode is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    @staticmethod
    def _empty_reward_terms():
        return {
            "step": 0.0,
            "score": 0.0,
            "health": 0.0,
            "victory": 0.0,
            "game_over": 0.0,
        }

    def _grid_position(self, position):
        x, y = position
        row = max(0, min(self.rows - 1, int(y // consts.BLOCK_HEIGHT)))
        col = max(0, min(self.cols - 1, int(x // consts.BLOCK_WIDTH)))
        return row, col

    def _mark_positions(self, grid, channel, positions):
        for position in positions:
            row, col = self._grid_position(position)
            grid[channel, row, col] = 1

    def _convert_observation(self, raw_observation):
        grid = np.zeros(
            (self.CHANNEL_COUNT, self.rows, self.cols),
            dtype=np.int8,
        )
        self._mark_positions(grid, self.WALLS, raw_observation["walls"])
        self._mark_positions(grid, self.COINS, raw_observation["coins"])
        self._mark_positions(grid, self.POINTS, raw_observation["points"])
        self._mark_positions(grid, self.BONUSES, raw_observation["bonuses"])
        self._mark_positions(grid, self.PACMAN, (raw_observation["pacman"],))
        self._mark_positions(grid, self.GHOSTS, raw_observation["ghosts"])
        return {
            "grid": grid,
            "pacman_position": np.asarray(
                raw_observation["pacman"], dtype=np.int32
            ),
            "ghost_positions": np.asarray(
                raw_observation["ghosts"], dtype=np.int32
            ).reshape(self.ghost_count, 2),
            "health": int(raw_observation["health"]),
        }

    def _build_info(self, raw_observation, **extra):
        info = {
            "score": raw_observation["score"],
            "health": raw_observation["health"],
            "coins_remaining": raw_observation["coins_remaining"],
            "points_remaining": raw_observation["points_remaining"],
            "elapsed_steps": self._elapsed_steps,
        }
        info.update(extra)
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options

        if seed is not None:
            random.seed(seed)
            self.action_space.seed(seed)

        if self._game is not None:
            pygame.quit()
        self._configure_headless_display()
        self._game = create_game()
        self._elapsed_steps = 0
        self._episode_finished = False
        self._last_raw_observation = self._game.observation()

        observation = self._convert_observation(self._last_raw_observation)
        info = self._build_info(
            self._last_raw_observation,
            reward_terms=self._empty_reward_terms(),
        )
        return observation, info

    def step(self, action):
        if self._game is None:
            raise ResetNeeded("Call reset() before step().")
        if self._episode_finished:
            raise ResetNeeded("Episode finished; call reset() before another step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected an integer from 0 to 3")

        direction = self._ACTION_TO_DIRECTION[int(action)]
        before = self._last_raw_observation
        previous_frame = before
        lost_health = 0
        terminal_info = {
            "victory": False,
            "game_over": False,
            "window_closed": False,
        }

        for _ in range(self.frames_per_action):
            raw_observation, terminated, terminal_info = self._game.advance(
                action=direction,
                render=self.render_mode == "human",
            )
            lost_health += max(
                0,
                previous_frame["health"] - raw_observation["health"],
            )
            previous_frame = raw_observation
            if self.render_mode == "human":
                self._clock.tick(self.metadata["render_fps"])
            if terminated or terminal_info["window_closed"]:
                break

        self._elapsed_steps += 1
        score_reward = float(raw_observation["score"] - before["score"])
        reward_terms = {
            "step": -0.01,
            "score": score_reward,
            "health": -25.0 * lost_health,
            "victory": 100.0 if terminal_info["victory"] else 0.0,
            "game_over": -100.0 if terminal_info["game_over"] else 0.0,
        }
        reward = float(sum(reward_terms.values()))

        terminated = bool(
            terminal_info["victory"] or terminal_info["game_over"]
        )
        truncated = bool(
            terminal_info["window_closed"]
            or self._elapsed_steps >= self.max_episode_steps
        )
        self._episode_finished = terminated or truncated
        self._last_raw_observation = raw_observation

        observation = self._convert_observation(raw_observation)
        info = self._build_info(
            raw_observation,
            action=direction.name,
            reward_terms=reward_terms,
            victory=terminal_info["victory"],
            game_over=terminal_info["game_over"],
            window_closed=terminal_info["window_closed"],
        )
        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        if self._game is not None:
            pygame.quit()
            self._game = None
            self._last_raw_observation = None
            self._episode_finished = False
