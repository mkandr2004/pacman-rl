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
    ENVIRONMENT_VERSION = "safe-borders-v4"
    REWARD_CONFIG = dict(point=3.0, coin=5.0, victory=500.0, health=-25.0,
                         game_over=-100.0, step=-0.01, blocked=-0.05, wall=0.0, fruit=0.0)

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
                    low=np.zeros(self.ghost_count * 2, dtype=np.int32),
                    high=np.tile(max_position, self.ghost_count),
                    dtype=np.int32,
                ),
                "health": spaces.Discrete(consts.HEALTH + 1),
                "effects": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
                "bonus_types": spaces.MultiBinary((4, self.rows, self.cols)),
            }
        )

        self._game = None
        self._last_raw_observation = None
        self._elapsed_steps = 0
        self._episode_finished = False
        self._clock = pygame.time.Clock()
        self._random_state = random.Random().getstate()
        self._owns_dummy_driver = False

    def _configure_headless_display(self):
        if self.render_mode is None:
            if "SDL_VIDEODRIVER" not in os.environ:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                self._owns_dummy_driver = True
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    @staticmethod
    def _empty_reward_terms():
        return {
            "step": 0.0,
            "score": 0.0,
            "health": 0.0,
            "victory": 0.0,
            "game_over": 0.0,
            "blocked": 0.0,
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
        bonus_types = np.zeros((4, self.rows, self.cols), dtype=np.int8)
        for channel, positions in enumerate(raw_observation["bonus_positions"]):
            self._mark_positions(bonus_types, channel, positions)
        return {
            "grid": grid,
            "effects": np.asarray(raw_observation["effects"], dtype=np.float32),
            "bonus_types": bonus_types,
            "pacman_position": np.asarray(
                raw_observation["pacman"], dtype=np.int32
            ),
            "ghost_positions": np.asarray(
                raw_observation["ghosts"], dtype=np.int32
            ).reshape(self.ghost_count * 2),
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
            self._random_state = random.Random(seed).getstate()
            self.action_space.seed(seed)

        if self._game is not None:
            pygame.quit()
        self._configure_headless_display()
        self._game = create_game()
        self._game.use_simulation_clock()
        self._elapsed_steps = 0
        self._episode_finished = False
        self._last_raw_observation = self._game.observation()

        observation = self._convert_observation(self._last_raw_observation)
        info = self._build_info(
            self._last_raw_observation,
            reward_terms=self._empty_reward_terms(),
        )
        return observation, info

    def step(self, action, *, ghost_actions=None, track_ghosts=False, keyboard=False):
        if self._game is None:
            raise ResetNeeded("Call reset() before step().")
        if self._episode_finished:
            raise ResetNeeded("Episode finished; call reset() before another step().")
        if keyboard and action is None and self.render_mode == "human":
            direction = None
        elif self.action_space.contains(action):
            direction = self._ACTION_TO_DIRECTION[int(action)]
        else:
            raise ValueError(f"Invalid action {action!r}; expected an integer from 0 to 3")

        before = self._last_raw_observation
        previous_frame = before
        lost_health = 0
        if track_ghosts:
            captures = []
            movement = dict.fromkeys(ghost_actions or (), 0)
            active_frames = 0
            ghosts = tuple(self._game._screen_field_mapper.get_ghosts())
        terminal_info = {
            "victory": False,
            "game_over": False,
            "window_closed": False,
        }

        for _ in range(self.frames_per_action):
            caller_random_state = random.getstate()
            random.setstate(self._random_state)
            try:
                raw_observation, terminated, terminal_info = self._game.advance(
                    action=direction,
                    render=self.render_mode == "human",
                    ghost_actions=ghost_actions,
                )
                self._random_state = random.getstate()
            finally:
                random.setstate(caller_random_state)
            lost_health += max(
                0,
                previous_frame["health"] - raw_observation["health"],
            )
            if track_ghosts and not terminal_info["window_closed"]:
                active_frames += 1
                if self._game.last_capture is not None:
                    captures.append((ghosts.index(self._game.last_capture), self._game._simulation_ms))
                for index in movement:
                    movement[index] += int(raw_observation["ghosts"][index] != previous_frame["ghosts"][index])
            previous_frame = raw_observation
            if self.render_mode == "human":
                self._clock.tick(self.metadata["render_fps"])
            if terminated or terminal_info["window_closed"]:
                break

        self._elapsed_steps += 1
        point_reward = self.REWARD_CONFIG["point"] * (before["points_remaining"] - raw_observation["points_remaining"])
        coin_reward = self.REWARD_CONFIG["coin"] * (before["coins_remaining"] - raw_observation["coins_remaining"])
        reward_terms = {
            "step": self.REWARD_CONFIG["step"],
            "score": point_reward + coin_reward,
            "health": self.REWARD_CONFIG["health"] * lost_health,
            "victory": self.REWARD_CONFIG["victory"] if terminal_info["victory"] else 0.0,
            "game_over": self.REWARD_CONFIG["game_over"] if terminal_info["game_over"] else 0.0,
            "blocked": self.REWARD_CONFIG["blocked"] if before["pacman"] == raw_observation["pacman"] else 0.0,
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
            action=direction.name if direction is not None else "KEYBOARD",
            reward_terms=reward_terms,
            victory=terminal_info["victory"],
            game_over=terminal_info["game_over"],
            window_closed=terminal_info["window_closed"],
            termination_reason=("victory" if terminal_info["victory"] else
                "game_over" if terminal_info["game_over"] else
                "window_closed" if terminal_info["window_closed"] else
                "timeout" if truncated else None),
        )
        if track_ghosts:
            info.update(capture_events=captures, ghost_moved_frames=movement, active_frames=active_frames)
        return observation, reward, terminated, truncated, info

    def action_masks(self):
        """Physical validity only: never consult rewards or future ghost moves."""
        if self._game is None:
            raise ResetNeeded("Call reset() before action_masks().")
        mapper = self._game._screen_field_mapper
        pacman = mapper.get_pacman()
        width, height = self._last_raw_observation["screen_size"]
        bounds = pygame.Rect(0, 0, width, height)
        deltas = ((-pacman.speed_horizontal, 0), (pacman.speed_horizontal, 0),
                  (0, -pacman.speed_vertical), (0, pacman.speed_vertical))
        mask = []
        for dx, dy in deltas:
            candidate = pacman.get_rect().move(dx, dy)
            collisions = [wall for wall in mapper.get_walls()
                          if candidate.colliderect(wall.get_rect())]
            valid = bounds.contains(candidate) and not any(
                self._game._controller.is_border_wall(wall) for wall in collisions
            ) and (pacman.is_breaker() or not collisions)
            mask.append(valid)
        # A trapped state still needs a defined distribution; normal collision
        # rules leave the player stationary while time and ghosts advance.
        return np.asarray(mask if any(mask) else [True] * 4, dtype=bool)

    def render(self):
        return None

    def close(self):
        if self._owns_dummy_driver:
            os.environ.pop("SDL_VIDEODRIVER", None)
            self._owns_dummy_driver = False
        if self._game is not None:
            pygame.quit()
            self._game = None
            self._last_raw_observation = None
            self._episode_finished = False
