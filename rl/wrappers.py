"""Observation wrappers used by learning algorithms."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizePositions(gym.ObservationWrapper):
    """Scale pixel coordinates to [0, 1] without changing grid channels."""

    POSITION_KEYS = ("pacman_position", "ghost_positions")

    def __init__(self, env):
        super().__init__(env)
        original_spaces = dict(env.observation_space.spaces)
        self._position_high = {}

        for key in self.POSITION_KEYS:
            original_space = original_spaces[key]
            self._position_high[key] = original_space.high.astype(np.float32)
            original_spaces[key] = spaces.Box(
                low=np.zeros(original_space.shape, dtype=np.float32),
                high=np.ones(original_space.shape, dtype=np.float32),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(original_spaces)

    def observation(self, observation):
        normalized = dict(observation)
        for key in self.POSITION_KEYS:
            high = np.maximum(self._position_high[key], 1.0)
            normalized[key] = (
                np.asarray(observation[key], dtype=np.float32) / high
            ).astype(np.float32)
        return normalized
