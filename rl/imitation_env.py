"""Observation contract shared by demonstration capture and autonomous BC."""

import numpy as np
from gymnasium import spaces

from rl.ghost_forecast import danger_map
from rl.navigation_lab import HistoryEnv
from rl.wrappers import NormalizePositions


class DemonstrationEnv(HistoryEnv):
    def __init__(self, danger=True, **kwargs):
        super().__init__(**kwargs)
        self.use_danger = danger
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "danger": spaces.Box(0, 1, (1, self.rows, self.cols), dtype=np.float32),
        })

    def _augment(self, obs):
        obs = super()._augment(obs)
        obs["danger"] = (danger_map(self._game, self.rows, self.cols) if self.use_danger
                         else np.zeros(self.observation_space["danger"].shape, dtype=np.float32))
        return obs


def make_imitation_env(**kwargs):
    return NormalizePositions(DemonstrationEnv(max_episode_steps=3000, **kwargs))
