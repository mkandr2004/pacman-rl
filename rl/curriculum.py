"""Experimental ghost counts with a fixed observation contract."""

import numpy as np
from gymnasium import spaces

from rl.pacman_env import PacmanEnv
from rl.ghost_forecast import danger_map


STAGES = (0, 1, 2, 4, 7)


class CurriculumEnv(PacmanEnv):
    def __init__(self, active_ghosts, danger_channel=False, **kwargs):
        super().__init__(**kwargs)
        if active_ghosts not in STAGES:
            raise ValueError(f"Ghost count must be one of {STAGES}")
        self.active_ghosts = active_ghosts
        self.danger_channel = danger_channel
        self._ghost_slots = None
        if danger_channel:
            self.observation_space = spaces.Dict({
                **self.observation_space.spaces,
                "danger": spaces.Box(0,1,shape=(1,self.rows,self.cols),dtype=np.float32),
            })

    def reset(self, **kwargs):
        self._ghost_slots = None
        _, info = super().reset(**kwargs)
        group = self._game._screen_field_mapper.get_ghosts()
        original = list(group)
        if self.active_ghosts == self.ghost_count:
            slots = list(range(self.ghost_count))
        else:
            slots = sorted(self.np_random.choice(self.ghost_count,self.active_ghosts,replace=False).tolist())
            group.remove(*(ghost for i,ghost in enumerate(original) if i not in slots))
        self._ghost_slots = slots
        raw = self._last_raw_observation = self._game.observation()
        return self._convert_observation(raw), info

    def _convert_observation(self, raw):
        if self._ghost_slots is None:
            obs = super()._convert_observation(raw)
        else:
            padded = [(0,0)]*self.ghost_count
            for slot, position in zip(self._ghost_slots,raw["ghosts"]):
                padded[slot] = position
            obs = super()._convert_observation({**raw,"ghosts":padded})
            # The upper-left border is not a ghost when unused slots are zero.
            obs["grid"][self.GHOSTS,0,0] = 0
        if self.danger_channel:
            obs["danger"] = danger_map(self._game,self.rows,self.cols)
        return obs
