"""Isolated diagnostic scenarios and experimental observations; no autopilot."""
import numpy as np
from gymnasium import spaces

from rl.pacman_env import PacmanEnv


class ScenarioEnv(PacmanEnv):
    def __init__(self, prefix=(), ghosts=True, fixture=None, **kwargs):
        super().__init__(**kwargs)
        self.prefix = tuple(prefix)
        self.with_ghosts = ghosts
        if fixture not in (None, 'lemon_upper', 'lemon_lower'):
            raise ValueError('Unknown diagnostic fixture')
        self.fixture = fixture
        self._ready = False
        self.scenario_origin = None

    def _convert_observation(self, raw):
        if not raw["ghosts"]:
            padded = {**raw, "ghosts": ((0, 0),) * self.ghost_count}
            obs = super()._convert_observation(padded)
            obs["grid"][self.GHOSTS].fill(0)
            return obs
        return super()._convert_observation(raw)

    def _on_setup_state(self):
        pass

    def reset(self, **kwargs):
        self._ready = False
        super().reset(**kwargs)
        self.scenario_initial_raw = self._last_raw_observation
        self._on_setup_state()
        # Prefix consists only of actions recorded from the baseline. It sets up
        # the same visited/collected state for every controller, never guides it.
        evaluation_limit = self.max_episode_steps
        self.max_episode_steps = max(evaluation_limit, len(self.prefix)+1)
        try:
            for action in self.prefix:
                _, _, terminated, truncated, _ = self.step(action)
                if terminated or truncated:
                    raise RuntimeError("Scenario prefix terminated; invalid scenario")
        finally:
            self.max_episode_steps = evaluation_limit
        if self.fixture:
            # Synthetic TEST state on the actual map. Obtain the existing lemon
            # via the normal collector and one real decision, never grant a
            # permanent power or pick a wall/action for the evaluated policy.
            player = self._game._screen_field_mapper.get_pacman()
            player.move_to(549, 255 if self.fixture == 'lemon_upper' else 495)
            if any(player.collide(w) for w in self._game._screen_field_mapper.get_walls()):
                raise RuntimeError('Fixture overlaps a wall')
            self._last_raw_observation = self._game.observation()
            self._on_setup_state()
            _, _, terminated, _, _ = self.step(1)
            if terminated or not player.is_breaker():
                raise RuntimeError('Fixture did not acquire lemon through game rules')
        if not self.with_ghosts:
            self._game._screen_field_mapper.get_ghosts().empty()
        self._elapsed_steps = 0
        self._episode_finished = False
        self._last_raw_observation = self._game.observation()
        self.scenario_origin = self._last_raw_observation["pacman"]
        self._ready = True
        return self._convert_observation(self._last_raw_observation), self._build_info(self._last_raw_observation)


class HistoryEnv(ScenarioEnv):
    """Same physical actions/rules; add spatial visitation and measured motion."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "visits": spaces.Box(0, 1, (1, self.rows, self.cols), dtype=np.float32),
            "motion": spaces.Box(-1, 1, (8 + self.ghost_count*2,), dtype=np.float32),
        })
        self._visits = np.zeros((1, self.rows, self.cols), dtype=np.float32)
        self._motion = np.zeros(8 + self.ghost_count*2, dtype=np.float32)

    def _augment(self, obs):
        return {**obs, "visits": np.clip(np.log1p(self._visits)/np.log(101), 0, 1).astype(np.float32),
                "motion": self._motion.copy()}

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if not self.with_ghosts:
            self._motion[8:] = 0
        return self._augment(obs), info

    def _on_setup_state(self):
        self._visits.fill(0)
        self._motion.fill(0)
        row, col = self._grid_position(self._last_raw_observation["pacman"])
        self._visits[0, row, col] += 1

    def step(self, action, **kwargs):
        before = self._last_raw_observation
        obs, reward, terminated, truncated, info = super().step(action, **kwargs)
        after = self._last_raw_observation
        self._motion[:6] = self._motion[2:8].copy()
        self._motion[6:8] = np.clip((np.array(after["pacman"])-before["pacman"])/60, -1, 1)
        self._motion[8:] = (np.clip((np.array(after["ghosts"])-before["ghosts"]).reshape(-1)/60, -1, 1)
                            if after["ghosts"] else 0)
        row, col = self._grid_position(after["pacman"])
        self._visits[0, row, col] += 1
        return self._augment(obs), reward, terminated, truncated, info
