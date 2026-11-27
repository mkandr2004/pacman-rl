import os
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from rl.pacman_env import PacmanEnv
from rl.policies import HeuristicPolicy


class HeuristicPolicyTest(unittest.TestCase):
    def _observation(self):
        return {
            "grid": np.zeros((6, 3, 3), dtype=np.int8),
            "pacman_position": np.array([60, 60], dtype=np.int32),
            "ghost_positions": np.empty(0, dtype=np.int32),
            "health": 3,
        }

    def test_policy_takes_the_shortest_path_to_a_point(self):
        observation = self._observation()
        observation["grid"][PacmanEnv.POINTS, 1, 2] = 1

        self.assertEqual(HeuristicPolicy().predict(observation), 1)

    def test_policy_routes_around_a_wall(self):
        observation = self._observation()
        observation["grid"][PacmanEnv.WALLS, 1, 2] = 1
        observation["grid"][PacmanEnv.POINTS, 2, 2] = 1

        self.assertEqual(HeuristicPolicy().predict(observation), 3)

    def test_policy_controls_the_real_environment(self):
        env = PacmanEnv(max_episode_steps=20)
        policy = HeuristicPolicy()
        try:
            observation, _ = env.reset(seed=5)
            for _ in range(20):
                action = policy.predict(observation)
                self.assertTrue(env.action_space.contains(action))
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
