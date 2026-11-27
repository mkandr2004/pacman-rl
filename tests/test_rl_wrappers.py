import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from rl.pacman_env import PacmanEnv
from rl.wrappers import NormalizePositions


class NormalizePositionsTest(unittest.TestCase):
    def test_positions_are_scaled_for_neural_networks(self):
        env = NormalizePositions(PacmanEnv(max_episode_steps=5))
        try:
            observation, _ = env.reset(seed=2)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertGreaterEqual(float(observation["pacman_position"].min()), 0.0)
            self.assertLessEqual(float(observation["pacman_position"].max()), 1.0)
            self.assertGreaterEqual(float(observation["ghost_positions"].min()), 0.0)
            self.assertLessEqual(float(observation["ghost_positions"].max()), 1.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
