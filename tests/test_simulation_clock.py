import os
import random
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from rl.pacman_env import PacmanEnv


class SimulationClockTest(unittest.TestCase):
    def test_invincibility_expires_by_frames_even_when_wall_clock_changes(self):
        env = PacmanEnv(frames_per_action=1, max_episode_steps=300)
        try:
            env.reset(seed=10)
            pacman = env._game._screen_field_mapper.get_pacman()
            pacman.make_invinsible()
            with patch("pygame.time.get_ticks", return_value=999999):
                for _ in range(199):
                    env.step(0)
                self.assertTrue(pacman.is_invinsible())
                env.step(0)
                self.assertFalse(pacman.is_invinsible())
        finally:
            env.close()

    def test_environment_does_not_change_the_learners_random_state(self):
        env = PacmanEnv(max_episode_steps=5)
        try:
            state = random.getstate()
            env.reset(seed=20)
            env.step(1)
            self.assertEqual(state, random.getstate())
        finally:
            env.close()

    def test_rendering_does_not_change_transitions(self):
        def rollout(render):
            env = PacmanEnv(frames_per_action=1, max_episode_steps=230)
            try:
                env.reset(seed=9)
                env.render_mode = "human" if render else None
                env._clock = Mock()
                # Exercise the same draw path without real-time throttling.
                for index in range(220):
                    _, _, terminal, _, _ = env.step(index % 4)
                    if terminal:
                        break
                return env._last_raw_observation
            finally:
                env.close()
        self.assertEqual(rollout(False), rollout(True))
