import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from rl.pacman_env import PacmanEnv


class AgentStateTest(unittest.TestCase):
    def setUp(self):
        self.env = PacmanEnv(max_episode_steps=30)
        self.env.reset(seed=11)
        self.player = self.env._game._screen_field_mapper.get_pacman()

    def tearDown(self):
        self.env.close()

    def test_repeated_and_overlapping_bonuses_restore_base_speed(self):
        self.player.make_breaker()
        self.player.make_breaker()
        self.assertGreater(self.player.speed_horizontal, 0)
        self.player.make_speed()
        self.player.make_speed()
        self.assertEqual(self.player.speed_horizontal, 3)
        self.player.check_breaker(3000)
        self.assertEqual(self.player.speed_horizontal, 6)
        self.player.check_speed(6000)
        self.assertEqual(self.player.speed_horizontal, 3)

    def test_wall_mask_allows_breaker_and_state_exposes_remaining_duration(self):
        self.assertFalse(self.env.action_masks()[0])
        self.assertTrue(self.env.action_masks()[1])
        self.player.make_breaker()
        self.assertFalse(self.env.action_masks()[0])
        observation, _, _, _, _ = self.env.step(1)
        self.assertGreater(observation["effects"][2], 0)
        self.assertTrue(self.env.observation_space.contains(observation))
