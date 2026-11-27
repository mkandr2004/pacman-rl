import math
import os
import unittest
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from rl.pacman_env import PacmanEnv
from rl.shaping import ProgressPotential, PotentialShaping


class ProgressShapingTest(unittest.TestCase):
    def test_shaping_preserves_actions_observations_and_physical_transitions(self):
        def rollout(shaped):
            env = PacmanEnv(max_episode_steps=20)
            if shaped:
                env = PotentialShaping(env)
            try:
                obs, _ = env.reset(seed=4)
                for action in [1, 1, 3, 0, 2, 1]:
                    obs, _, _, _, _ = env.step(action)
                return obs
            finally:
                env.close()
        original, shaped = rollout(False), rollout(True)
        self.assertEqual(original.keys(), shaped.keys())
        for key in original:
            np.testing.assert_array_equal(original[key], shaped[key])

    def raw(self, x):
        return dict(screen_size=(300, 180), pacman=(x, 60), effects=(0, 0, 0, .5, .5),
            points=((180, 60),), coins=(), walls=(),
            bonus_positions=((), (), (), ()))

    def test_progress_is_visible_inside_a_cell(self):
        potential = ProgressPotential()
        self.assertGreater(potential(self.raw(72)), potential(self.raw(60)))

    def test_closed_loop_does_not_generate_discounted_shaping_profit(self):
        potential = ProgressPotential()
        a, b = potential(self.raw(60)), potential(self.raw(72))
        gamma = .995
        discounted_shaping = (gamma*b-a) + gamma*(gamma*a-b)
        self.assertAlmostEqual(discounted_shaping, (gamma**2-1)*a)
        self.assertLessEqual(discounted_shaping, 0)

    def test_unreachable_collectibles_have_finite_potential(self):
        raw = self.raw(60)
        raw["walls"] = ((120, 60),)
        potential = ProgressPotential()
        self.assertTrue(math.isfinite(potential(raw)))
        raw["effects"] = (0, 0, 1, .1667, .1667)
        self.assertGreater(potential(raw), 0)

    def test_timeout_preserves_bootstrap_and_observation_contract(self):
        env = PotentialShaping(PacmanEnv(max_episode_steps=2))
        try:
            obs, _ = env.reset(seed=8)
            self.assertTrue(env.observation_space.contains(obs))
            initial = env._previous
            shaping_sum = 0.0
            for index in range(2):
                obs, reward, terminal, truncated, info = env.step(1)
                shaping_sum += env.gamma**index * info["reward_terms"]["potential"]
                self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
            self.assertFalse(terminal)
            self.assertTrue(truncated)
            self.assertGreater(info["potential_after"], 0)
            self.assertAlmostEqual(shaping_sum, -initial + env.gamma**2 * info["potential_after"])
        finally:
            env.close()

    def test_true_termination_uses_zero_final_potential(self):
        env = PotentialShaping(PacmanEnv(max_episode_steps=10))
        try:
            env.reset(seed=9)
            field = env.unwrapped._game._screen_field_mapper.get_field()
            field._points_found = field._total_points
            field._coins_found = field._total_coins
            _, _, terminal, _, info = env.step(1)
            self.assertTrue(terminal)
            self.assertEqual(info["potential_after"], 0)
        finally:
            env.close()
