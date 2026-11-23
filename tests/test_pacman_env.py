import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from gymnasium.utils.env_checker import check_env

from rl.pacman_env import PacmanEnv


class PacmanEnvTest(unittest.TestCase):
    def test_environment_passes_gymnasium_checker(self):
        env = PacmanEnv(max_episode_steps=20)
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_reset_returns_a_structured_grid_observation(self):
        env = PacmanEnv(max_episode_steps=20)
        try:
            observation, info = env.reset(seed=7)

            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(
                observation["grid"].shape,
                (env.CHANNEL_COUNT, env.rows, env.cols),
            )
            self.assertEqual(int(observation["grid"][env.PACMAN].sum()), 1)
            self.assertGreater(int(observation["grid"][env.WALLS].sum()), 0)
            self.assertEqual(observation["pacman_position"].shape, (2,))
            self.assertEqual(
                observation["ghost_positions"].shape,
                (env.ghost_count, 2),
            )
            self.assertEqual(observation["health"], 3)
            self.assertEqual(info["score"], 0)
        finally:
            env.close()

    def test_step_uses_four_actions_and_reports_reward_components(self):
        env = PacmanEnv(frames_per_action=4, max_episode_steps=20)
        try:
            observation, _ = env.reset(seed=11)
            initial_grid = observation["grid"].copy()

            total_reward = 0.0
            for _ in range(5):
                observation, reward, terminated, truncated, info = env.step(1)
                total_reward += reward
                if terminated or truncated:
                    break

            self.assertFalse((initial_grid[env.PACMAN] == observation["grid"][env.PACMAN]).all())
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
            self.assertEqual(info["action"], "RIGHT")
            self.assertIsInstance(total_reward, float)
        finally:
            env.close()

    def test_time_limit_truncates_without_claiming_a_victory(self):
        env = PacmanEnv(max_episode_steps=1)
        try:
            env.reset(seed=3)
            _, _, terminated, truncated, info = env.step(1)

            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["victory"])
            self.assertEqual(info["elapsed_steps"], 1)
        finally:
            env.close()

    def test_same_seed_and_actions_reproduce_the_same_transition(self):
        def rollout():
            env = PacmanEnv(frames_per_action=4, max_episode_steps=20)
            try:
                observation, _ = env.reset(seed=19)
                rewards = []
                info = None
                for action in (1, 1, 3, 0, 2, 1):
                    observation, reward, terminated, truncated, info = env.step(action)
                    rewards.append(reward)
                    if terminated or truncated:
                        break
                return (
                    observation["pacman_position"].tolist(),
                    observation["ghost_positions"].tolist(),
                    rewards,
                    info["score"],
                    info["health"],
                )
            finally:
                env.close()

        self.assertEqual(rollout(), rollout())


if __name__ == "__main__":
    unittest.main()
