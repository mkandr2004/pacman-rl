import unittest

import numpy as np
import torch

from rl.interaction_dqn import UniformReplay, bellman_targets


class InteractionDQNTests(unittest.TestCase):
    def test_bellman_uses_only_legal_actions_and_no_terminal_bootstrap(self):
        values = torch.tensor([[19., -4., -2., 100.], [500., 500., 500., 500.]])
        masks = torch.tensor([[False, True, True, False], [True, True, True, True]])
        target = bellman_targets(values, masks, torch.tensor([3., 500.]),
                                 torch.tensor([False, True]), gamma=.5)
        torch.testing.assert_close(target, torch.tensor([2., 500.]), rtol=0, atol=0)

    def test_time_limit_uses_the_observed_successor_saved_before_reset(self):
        obs = (np.array([1.], dtype=np.float32),)
        successor = (np.array([9.], dtype=np.float32),)
        mask = np.array([True, False, False, False])
        next_mask = np.array([False, True, False, False])
        replay = UniformReplay(2, obs, np.random.default_rng(1))
        replay.add(obs, mask, 0, 1., successor, next_mask, False, True)
        successor[0][0] = 20
        _, following, _, rewards, masks, terminal = replay.batch(1)
        self.assertEqual(float(following[0][0][0]), 9.)
        self.assertTrue(replay.truncated[0])
        target = bellman_targets(torch.tensor([[0., 6., 10., 10.]]), masks, rewards, terminal, .5)
        torch.testing.assert_close(target, torch.tensor([4.]), rtol=0, atol=0)

    def test_replay_rejects_illegal_actions(self):
        obs = (np.array([0.], dtype=np.float32),)
        mask = np.array([False, True, False, False])
        replay = UniformReplay(2, obs, np.random.default_rng(2))
        with self.assertRaises(ValueError):
            replay.add(obs, mask, 0, 0., obs, mask, False, False)
        self.assertEqual(replay.size, 0)
