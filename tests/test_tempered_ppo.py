import unittest

import numpy as np
import torch

from rl.imitation_env import make_imitation_env
from rl.tempered_ppo import TemperedCategorical, TemperedPolicy


class TemperedPolicyTests(unittest.TestCase):
    def test_temperature_and_physical_mask(self):
        dist = TemperedCategorical(4, .5).proba_distribution(torch.tensor([[0., .5, 1., 20.]]))
        dist.apply_masking(np.array([[True, True, True, False]]))
        torch.testing.assert_close(dist.distribution.probs, torch.cat(
            [torch.softmax(torch.tensor([[0., 1., 2.]]), dim=-1), torch.zeros(1, 1)], dim=-1))

    def test_collection_and_update_have_same_log_probability(self):
        env = make_imitation_env(danger=False)
        try:
            obs, _ = env.reset(seed=88600)
            model = TemperedPolicy(env.observation_space, env.action_space, lambda _: 3e-5, temperature=.5)
            tensors, _ = model.obs_to_tensor(obs)
            mask = env.unwrapped.action_masks()
            with torch.no_grad():
                actions, _, logged = model(tensors, action_masks=mask)
                _, recomputed, _ = model.evaluate_actions(tensors, actions, action_masks=mask)
            self.assertTrue(mask[int(actions.item())])
            torch.testing.assert_close(logged, recomputed, rtol=0, atol=0)
        finally:
            env.close()
