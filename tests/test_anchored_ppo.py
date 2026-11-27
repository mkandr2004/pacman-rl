import unittest

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.distributions import MaskableCategorical

from rl.anchored_ppo import AuditedMaskablePPO, masked_kl
from rl.imitation_env import make_imitation_env
from scripts.train_ppo_finetuning import RecoveryStarts


class AnchoredPPOTests(unittest.TestCase):
    def test_masked_kl_ignores_illegal_logit(self):
        forbidden = torch.tensor([[False, True, True, False]])
        reference = MaskableCategorical(logits=torch.tensor([[6., 0., 0., 1.]]), masks=forbidden)
        raw_logits = torch.tensor([[-1., 1., 0., 9.]], requires_grad=True)
        changed = MaskableCategorical(logits=raw_logits, masks=forbidden)
        value = masked_kl(reference, changed)
        self.assertAlmostEqual(value.item(), np.logaddexp(0., 1.) - .5 - np.log(2.), places=6)
        value.backward()
        self.assertLess(raw_logits.grad[0,0].abs().item(), 1e-6)
        self.assertLess(raw_logits.grad[0,3].abs().item(), 1e-6)

    def test_anchored_update_trains_the_learner_and_leaves_reference_frozen(self):
        torch.set_num_threads(1)
        env = make_imitation_env(danger=False)
        try:
            kwargs = dict(n_steps=8, batch_size=4, n_epochs=2, seed=12,
                          policy_kwargs={"share_features_extractor": False, "net_arch": [8]})
            original = MaskablePPO("MultiInputPolicy", env, **kwargs)
            clone = AuditedMaskablePPO("MultiInputPolicy", env, anchor_coefficient=1.,
                                       anchor_policy=original.policy, **kwargs)
            before = {key: value.clone() for key,value in original.policy.state_dict().items()}
            clone.policy.load_state_dict(before)
            clone.learn(8, log_interval=None)
            self.assertTrue(all(torch.equal(value,original.policy.state_dict()[key])
                                for key,value in before.items()))
            self.assertTrue(any(not torch.equal(value,clone.policy.state_dict()[key])
                                for key,value in before.items() if key.startswith("action_net.")))
            self.assertEqual(len(clone.rollout_diagnostics),1)
        finally:
            env.close()

    def test_recovery_reset_reproduces_the_seed_and_prefix(self):
        row = dict(seed=1234, prefix=[3,3,3], family="example")
        expected = make_imitation_env(danger=False, prefix=row["prefix"])
        try:
            expected_observation, _ = expected.reset(seed=row["seed"])
            state = expected.unwrapped._last_raw_observation
        finally:
            expected.close()
        sampled = RecoveryStarts([row],42)
        try:
            for _ in range(4):
                obs, _ = sampled.reset()
            self.assertEqual(sampled.family,"example")
            self.assertEqual(sampled.unwrapped._last_raw_observation, state)
            for key,value in expected_observation.items():
                np.testing.assert_array_equal(value, obs[key])
        finally:
            sampled.close()


if __name__ == "__main__":
    unittest.main()
