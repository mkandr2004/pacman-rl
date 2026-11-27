"""Masked policy using the same temperature for action sampling and PPO likelihoods."""

import math

from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy


class TemperedCategorical(MaskableCategoricalDistribution):
    def __init__(self, actions, temperature):
        super().__init__(actions)
        self.temperature = temperature

    def proba_distribution(self, action_logits):
        return super().proba_distribution(action_logits / self.temperature)


class TemperedPolicy(MaskableMultiInputActorCriticPolicy):
    def __init__(self, *args, temperature=1., **kwargs):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Policy temperature must be finite and positive.")
        super().__init__(*args, **kwargs)
        self.action_dist = TemperedCategorical(self.action_space.n, temperature)
