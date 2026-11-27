"""Maskable PPO with optional forward KL to a frozen masked actor."""

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


def masked_kl(reference, current):
    """KL(reference || current) after both distributions apply the same mask."""
    probabilities = reference.probs
    return (probabilities * (reference.logits - current.logits)).sum(dim=-1)


class AuditedMaskablePPO(MaskablePPO):
    def __init__(self, *args, anchor_coefficient=0., anchor_policy=None, **kwargs):
        self.anchor_coefficient = anchor_coefficient
        self.anchor_policy = anchor_policy
        self.rollout_diagnostics = []
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["anchor_policy", "rollout_diagnostics"]

    def train(self):
        if self.anchor_coefficient:
            self._train_anchored()
        else:
            super().train()
        log = self.logger.name_to_value
        self.rollout_diagnostics.append(dict(
            steps=self.num_timesteps,
            **{key.removeprefix("train/"): float(log[key]) for key in (
                "train/approx_kl", "train/anchor_kl", "train/kl_stop", "train/clip_fraction",
                "train/explained_variance", "train/value_loss", "train/entropy_loss"
            ) if key in log}))

    def _train_anchored(self):
        if self.anchor_policy is None:
            raise RuntimeError("Attach the verified frozen actor before resuming anchored PPO.")
        if self.clip_range_vf is not None:
            raise ValueError("Anchored PPO in this experiment does not use value clipping.")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        diagnostics = {key: [] for key in ("approx_kl", "anchor_kl", "clip_fraction",
                                            "value_loss", "entropy_loss", "loss")}
        stopped = False
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                actions = batch.actions.long().flatten()
                distribution = self.policy.get_distribution(batch.observations, action_masks=batch.action_masks)
                log_probability = distribution.log_prob(actions)
                log_ratio = log_probability - batch.old_log_prob
                ratio = log_ratio.exp()
                with torch.no_grad():
                    local_kl = ((ratio - 1) - log_ratio).mean().item()
                    reference = self.anchor_policy.get_distribution(
                        batch.observations, action_masks=batch.action_masks).distribution
                diagnostics["approx_kl"].append(local_kl)
                if self.target_kl is not None and local_kl > 1.5 * self.target_kl:
                    stopped = True
                    break
                values = self.policy.predict_values(batch.observations).flatten()
                advantages = batch.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                clipped = ratio.clamp(1 - clip, 1 + clip)
                policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
                value_loss = (batch.returns - values).square().mean()
                entropy_loss = -distribution.entropy().mean()
                anchor_kl = masked_kl(reference, distribution.distribution).mean()
                loss = (policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                        + self.anchor_coefficient * anchor_kl)
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                diagnostics["anchor_kl"].append(anchor_kl.item())
                diagnostics["value_loss"].append(value_loss.item())
                diagnostics["entropy_loss"].append(entropy_loss.item())
                diagnostics["loss"].append(loss.item())
                diagnostics["clip_fraction"].append(((ratio - 1).abs() > clip).float().mean().item())
            self._n_updates += 1
            if stopped:
                break
        for name, samples in diagnostics.items():
            self.logger.record(f"train/{name}", float(np.mean(samples)) if samples else 0.)
        self.logger.record("train/kl_stop", int(stopped))
        self.logger.record("train/explained_variance", explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
