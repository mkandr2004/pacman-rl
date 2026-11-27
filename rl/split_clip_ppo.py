"""Instrument actual optimizer steps and optionally clip independent gradient groups."""

from math import inf

import torch
from torch.nn.utils import clip_grad_norm_, parameters_to_vector

from rl.critic_ppo_audit import CriticAuditPPO, describe_samples, parameter_groups


def vector(parameters):
    return parameters_to_vector([p.detach() for p in parameters])


class WeightAuditPPO(CriticAuditPPO):
    def __init__(self, *args, split_gradients=False, **kwargs):
        self.split_gradients = split_gradients
        self.initial_actor_weights = None
        self.update_measurements = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["initial_actor_weights", "update_measurements"]

    def _update(self):
        groups = parameter_groups(self.policy)
        actor = groups["actor"]
        params = actor + groups["critic"]
        if len({id(p) for p in params}) != len(params) or {id(p) for p in params} != {
            id(p) for p in self.policy.parameters()
        }:
            raise RuntimeError("Split clipping requires disjoint actor and critic parameters.")
        cap = self.max_grad_norm
        if self.initial_actor_weights is None:
            self.initial_actor_weights = vector(actor)
        before_rollout = vector(actor)
        captured, changes = [], []
        before_step = None

        def before_optimizer(optimizer, args, kwargs):
            nonlocal before_step
            if self.split_gradients:
                for group in groups.values():
                    clip_grad_norm_(group, cap, error_if_nonfinite=True)
            captured.append({name: torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in group if p.grad is not None])
            ).item() for name, group in groups.items()})
            before_step = vector(actor)

        def after_optimizer(optimizer, args, kwargs):
            changes.append(torch.linalg.vector_norm(vector(actor)-before_step).item())

        hooks = (self.policy.optimizer.register_step_pre_hook(before_optimizer),
                 self.policy.optimizer.register_step_post_hook(after_optimizer))
        try:
            # The inherited PPO update retains its own minibatches, losses, KL stop and Adam step.
            # With split clipping its generic clip is a no-op for finite gradients.
            if self.split_gradients:
                self.max_grad_norm = inf
            super()._update()
        finally:
            self.max_grad_norm = cap
            for hook in hooks:
                hook.remove()
        final = vector(actor)
        assert torch.isfinite(final).all()
        assert all(row["actor"] <= cap + 1e-5 and row["critic"] <= cap + 1e-5 for row in captured)
        self.update_measurements = dict(
            post_clip={g: describe_samples([sample[g] for sample in captured]) for g in groups},
            step_l2=describe_samples(changes),
            step_path_l2=sum(changes),
            rollout_delta_l2=torch.linalg.vector_norm(final-before_rollout).item(),
            total_delta_l2=torch.linalg.vector_norm(final-self.initial_actor_weights).item(),
            relative_total_delta=(torch.linalg.vector_norm(final-self.initial_actor_weights) /
                                  torch.linalg.vector_norm(self.initial_actor_weights)).item(),
        )

    def train(self):
        super().train()
        self.rollout_diagnostics[-1]["actor_updates"] = self.update_measurements
