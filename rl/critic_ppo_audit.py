"""Read-only instrumentation around the regular MaskablePPO update."""

import hashlib

import numpy as np
import torch

from rl.anchored_ppo import AuditedMaskablePPO, masked_kl


def describe_samples(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values)),
                quantiles=np.quantile(values, [.05, .5, .95]).tolist())


def parameter_groups(policy):
    actor = (policy.pi_features_extractor, policy.mlp_extractor.policy_net, policy.action_net)
    critic = (policy.vf_features_extractor, policy.mlp_extractor.value_net, policy.value_net)
    return {name: [p for module in modules for p in module.parameters()]
            for name, modules in (("actor", actor), ("critic", critic))}


class CriticAuditPPO(AuditedMaskablePPO):
    def __init__(self, *args, **kwargs):
        self.first_rollout = None
        self.fixed_probe = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["first_rollout", "fixed_probe"]

    @torch.no_grad()
    def probe(self):
        observations, masks, reference, obs_values = self.fixed_probe
        current = self.policy.get_distribution(observations, action_masks=masks).distribution
        values = self.policy.predict_values(observations).flatten()
        return dict(actor_kl=float(masked_kl(reference, current).mean()),
                    entropy=float(current.entropy().mean()),
                    value_mean=float(values.mean()), value_sd=float(values.std(unbiased=False)),
                    value_rmse_drift=float((values - obs_values).square().mean().sqrt()))

    def _update(self):
        super().train()

    def train(self):
        buffer = self.rollout_buffer
        row = dict(gae=describe_samples(buffer.advantages), values=describe_samples(buffer.values),
                   targets=describe_samples(buffer.returns),
                   gae_positive=float(np.mean(buffer.advantages > 0)))
        if self.first_rollout is None:
            names = ("actions", "action_masks", "log_probs", "rewards", "episode_starts")
            arrays = {key: getattr(buffer, key).copy() for key in names}
            checksum = hashlib.sha256()
            for array in (*arrays.values(), *buffer.observations.values()):
                checksum.update(array.tobytes())
            self.first_rollout = dict(**arrays, advantages=buffer.advantages.copy(),
                                      values=buffer.values.copy(), targets=buffer.returns.copy())
            row["trajectory_sha256"] = checksum.hexdigest()
            obs = {key: torch.as_tensor(value[::4, 0].copy(), device=self.device)
                   for key, value in buffer.observations.items()}
            masks = buffer.action_masks[::4, 0].copy()
            with torch.no_grad():
                reference = self.policy.get_distribution(obs, action_masks=masks).distribution
                values = self.policy.predict_values(obs).flatten()
            self.fixed_probe = (obs, masks, reference, values)

        policy = self.policy
        groups = parameter_groups(policy)
        squared_norms = {name: [] for name in groups}
        updates = []

        def record_norm(group, gradient):
            squared_norms[group].append(gradient.detach().square().sum())

        def record_step(optimizer, args, kwargs):
            norms = {name: torch.stack(sums).sum().sqrt().item() for name, sums in squared_norms.items()}
            updates.append(norms)
            for sums in squared_norms.values():
                sums.clear()

        handles = [p.register_hook(lambda g, name=name: record_norm(name, g))
                   for name, params in groups.items() for p in params]
        handles.append(policy.optimizer.register_step_pre_hook(record_step))
        try:
            self._update()
        finally:
            for handle in handles:
                handle.remove()

        total = [np.hypot(u["actor"], u["critic"]) for u in updates]
        row.update(optimizer_steps=len(updates), gradient_norm={
            **{name: describe_samples([u[name] for u in updates]) for name in groups},
            "global": describe_samples(total), "clip_rate": float(np.mean(np.asarray(total) > self.max_grad_norm)),
        }, fixed_probe=self.probe())
        self.rollout_diagnostics[-1].update(row)
