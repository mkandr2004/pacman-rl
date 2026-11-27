"""Uniform-replay DQN for the original four-direction Pac-Man game."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FORMAT = "pac-ai-arena-interaction-dqn-v1"
VECTOR_SIZE = 47


def pack_observation(obs):
    grid = np.concatenate((obs["grid"], obs["bonus_types"]), axis=0)
    health = np.zeros(4, dtype=np.float32)
    health[int(obs["health"])] = 1
    vector = np.concatenate((obs["pacman_position"], obs["ghost_positions"], obs["effects"],
                             health, obs["motion"])).astype(np.float32, copy=False)
    return grid, obs["visits"], vector


def masked_values(values, masks):
    return values.masked_fill(~masks, -torch.inf)


def bellman_targets(target_values, next_masks, rewards, terminated, gamma):
    """Bootstrap through time limits, and suppress bootstrap at real terminal states."""
    future = masked_values(target_values, next_masks).amax(dim=1)
    return rewards + gamma * torch.where(terminated, 0., future)


class PacmanQNetwork(nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.spatial = nn.Sequential(nn.Conv2d(11, 16, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Flatten(),
                                     nn.Linear(24 * ((rows + 3) // 4) * ((cols + 3) // 4), 96),
                                     nn.ReLU())
        self.local = nn.Sequential(nn.Flatten(), nn.Linear(11 * 25, 48), nn.ReLU())
        self.vector = nn.Sequential(nn.Linear(VECTOR_SIZE, 48), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, 4))
        self.register_buffer("offsets", torch.arange(5), persistent=False)

    def forward(self, grid, visits, context):
        spatial = torch.cat((grid.float(), visits), dim=1)
        scale = context.new_tensor((self.cols * 60 - 1, self.rows * 60 - 1))
        center = ((context[:, :2] * scale).round().long() // 60)
        padded = F.pad(spatial, (2, 2, 2, 2)).permute(0, 2, 3, 1)
        row = center[:, 1, None, None] + self.offsets[None, :, None]
        col = center[:, 0, None, None] + self.offsets[None, None, :]
        local = padded[torch.arange(len(context), device=context.device)[:, None, None], row, col]
        return self.head(torch.cat((self.spatial(spatial), self.local(local), self.vector(context)), dim=1))

    @torch.inference_mode()
    def greedy_action(self, packed, mask):
        inputs = (torch.as_tensor(part)[None] for part in packed)
        values = self(*inputs)
        return int(masked_values(values, torch.as_tensor(mask)[None]).argmax(dim=1))

    def predict(self, obs, *, deterministic=True, action_masks):
        if not deterministic:
            raise ValueError("Use the trainer's isolated epsilon-greedy sampler for exploration.")
        return self.greedy_action(pack_observation(obs), action_masks), None


class UniformReplay:
    def __init__(self, capacity, example, rng):
        self.capacity = capacity
        self.rng = rng
        self.size = 0
        self.index = 0
        self.current = tuple(np.empty((capacity, *value.shape), dtype=value.dtype) for value in example)
        self.following = tuple(np.empty_like(value) for value in self.current)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.masks = np.empty((capacity, 4), dtype=np.bool_)
        self.next_masks = np.empty_like(self.masks)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty_like(self.terminated)

    def add(self, obs, mask, action, reward, next_obs, next_mask, terminated, truncated):
        if not mask[action] or not next_mask.any():
            raise ValueError("A replay transition must have a legal chosen action and a valid next mask.")
        index = self.index
        for storage, value in zip(self.current, obs):
            storage[index] = value
        for storage, value in zip(self.following, next_obs):
            storage[index] = value
        self.actions[index] = action
        self.masks[index] = mask
        self.next_masks[index] = next_mask
        self.rewards[index] = reward
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.size = min(self.size + 1, self.capacity)
        self.index = (index + 1) % self.capacity

    def batch(self, count):
        indices = self.rng.integers(self.size, size=count)
        current = tuple(torch.as_tensor(value[indices]) for value in self.current)
        following = tuple(torch.as_tensor(value[indices]) for value in self.following)
        fields = tuple(torch.as_tensor(value[indices]) for value in
                       (self.actions, self.rewards, self.next_masks, self.terminated))
        return current, following, *fields

    @property
    def allocated_bytes(self):
        arrays = (*self.current, *self.following, self.actions, self.masks, self.next_masks,
                  self.rewards, self.terminated, self.truncated)
        return sum(value.nbytes for value in arrays)


def load_dqn(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != FORMAT:
        raise ValueError("Unsupported Pac-Man DQN checkpoint.")
    model = PacmanQNetwork(checkpoint["rows"], checkpoint["cols"])
    model.load_state_dict(checkpoint["model"])
    return model.eval()
