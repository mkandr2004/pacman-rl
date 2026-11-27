"""Experimental learned spatial encoder, with no action selection heuristic."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatialHistoryExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=192)
        rows, cols = observation_space['grid'].shape[-2:]
        self.spatial = nn.Sequential(
            nn.Conv2d(11, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32*((rows+1)//2)*((cols+1)//2), 128), nn.ReLU())
        vector_size = (observation_space['pacman_position'].shape[0]
                       + observation_space['ghost_positions'].shape[0]
                       + observation_space['effects'].shape[0]
                       + observation_space['health'].n
                       + observation_space['motion'].shape[0])
        self.vector = nn.Sequential(nn.Linear(vector_size, 64), nn.ReLU())

    def forward(self, observations):
        spatial = torch.cat([observations[k].float() for k in ('grid', 'bonus_types', 'visits')], dim=1)
        vector = torch.cat([observations[k].float().flatten(start_dim=1) for k in
                            ('pacman_position', 'ghost_positions', 'effects', 'health', 'motion')], dim=1)
        return torch.cat([self.spatial(spatial), self.vector(vector)], dim=1)
