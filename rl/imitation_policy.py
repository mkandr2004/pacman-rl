"""PPO-compatible spatial feature extractor for behavioral cloning."""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImitationCNN(BaseFeaturesExtractor):
    """Combine a map encoding with a player-centered 5×5 view."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        self.rows, self.cols = observation_space["grid"].shape[-2:]
        self.spatial = nn.Sequential(
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * ((self.rows + 1) // 2) * ((self.cols + 1) // 2), 128), nn.ReLU(),
        )
        self.local_encoder = nn.Sequential(nn.Flatten(), nn.Linear(24 * 25, 64), nn.ReLU())
        vector_size = 4 + sum(observation_space[key].shape[0] for key in (
            "pacman_position", "ghost_positions", "effects", "motion"))
        self.vector_encoder = nn.Sequential(nn.Linear(vector_size + 2, 64), nn.ReLU())
        self.register_buffer("offsets", torch.arange(5))

    def forward(self, obs):
        grid = torch.cat([obs[key].float() for key in ("grid", "bonus_types", "visits", "danger")], dim=1)
        features = self.spatial(grid)
        scale = obs["pacman_position"].new_tensor([self.cols * 60 - 1, self.rows * 60 - 1])
        pixels = (obs["pacman_position"] * scale).round()
        cells = torch.div(pixels, 60, rounding_mode="floor").long()
        padded = nn.functional.pad(features.permute(0, 2, 3, 1), (0, 0, 2, 2, 2, 2))
        row = cells[:, 1, None, None] + self.offsets[None, :, None]
        col = cells[:, 0, None, None] + self.offsets[None, None, :]
        local = padded[torch.arange(len(pixels), device=pixels.device)[:, None, None], row, col]
        vector = torch.cat([obs[key].float().flatten(start_dim=1) for key in (
            "pacman_position", "ghost_positions", "effects", "health", "motion")
        ] + [(pixels % 60) / 60], dim=1)
        return torch.cat((self.global_encoder(features), self.local_encoder(local),
                          self.vector_encoder(vector)), dim=1)
