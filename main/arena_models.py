"""The exact, validated local policies exposed by the game lobby."""

from dataclasses import dataclass
import hashlib
from pathlib import Path


MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class ArenaModel:
    name: str
    description: str
    wins: int
    checkpoint: str
    sha256: str

    @property
    def path(self):
        return MODEL_DIRECTORY / self.checkpoint


@dataclass(frozen=True)
class GhostModel:
    name: str
    checkpoint: str
    sha256: str
    stochastic: bool = False
    count: int = 1

    @property
    def path(self):
        return MODEL_DIRECTORY / self.checkpoint


ARENA_MODELS = (
    ArenaModel("DAgger 42", "Imitation seule", 98, "arena_dagger42.zip",
               "aa44bfa613dde1229d5e655c98e303187bd0b9364a96e2f33e25780d1627dfad"),
    ArenaModel("Hybride 337", "Imitation + critique préentraîné + PPO", 97, "arena_hybrid337.zip",
               "13e9421dbce9b91b22dc59ba7c9a7914efd7f27056387564b232206049aefc5f"),
)

GHOST_MODEL = GhostModel("Fantôme lent PPO", "arena_slowghost_ppo.zip",
    "6362278283f4877e93f4973f12bfc88b78968e4d6dded39db2e336fb0bf456fb", stochastic=True)

TEAM_MODEL = GhostModel("Équipe PPO", "arena_ghost_team_2.zip",
    "a1ff9582e14fb3c4736870473352d739455eef01d6ee9c03f0bea62f85d03279", stochastic=True, count=2)


def load_model(entry):
    with entry.path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != entry.sha256:
            raise ValueError(f"Empreinte incorrecte pour {entry.checkpoint}.")

    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    model = MaskablePPO.load(entry.path, device="cpu")
    model.policy.set_training_mode(False)
    return model
