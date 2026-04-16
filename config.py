from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class TrainingConfig:
    gamma: float
    grad_clip: float
    batch_size: int
    learning_rate: float
    replay_capacity: int
    learn_start: int
    learn_every: int
    target_sync_every: int
    num_episodes: int
    max_steps_per_episode: int
    use_double_dqn: bool
    n_steps: int
    loss: str


@dataclass
class EpsilonConfig:
    start: float
    end: float
    decay_steps: int


@dataclass
class NetworkConfig:
    type: str          # vanilla | deep | dueling
    hidden_dim: int


@dataclass
class EnvConfig:
    seed: int


@dataclass
class Config:
    training: TrainingConfig
    epsilon: EpsilonConfig
    network: NetworkConfig
    env: EnvConfig


def load_config(path: str = "config.yaml") -> Config:
    """Đọc file YAML và trả về Config object."""
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        training=TrainingConfig(**raw["training"]),
        epsilon=EpsilonConfig(**raw["epsilon"]),
        network=NetworkConfig(**raw["network"]),
        env=EnvConfig(**raw["env"]),
    )
