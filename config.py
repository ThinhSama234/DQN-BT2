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
    save_every: int = 500
    eval_games: int = 5           # số game để trung bình khi eval
    lr_decay_every: int = 3000    # decay LR mỗi n episode
    lr_decay_factor: float = 0.5  # nhân LR với factor này


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
class GuideConfig:
    enabled:     bool  = False
    depth:       int   = 1        # depth=1 nhanh, depth=2 tốt hơn nhưng chậm hơn
    decay_steps: int   = 30000    # bước để guide_prob: 1.0 → 0.0
    min_prob:    float = 0.0      # xác suất tối thiểu sau khi decay xong


@dataclass
class Config:
    training: TrainingConfig
    epsilon:  EpsilonConfig
    network:  NetworkConfig
    env:      EnvConfig
    guide:    GuideConfig = None  # optional — backward compat với config cũ


def load_config(path: str = "config.yaml") -> Config:
    """Đọc file YAML và trả về Config object."""
    raw = yaml.safe_load(Path(path).read_text())

    # guide là optional — fallback về disabled nếu không có trong YAML
    guide_raw = raw.get("guide", {})
    guide = GuideConfig(
        enabled=guide_raw.get("enabled", False),
        depth=guide_raw.get("depth", 1),
        decay_steps=guide_raw.get("decay_steps", 30000),
        min_prob=guide_raw.get("min_prob", 0.0),
    )

    return Config(
        training=TrainingConfig(**raw["training"]),
        epsilon=EpsilonConfig(**raw["epsilon"]),
        network=NetworkConfig(**raw["network"]),
        env=EnvConfig(**raw["env"]),
        guide=guide,
    )
