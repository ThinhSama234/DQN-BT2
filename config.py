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
    eval_games: int = 3           # số game để trung bình khi eval
    eval_every: int = 50          # eval mỗi n episode
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
class RewardShapingConfig:
    enabled:             bool  = False
    empty_cells_weight:  float = 10.0   # bonus mỗi ô trống
    corner_weight:       float = 50.0   # bonus khi max tile ở góc
    monotonicity_weight: float = 5.0    # bonus board đơn điệu


@dataclass
class PERConfig:
    enabled:    bool  = False
    alpha:      float = 0.6   # mức độ ưu tiên (0=uniform, 1=full priority)
    beta_start: float = 0.4   # IS weight ban đầu (anneals → 1.0)
    beta_end:   float = 1.0
    eps:        float = 1e-6  # tránh priority = 0


@dataclass
class GuideConfig:
    enabled:     bool  = False
    depth:       int   = 1        # depth=1 nhanh, depth=2 tốt hơn nhưng chậm hơn
    decay_steps: int   = 30000    # bước để guide_prob: 1.0 → 0.0
    min_prob:    float = 0.0      # xác suất tối thiểu sau khi decay xong


@dataclass
class Config:
    training:        TrainingConfig
    epsilon:         EpsilonConfig
    network:         NetworkConfig
    env:             EnvConfig
    guide:           GuideConfig           = None
    reward_shaping:  RewardShapingConfig   = None
    per:             PERConfig             = None


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

    rs_raw = raw.get("reward_shaping", {})
    reward_shaping = RewardShapingConfig(
        enabled=rs_raw.get("enabled", False),
        empty_cells_weight=rs_raw.get("empty_cells_weight", 10.0),
        corner_weight=rs_raw.get("corner_weight", 50.0),
        monotonicity_weight=rs_raw.get("monotonicity_weight", 5.0),
    )

    per_raw = raw.get("per", {})
    per = PERConfig(
        enabled=per_raw.get("enabled", False),
        alpha=per_raw.get("alpha", 0.6),
        beta_start=per_raw.get("beta_start", 0.4),
        beta_end=per_raw.get("beta_end", 1.0),
        eps=per_raw.get("eps", 1e-6),
    )

    return Config(
        training=TrainingConfig(**raw["training"]),
        epsilon=EpsilonConfig(**raw["epsilon"]),
        network=NetworkConfig(**raw["network"]),
        env=EnvConfig(**raw["env"]),
        guide=guide,
        reward_shaping=reward_shaping,
        per=per,
    )
