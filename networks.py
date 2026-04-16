"""
Tất cả kiến trúc Q-network.
Chọn mạng qua cfg.network.type: "vanilla" | "deep" | "dueling"
"""
import torch
import torch.nn as nn

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
class DQNNetwork(nn.Module):
    """
    Vanilla MLP — 2 hidden layers.
    Nhanh, đơn giản, dùng làm baseline.

    obs → hidden → hidden → num_actions
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
class DeepDQNNetwork(nn.Module):
    """
    Deeper MLP — 3 hidden layers với LayerNorm để ổn định gradient.
    Phù hợp khi obs_dim lớn hoặc cần học biểu diễn phức tạp hơn vanilla.

    obs → [hidden+LN] → [hidden+LN] → [hidden/2] → num_actions
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN với CNN feature extractor cho bàn cờ 4×4.

    Input obs: one-hot 288 floats (16 ô × 18 channel).
    CNN xử lý đúng bằng cách reshape → (18, 4, 4) trước khi đưa vào Conv.

    Kiến trúc:
        obs (288,) → reshape (18, 4, 4) ──CNN──▶ shared FC ──┬── Value head    V(s)
                                                              └── Advantage head A(s,a)
    Kết hợp: Q(s,a) = V(s) + A(s,a) − mean_a A(s,.)
    """

    _ROWS = 4
    _COLS = 4
    # One-hot format: 16 ô × 18 channel = 288. Raw format: 16 floats.
    _ONEHOT_CHANNELS = 18
    _ONEHOT_DIM      = _ROWS * _COLS * _ONEHOT_CHANNELS  # 288

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim

        # Tự động chọn số input channel dựa vào obs_dim
        # obs_dim=288 → one-hot 18-channel | obs_dim=16 → raw 1-channel
        if obs_dim == self._ONEHOT_DIM:
            in_ch = self._ONEHOT_CHANNELS   # 18
        else:
            in_ch = 1                        # raw tile values

        self._in_ch = in_ch

        # CNN: (B, in_ch, 4, 4) → feature vector
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64,  kernel_size=3, stride=1, padding=1),  # → (B, 64,  4, 4)
            nn.ReLU(),
            nn.Conv2d(64,  128,   kernel_size=3, stride=1, padding=1),  # → (B, 128, 4, 4)
            nn.ReLU(),
            nn.Conv2d(128, 128,   kernel_size=1),                        # → (B, 128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                                                 # → (B, 2048)
        )
        _cnn_out = 128 * self._ROWS * self._COLS  # 2048

        # Shared FC
        self.shared = nn.Sequential(
            nn.Linear(_cnn_out, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Value stream: V(s) → 1
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s,a) → num_actions
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._in_ch == self._ONEHOT_CHANNELS:
            # one-hot: (B, 288) → (B, 18, 4, 4)
            board = x.view(-1, self._ROWS, self._COLS, self._ONEHOT_CHANNELS) \
                     .permute(0, 3, 1, 2).contiguous()
        else:
            # raw: (B, 16) → (B, 1, 4, 4)
            board = x.view(-1, 1, self._ROWS, self._COLS)

        shared    = self.shared(self.cnn(board))
        value     = self.value_head(shared)      # (B, 1)
        advantage = self.advantage_head(shared)  # (B, num_actions)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ─────────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type] = {
    "vanilla": DQNNetwork,
    "deep":    DeepDQNNetwork,
    "dueling": DuelingDQNNetwork,
}


def build_network(cfg: Config, obs_dim: int, num_actions: int) -> nn.Module:
    """
    Factory function — tạo Q-network dựa trên cfg.network.type.

    Args:
        cfg:         Config object đọc từ config.yaml
        obs_dim:     Kích thước observation
        num_actions: Số action

    Returns:
        nn.Module chưa được .to(device)
    """
    net_type = getattr(cfg.network, "type", "vanilla")
    cls = _REGISTRY.get(net_type)
    if cls is None:
        raise ValueError(
            f"Network type '{net_type}' không hợp lệ. "
            f"Chọn một trong: {list(_REGISTRY)}"
        )
    return cls(obs_dim, num_actions, hidden_dim=cfg.network.hidden_dim)
