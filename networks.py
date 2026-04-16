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

    Kiến trúc:
        [board 4×4] ──CNN──┐
                           ├── shared FC ──┬── Value head    → V(s)       (scalar)
        [extra features] ──┘               └── Advantage head → A(s,a)   (num_actions)

    Kết hợp: Q(s,a) = V(s) + (A(s,a) − mean_a A(s,.))
    → Value stream học "trạng thái này tốt/xấu ra sao"
    → Advantage stream học "action này tốt hơn/kém hơn trung bình bao nhiêu"

    Lợi thế so với vanilla:
    - Ổn định hơn khi nhiều action có Q-value gần nhau
    - Học V(s) độc lập giúp generalize tốt hơn
    """

    _BOARD_ELEMS = 16  # 4×4 board = 16 phần tử đầu trong obs

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        n_extra = max(0, obs_dim - self._BOARD_ELEMS)

        # CNN: 1×4×4 → 16×5×5 → 32×4×4 → flatten(512)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),  # → 16×5×5
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0), # → 32×4×4
            nn.ReLU(),
            nn.Flatten(),                                           # → 512
        )
        _cnn_out = 32 * 4 * 4  # 512

        # Shared FC (CNN features + optional explicit features)
        self.shared = nn.Sequential(
            nn.Linear(_cnn_out + n_extra, hidden_dim), nn.ReLU(),
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
        board = x[:, :self._BOARD_ELEMS].view(-1, 1, 4, 4)
        extra = x[:, self._BOARD_ELEMS:]

        cnn_out = self.cnn(board)
        combined = torch.cat([cnn_out, extra], dim=1) if extra.size(1) > 0 else cnn_out

        shared = self.shared(combined)
        value     = self.value_head(shared)       # (B, 1)
        advantage = self.advantage_head(shared)   # (B, num_actions)

        # Q(s,a) = V(s) + A(s,a) − mean_a A(s,.)
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
