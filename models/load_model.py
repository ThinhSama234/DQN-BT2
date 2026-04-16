import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from networks import build_network, DQNNetwork
from dqn_update import DEVICE


def load_checkpoint(
    path: str,
    obs_dim: int,
    num_actions: int,
    device: torch.device = DEVICE,
) -> tuple[nn.Module, dict]:
    """
    Load checkpoint và tái tạo q_net ở chế độ eval.

    Tự động đọc network type từ cfg được lưu trong checkpoint
    (vanilla / deep / dueling). Nếu checkpoint cũ không có cfg,
    fallback về DQNNetwork.

    Args:
        path:        Đường dẫn file .pt
        obs_dim:     Kích thước observation (phải khớp với lúc lưu).
        num_actions: Số action.
        device:      Thiết bị load lên.

    Returns:
        (q_net, meta) — q_net đã load weights và set eval(),
                        meta là dict chứa episode/global_step/cfg.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    cfg = ckpt.get("cfg")
    if cfg is not None:
        q_net = build_network(cfg, obs_dim, num_actions).to(device)
    else:
        # Fallback cho checkpoint cũ không lưu cfg
        q_net = DQNNetwork(obs_dim, num_actions).to(device)

    q_net.load_state_dict(ckpt["q_net_state_dict"])
    q_net.eval()

    meta = {
        "episode":     ckpt.get("episode"),
        "global_step": ckpt.get("global_step"),
        "cfg":         cfg,
    }
    return q_net, meta


def load_best(
    save_dir: str,
    obs_dim: int,
    num_actions: int,
    cfg,
    device: torch.device = DEVICE,
) -> nn.Module:
    """
    Load best_model.pt từ thư mục checkpoints.

    Args:
        save_dir:    Thư mục chứa best_model.pt.
        obs_dim:     Kích thước observation.
        num_actions: Số action.
        cfg:         Config object — dùng để tái tạo đúng kiến trúc mạng.
        device:      Thiết bị load lên.
    """
    path = os.path.join(save_dir, "best_model.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    q_net = build_network(cfg, obs_dim, num_actions).to(device)
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    q_net.eval()
    return q_net
