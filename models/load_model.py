import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from networks import DQNNetwork
from dqn_update import DEVICE


def load_checkpoint(
    path: str,
    obs_dim: int,
    num_actions: int,
    hidden_dim: int = 256,
    device: torch.device = DEVICE,
) -> tuple[nn.Module, dict]:
    """
    Load checkpoint và tái tạo q_net ở chế độ eval.

    Args:
        path: Đường dẫn file .pt
        obs_dim: Kích thước observation (phải khớp với lúc lưu).
        num_actions: Số action.
        hidden_dim: Số neuron hidden layer.
        device: Thiết bị load lên.

    Returns:
        (q_net, meta) — q_net đã load weights và set eval(),
                        meta là dict chứa episode/global_step/cfg.
    """
    ckpt = torch.load(path, map_location=device)

    q_net = DQNNetwork(obs_dim, num_actions, hidden_dim).to(device)
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    q_net.eval()

    meta = {
        "episode":     ckpt.get("episode"),
        "global_step": ckpt.get("global_step"),
        "cfg":         ckpt.get("cfg"),
    }
    return q_net, meta


def load_best(
    save_dir: str,
    obs_dim: int,
    num_actions: int,
    hidden_dim: int = 256,
    device: torch.device = DEVICE,
) -> nn.Module:
    """Shortcut: load best_model.pt từ thư mục checkpoints."""
    path = os.path.join(save_dir, "best_model.pt")
    ckpt = torch.load(path, map_location=device)
    q_net = DQNNetwork(obs_dim, num_actions, hidden_dim).to(device)
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    q_net.eval()
    return q_net
