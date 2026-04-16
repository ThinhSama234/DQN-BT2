import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


def save_checkpoint(
    q_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: int,
    global_step: int,
    cfg,
    save_dir: str = "checkpoints",
) -> str:
    """
    Lưu checkpoint gồm trọng số q_net, trạng thái optimizer, và metadata.

    Args:
        q_net: Q-network đang train.
        optimizer: Optimizer tương ứng.
        episode: Episode hiện tại.
        global_step: Tổng số bước đã train.
        cfg: Config object (lưu kèm để tái tạo mạng khi load).
        save_dir: Thư mục lưu file. Mặc định "checkpoints/".

    Returns:
        str: Đường dẫn file đã lưu.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"dqn_ep{episode:05d}_step{global_step}.pt")

    torch.save({
        "episode":               episode,
        "global_step":           global_step,
        "q_net_state_dict":      q_net.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "cfg":                   cfg,
    }, path)

    return path


def save_best(q_net: nn.Module, save_dir: str = "checkpoints") -> str:
    """Lưu riêng một file best_model.pt (ghi đè mỗi khi cải thiện)."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "best_model.pt")
    torch.save({"q_net_state_dict": q_net.state_dict()}, path)
    return path
