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
    scheduler=None,
    best_eval_return: float = float("-inf"),
    save_dir: str = "checkpoints",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"dqn_ep{episode:05d}_step{global_step}.pt")
    torch.save({
        "episode":               episode,
        "global_step":           global_step,
        "q_net_state_dict":      q_net.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict() if scheduler else None,
        "best_eval_return":      best_eval_return,
        "cfg":                   cfg,
    }, path)
    return path


def save_best(
    q_net: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    episode: int = 0,
    global_step: int = 0,
    cfg=None,
    scheduler=None,
    best_eval_return: float = float("-inf"),
    save_dir: str = "checkpoints",
) -> str:
    """Lưu best_model.pt với đầy đủ state để có thể resume."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "best_model.pt")
    torch.save({
        "episode":               episode,
        "global_step":           global_step,
        "q_net_state_dict":      q_net.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict":  scheduler.state_dict() if scheduler else None,
        "best_eval_return":      best_eval_return,
        "cfg":                   cfg,
    }, path)
    return path
