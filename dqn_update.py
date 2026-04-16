"""
Core DQN utilities: action selection, epsilon schedule, update functions.
Networks → networks.py | Loss functions → utils/losses.py
"""
import logging
import random

import numpy as np
import torch
import torch.nn as nn

from config import Config
from replay_buffer import Transition, NStepTransition
from utils.losses import compute_loss

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Action utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_legal_mask(num_actions: int, legal_actions_list: list) -> np.ndarray:
    """
    Tạo binary mask float32 shape (num_actions,).
    1.0 ở các vị trí action hợp lệ, 0.0 còn lại.
    """
    mask = np.zeros(num_actions, dtype=np.float32)
    mask[legal_actions_list] = 1.0
    return mask


@torch.no_grad()
def masked_greedy_action(
    qnet: nn.Module,
    obs: np.ndarray,
    legal_actions_list: list,
    num_actions: int,
    epsilon: float = 0.0,
    device: torch.device = DEVICE,
) -> int:
    """
    Epsilon-greedy action selection chỉ trong tập action hợp lệ.
    Action bất hợp lệ bị che bằng -1e9 trước khi argmax.

    Args:
        qnet:               Q-network ở chế độ eval (hoặc train).
        obs:                Observation hiện tại, shape (obs_dim,).
        legal_actions_list: Danh sách action hợp lệ.
        num_actions:        Tổng số action.
        epsilon:            Xác suất chọn ngẫu nhiên.
        device:             CPU hoặc CUDA.

    Returns:
        int: Index action được chọn.
    """
    if random.random() < epsilon:
        return random.choice(legal_actions_list)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q = qnet(obs_t).squeeze(0)

    mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    mask[legal_actions_list] = True
    q = q.masked_fill(~mask, -1e9)

    return int(torch.argmax(q).item())


def epsilon_by_step(step: int, cfg: Config) -> float:
    """
    Lịch giảm epsilon tuyến tính: start → end trong decay_steps bước.
    Sau decay_steps giữ nguyên ở end.
    """
    frac = min(1.0, step / cfg.epsilon.decay_steps)
    return cfg.epsilon.start + frac * (cfg.epsilon.end - cfg.epsilon.start)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_update_stats(tag: str, q_sa: torch.Tensor, target: torch.Tensor,
                      loss: torch.Tensor, grad_norm: float):
    """Ghi DEBUG stats sau mỗi update để dễ debug divergence."""
    logger.debug(
        "[%s] loss=%.5f | q_sa mean=%.3f std=%.3f max=%.3f | "
        "target mean=%.3f std=%.3f | grad_norm=%.4f",
        tag,
        loss.item(),
        q_sa.mean().item(), q_sa.std().item(), q_sa.max().item(),
        target.mean().item(), target.std().item(),
        grad_norm,
    )


def _to(arr, dtype, device):
    return torch.tensor(np.asarray(arr), dtype=dtype, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla DQN update (1-step, standard Bellman)
# ─────────────────────────────────────────────────────────────────────────────

def dqn_update(
    batch: Transition,
    q_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> float:
    """
    1-step DQN update.

    target = r + γ * max_a' Q_target(s', a')   (0 nếu done)
    loss   = loss_fn(Q(s,a), target)

    Args:
        batch:      Transition batch từ ReplayBuffer.
        q_net:      Q-network đang train.
        target_net: Target network (sync chậm).
        optimizer:  Optimizer cho q_net.
        cfg:        Config object.

    Returns:
        float: loss value của bước này.
    """
    obs             = _to(batch.state,           torch.float32, DEVICE)
    actions         = _to(batch.action,          torch.int64,   DEVICE).unsqueeze(1)
    rewards         = _to(batch.reward,          torch.float32, DEVICE)
    next_obs        = _to(batch.next_state,      torch.float32, DEVICE)
    dones           = _to(batch.done,            torch.float32, DEVICE)
    next_legal_mask = _to(batch.next_legal_mask, torch.bool,    DEVICE)

    q_sa = q_net(obs).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_obs).masked_fill(~next_legal_mask, -1e9)
        next_max_q = torch.max(next_q, dim=1).values
        next_max_q = torch.where(dones > 0.5, torch.zeros_like(next_max_q), next_max_q)
        target = rewards + cfg.training.gamma * next_max_q

    loss = compute_loss(q_sa, target, cfg.training.loss)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(q_net.parameters(), cfg.training.grad_clip)
    optimizer.step()

    _log_update_stats("dqn", q_sa, target, loss, float(grad_norm))
    return float(loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# Double DQN update (N-step return)
# ─────────────────────────────────────────────────────────────────────────────

def double_dqn_update(
    batch: NStepTransition,
    q_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> float:
    """
    Double DQN + N-step return update.

    Double DQN: q_net chọn action tốt nhất, target_net đánh giá giá trị đó.
    → Giảm overestimation bias so với vanilla DQN.

    N-step target:
        target = n_step_reward + γ^N * Q_target(s_{t+N}, argmax_a Q(s_{t+N}, a))
                 (0 nếu n_step_done=True)

    Args:
        batch:      NStepTransition batch từ NStepReplayBuffer.
        q_net:      Q-network đang train.
        target_net: Target network (sync chậm).
        optimizer:  Optimizer cho q_net.
        cfg:        Config object.

    Returns:
        float: loss value của bước này.
    """
    obs                   = _to(batch.state,                torch.float32, DEVICE)
    actions               = _to(batch.action,               torch.int64,   DEVICE).unsqueeze(1)
    n_rewards             = _to(batch.n_step_reward,        torch.float32, DEVICE)
    n_next_obs            = _to(batch.n_step_next_state,    torch.float32, DEVICE)
    n_dones               = _to(batch.n_step_done,          torch.float32, DEVICE)
    n_next_legal_mask     = _to(batch.n_step_next_legal_mask, torch.bool,  DEVICE)

    # Q(s, a) — dự đoán từ q_net
    q_sa = q_net(obs).gather(1, actions).squeeze(1)

    with torch.no_grad():
        # Double DQN: q_net chọn action, target_net đánh giá
        q_next_online = q_net(n_next_obs).masked_fill(~n_next_legal_mask, -1e9)
        best_next_actions = q_next_online.argmax(dim=1).unsqueeze(1)  # argmax từ q_net

        q_next_target = target_net(n_next_obs)
        n_step_max_q  = q_next_target.gather(1, best_next_actions).squeeze(1)  # đánh giá bằng target_net

        # Zero out terminal states
        n_step_max_q = torch.where(n_dones > 0.5, torch.zeros_like(n_step_max_q), n_step_max_q)

        # N-step Bellman target
        target = n_rewards + (cfg.training.gamma ** cfg.training.n_steps) * n_step_max_q

    loss = compute_loss(q_sa, target, cfg.training.loss)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(q_net.parameters(), cfg.training.grad_clip)
    optimizer.step()

    _log_update_stats("double_dqn", q_sa, target, loss, float(grad_norm))
    return float(loss.item())
