import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

from config import Config, load_config
from dqn_update import (
    DEVICE,
    make_legal_mask,
    masked_greedy_action,
    epsilon_by_step as _epsilon_by_step,
    dqn_update as _dqn_update,
    double_dqn_update as _double_dqn_update,
)
from networks import build_network
from environment_game import OpenSpiel2048Env
from replay_buffer import ReplayBuffer, NStepReplayBuffer

# ── Load config ──────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
cfg: Config = load_config(_cfg_path)

# ── Constants từ config (để train.py dùng trực tiếp) ─────────────────────────
SEED                  = cfg.env.seed
NUM_EPISODES          = cfg.training.num_episodes
MAX_STEPS_PER_EPISODE = cfg.training.max_steps_per_episode
LEARN_START           = cfg.training.learn_start
LEARN_EVERY           = cfg.training.learn_every
BATCH_SIZE            = cfg.training.batch_size
TARGET_SYNC_EVERY     = cfg.training.target_sync_every

# ── Environment ───────────────────────────────────────────────────────────────
train_env = OpenSpiel2048Env(seed=SEED)
obs_dim     = train_env.obs_dim
num_actions = train_env.num_actions

# ── Networks (kiến trúc chọn qua cfg.network.type) ───────────────────────────
q_net      = build_network(cfg, obs_dim, num_actions).to(DEVICE)
target_net = build_network(cfg, obs_dim, num_actions).to(DEVICE)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = optim.Adam(q_net.parameters(), lr=cfg.training.learning_rate)

# ── Replay buffer (loại phụ thuộc vào use_double_dqn) ────────────────────────
if cfg.training.use_double_dqn:
    replay = NStepReplayBuffer(
        capacity=cfg.training.replay_capacity,
        n_steps=cfg.training.n_steps,
        gamma=cfg.training.gamma,
    )
else:
    replay = ReplayBuffer(capacity=cfg.training.replay_capacity)

# ── Wrapper functions (khớp với API train.py — 1 arg) ─────────────────────────
def epsilon_by_step(step: int) -> float:
    return _epsilon_by_step(step, cfg)


def dqn_update(batch) -> float:
    """Tự động chọn vanilla hay double DQN dựa vào config."""
    if cfg.training.use_double_dqn:
        return _double_dqn_update(batch, q_net, target_net, optimizer, cfg)
    return _dqn_update(batch, q_net, target_net, optimizer, cfg)


def main():
    """In thông tin setup để xác nhận trước khi train."""
    print("=" * 55)
    print(f"  Device    : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM      : {vram_gb:.2f} GB")
    print(f"  Network   : {cfg.network.type}  hidden={cfg.network.hidden_dim}")
    params = sum(p.numel() for p in q_net.parameters())
    print(f"  Params    : {params:,}")
    print(f"  obs_dim   : {obs_dim}  |  num_actions: {num_actions}")
    print(f"  Mode      : {'Double DQN + N-step=' + str(cfg.training.n_steps) if cfg.training.use_double_dqn else 'Vanilla DQN'}")
    print(f"  Loss      : {cfg.training.loss}")
    print(f"  Episodes  : {NUM_EPISODES}  |  Buffer: {cfg.training.replay_capacity:,}")
    print(f"  lr={cfg.training.learning_rate}  gamma={cfg.training.gamma}  grad_clip={cfg.training.grad_clip}")
    print("=" * 55)


if __name__ == "__main__":
    main()
