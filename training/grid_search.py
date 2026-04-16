"""
Grid search / Random search để tinh chỉnh hyperparameter.

Cách dùng:
    # Random search 20 trials (mặc định)
    cd training && python grid_search.py

    # Grid search toàn bộ combinations
    cd training && python grid_search.py --mode grid

    # Chỉ random search N trials
    cd training && python grid_search.py --mode random --trials 10

    # Giới hạn số episode mỗi trial để chạy nhanh
    cd training && python grid_search.py --episodes 200

Kết quả lưu tại:
    grid_search_results/
    ├── results.json          ← toàn bộ kết quả
    ├── best_config.yaml      ← config tốt nhất
    └── trial_XXX/            ← checkpoint của trial nào đó (nếu --save-best)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import copy
import itertools
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from config import Config, load_config
from dqn_update import (
    DEVICE, make_legal_mask, masked_greedy_action,
    epsilon_by_step, dqn_update, double_dqn_update,
)
from environment_game import OpenSpiel2048Env
from networks import build_network
from replay_buffer import ReplayBuffer, NStepReplayBuffer

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
_OUT_DIR  = Path(os.path.dirname(__file__)) / ".." / "grid_search_results"


# ─────────────────────────────────────────────────────────────────────────────
# Search space — chỉnh ở đây để thêm/bớt giá trị cần tìm
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchSpace:
    """
    Định nghĩa không gian tìm kiếm.
    Mỗi field là list các giá trị cần thử.

    Grid search  → thử tất cả combinations (tích Cartesian).
    Random search → sample ngẫu nhiên N tổ hợp từ không gian này.
    """
    # Optimizer
    learning_rate: list = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])

    # Bellman
    gamma:   list = field(default_factory=lambda: [0.95, 0.99])
    n_steps: list = field(default_factory=lambda: [1, 3, 5])

    # Sampling
    batch_size: list = field(default_factory=lambda: [128, 256])

    # Network
    network_type: list = field(default_factory=lambda: ["vanilla", "deep", "dueling"])
    hidden_dim:   list = field(default_factory=lambda: [128, 256])

    # Epsilon decay
    eps_decay_steps: list = field(default_factory=lambda: [50_000, 100_000])


def _all_combinations(space: SearchSpace) -> list[dict]:
    """Tích Cartesian toàn bộ search space."""
    keys = [f.name for f in space.__dataclass_fields__.values()]
    values = [getattr(space, k) for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _random_combinations(space: SearchSpace, n: int, seed: int = 0) -> list[dict]:
    """Sample ngẫu nhiên n tổ hợp từ search space."""
    rng = random.Random(seed)
    keys = [f.name for f in space.__dataclass_fields__.values()]
    values = [getattr(space, k) for k in keys]
    seen: set[tuple] = set()
    combos = []
    max_attempts = n * 20
    for _ in range(max_attempts):
        combo = tuple(rng.choice(v) for v in values)
        if combo not in seen:
            seen.add(combo)
            combos.append(dict(zip(keys, combo)))
        if len(combos) >= n:
            break
    return combos


# ─────────────────────────────────────────────────────────────────────────────
# Config override
# ─────────────────────────────────────────────────────────────────────────────

def _apply_overrides(base_cfg: Config, params: dict) -> Config:
    """
    Tạo Config mới từ base_cfg với các giá trị trong params được ghi đè.
    Không mutate base_cfg.
    """
    cfg = copy.deepcopy(base_cfg)

    mapping = {
        "learning_rate":  ("training", "learning_rate"),
        "gamma":          ("training", "gamma"),
        "n_steps":        ("training", "n_steps"),
        "batch_size":     ("training", "batch_size"),
        "network_type":   ("network",  "type"),
        "hidden_dim":     ("network",  "hidden_dim"),
        "eps_decay_steps":("epsilon",  "decay_steps"),
    }

    for key, value in params.items():
        if key not in mapping:
            continue
        section, attr = mapping[key]
        setattr(getattr(cfg, section), attr, value)

    # n_steps > 1 → dùng double DQN tự động
    cfg.training.use_double_dqn = (cfg.training.n_steps > 1)

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Experiment builder — tạo mới mọi object cho mỗi trial
# ─────────────────────────────────────────────────────────────────────────────

def _build_experiment(cfg: Config) -> dict:
    """Khởi tạo env, network, optimizer, buffer từ cfg."""
    env = OpenSpiel2048Env(seed=cfg.env.seed)

    q_net      = build_network(cfg, env.obs_dim, env.num_actions).to(DEVICE)
    target_net = build_network(cfg, env.obs_dim, env.num_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.training.learning_rate)

    if cfg.training.use_double_dqn:
        replay = NStepReplayBuffer(
            capacity=cfg.training.replay_capacity,
            n_steps=cfg.training.n_steps,
            gamma=cfg.training.gamma,
        )
    else:
        replay = ReplayBuffer(capacity=cfg.training.replay_capacity)

    return dict(
        cfg=cfg, env=env,
        q_net=q_net, target_net=target_net,
        optimizer=optimizer, replay=replay,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(exp: dict, num_episodes: int, eval_every: int = 25) -> dict:
    """
    Chạy một trial training.

    Returns:
        dict với các metrics: mean_eval_return, best_eval_return,
                              mean_ep_return (cuối), wall_time_s
    """
    cfg        = exp["cfg"]
    env        = exp["env"]
    q_net      = exp["q_net"]
    target_net = exp["target_net"]
    optimizer  = exp["optimizer"]
    replay     = exp["replay"]

    update_fn = double_dqn_update if cfg.training.use_double_dqn else dqn_update

    num_actions = env.num_actions
    global_step = 0
    ep_returns: list[float] = []
    eval_returns: list[float] = []

    t0 = time.time()

    for episode in range(1, num_episodes + 1):
        obs  = env.reset(seed=cfg.env.seed + episode)
        done = False
        ep_return = 0.0
        ep_len    = 0

        while not done and ep_len < cfg.training.max_steps_per_episode:
            eps   = epsilon_by_step(global_step, cfg)
            legal = env.legal_actions()
            if not legal:
                break
            legal_mask = make_legal_mask(num_actions, legal)

            action = masked_greedy_action(
                qnet=q_net, obs=obs,
                legal_actions_list=legal, num_actions=num_actions,
                epsilon=eps, device=DEVICE,
            )

            next_obs, reward, done, info = env.step(action)
            next_legal      = info["legal_actions"] if not done else []
            next_legal_mask = make_legal_mask(num_actions, next_legal)

            replay.add(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)

            obs        = next_obs
            ep_return += reward
            ep_len    += 1
            global_step += 1

            if (len(replay) >= cfg.training.learn_start
                    and global_step % cfg.training.learn_every == 0):
                batch = replay.sample(cfg.training.batch_size)
                update_fn(batch, q_net, target_net, optimizer, cfg)

            if global_step % cfg.training.target_sync_every == 0:
                target_net.load_state_dict(q_net.state_dict())

        ep_returns.append(ep_return)

        # Greedy eval
        if episode % eval_every == 0:
            ret_eval = _greedy_eval(q_net, cfg, num_actions)
            eval_returns.append(ret_eval)

    wall_time = time.time() - t0
    last_n = ep_returns[-min(50, len(ep_returns)):]

    return {
        "mean_eval_return":  float(np.mean(eval_returns)) if eval_returns else 0.0,
        "best_eval_return":  float(max(eval_returns))     if eval_returns else 0.0,
        "mean_ep_return":    float(np.mean(last_n)),
        "wall_time_s":       round(wall_time, 1),
        "total_steps":       global_step,
    }


def _greedy_eval(q_net, cfg: Config, num_actions: int, n_ep: int = 3) -> float:
    """Chạy n_ep episodes greedy, trả về return trung bình."""
    env = OpenSpiel2048Env(seed=9999)
    total = 0.0
    for i in range(n_ep):
        obs  = env.reset(seed=9000 + i)
        done = False
        steps = 0
        while not done and steps < cfg.training.max_steps_per_episode:
            legal = env.legal_actions()
            if not legal:
                break
            action = masked_greedy_action(
                qnet=q_net, obs=obs,
                legal_actions_list=legal, num_actions=num_actions,
                epsilon=0.0, device=DEVICE,
            )
            obs, reward, done, _ = env.step(action)
            total += reward
            steps += 1
    return total / n_ep


# ─────────────────────────────────────────────────────────────────────────────
# Grid / random search runner
# ─────────────────────────────────────────────────────────────────────────────

def run_search(
    mode: str = "random",
    n_trials: int = 20,
    num_episodes: int = 300,
    save_best: bool = True,
    seed: int = 0,
):
    """
    Chạy grid search hoặc random search.

    Args:
        mode:         "grid" hoặc "random"
        n_trials:     Số trial khi mode="random" (bỏ qua khi mode="grid")
        num_episodes: Số episode mỗi trial (nên ngắn hơn train thật, VD 200-500)
        save_best:    Lưu config tốt nhất ra best_config.yaml
        seed:         Seed cho random search
    """
    base_cfg = load_config(_CFG_PATH)
    space    = SearchSpace()

    if mode == "grid":
        combos = _all_combinations(space)
        print(f"\nGrid search: {len(combos)} combinations")
    else:
        combos = _random_combinations(space, n=n_trials, seed=seed)
        print(f"\nRandom search: {len(combos)} trials  (seed={seed})")

    print(f"Episodes per trial: {num_episodes}  |  Device: {DEVICE}\n")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for i, params in enumerate(combos):
        trial_id = f"trial_{i+1:03d}"
        print(f"[{i+1}/{len(combos)}] {trial_id}  params={_fmt_params(params)}", end="  ", flush=True)

        cfg = _apply_overrides(base_cfg, params)
        exp = _build_experiment(cfg)

        try:
            metrics = _run_trial(exp, num_episodes=num_episodes)
        except Exception as e:
            print(f"  ERROR: {e}")
            metrics = {"mean_eval_return": float("-inf"), "error": str(e)}

        record = {
            "trial_id": trial_id,
            "params":   params,
            "metrics":  metrics,
        }
        all_results.append(record)

        print(f"eval={metrics.get('mean_eval_return', 0):.1f}"
              f"  best={metrics.get('best_eval_return', 0):.1f}"
              f"  time={metrics.get('wall_time_s', 0):.0f}s")

        # Lưu kết quả sau mỗi trial (để không mất kết quả nếu crash)
        _save_results(all_results)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(all_results)

    if save_best:
        best = max(all_results, key=lambda r: r["metrics"].get("mean_eval_return", float("-inf")))
        _save_best_config(base_cfg, best)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_params(params: dict) -> str:
    short = {
        "learning_rate":   "lr",
        "gamma":           "γ",
        "n_steps":         "N",
        "batch_size":      "bs",
        "network_type":    "net",
        "hidden_dim":      "hid",
        "eps_decay_steps": "eps_steps",
    }
    return " ".join(f"{short.get(k,k)}={v}" for k, v in params.items())


def _save_results(results: list[dict]):
    path = _OUT_DIR / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _save_best_config(base_cfg: Config, best_record: dict):
    cfg = _apply_overrides(base_cfg, best_record["params"])
    out = {
        "training": {
            "gamma":               cfg.training.gamma,
            "grad_clip":           cfg.training.grad_clip,
            "batch_size":          cfg.training.batch_size,
            "learning_rate":       cfg.training.learning_rate,
            "replay_capacity":     cfg.training.replay_capacity,
            "learn_start":         cfg.training.learn_start,
            "learn_every":         cfg.training.learn_every,
            "target_sync_every":   cfg.training.target_sync_every,
            "num_episodes":        cfg.training.num_episodes,
            "max_steps_per_episode": cfg.training.max_steps_per_episode,
            "use_double_dqn":      cfg.training.use_double_dqn,
            "n_steps":             cfg.training.n_steps,
            "loss":                cfg.training.loss,
        },
        "epsilon": {
            "start":        cfg.epsilon.start,
            "end":          cfg.epsilon.end,
            "decay_steps":  cfg.epsilon.decay_steps,
        },
        "network": {
            "type":       cfg.network.type,
            "hidden_dim": cfg.network.hidden_dim,
        },
        "env": {"seed": cfg.env.seed},
    }
    path = _OUT_DIR / "best_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True)
    print(f"\nBest config saved → {path}")


def _print_summary(results: list[dict]):
    valid = [r for r in results if "error" not in r["metrics"]]
    if not valid:
        print("\nKhông có trial nào thành công.")
        return

    sorted_results = sorted(
        valid,
        key=lambda r: r["metrics"].get("mean_eval_return", float("-inf")),
        reverse=True,
    )

    print("\n" + "=" * 70)
    print(f"  {'Rank':<5} {'Trial':<12} {'MeanEval':>10} {'BestEval':>10}  Params")
    print("-" * 70)
    for rank, r in enumerate(sorted_results[:10], 1):  # top 10
        m = r["metrics"]
        print(
            f"  {rank:<5} {r['trial_id']:<12}"
            f" {m.get('mean_eval_return',0):>10.1f}"
            f" {m.get('best_eval_return',0):>10.1f}"
            f"  {_fmt_params(r['params'])}"
        )
    print("=" * 70)
    best = sorted_results[0]
    print(f"\n  Best trial: {best['trial_id']}")
    print(f"  Params    : {_fmt_params(best['params'])}")
    print(f"  MeanEval  : {best['metrics'].get('mean_eval_return', 0):.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search cho DQN 2048")
    parser.add_argument("--mode",     choices=["grid", "random"], default="random",
                        help="grid = thử tất cả; random = sample ngẫu nhiên (mặc định: random)")
    parser.add_argument("--trials",   type=int, default=20,
                        help="Số trial khi mode=random (mặc định: 20)")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Số episode mỗi trial (mặc định: 300)")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Seed cho random search (mặc định: 0)")
    parser.add_argument("--no-save",  action="store_true",
                        help="Không lưu best_config.yaml")
    args = parser.parse_args()

    run_search(
        mode=args.mode,
        n_trials=args.trials,
        num_episodes=args.episodes,
        save_best=not args.no_save,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
