import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
from expectimax import ExpectiMaxGuide

from config import load_config
from networks import DQNNetwork
from dqn_update import masked_greedy_action, DEVICE
from environment_game import OpenSpiel2048Env
from models.load_model import load_checkpoint, load_best


def run_episode(q_net, env, num_actions, max_steps=10_000, render=False):
    """
    Chạy một episode greedy (epsilon=0), trả về stats.

    Returns:
        dict: return, steps, max_tile
    """
    obs  = env.reset()
    done = False
    total_return = 0.0
    steps = 0

    while not done and steps < max_steps:
        legal = env.legal_actions()
        if not legal:
            break

        action = masked_greedy_action(
            qnet=q_net,
            obs=obs,
            legal_actions_list=legal,
            num_actions=num_actions,
            epsilon=0.0,
            device=DEVICE,
        )

        obs, reward, done, _ = env.step(action)
        total_return += reward
        steps += 1

        if render:
            print(env.board)
            print(f"  reward={reward:.0f}  total={total_return:.0f}\n")

    max_tile = int(env.board.max())

    return {"return": total_return, "steps": steps, "max_tile": max_tile}


def run_inference(checkpoint_path: str, n_episodes: int = 10, render: bool = False):
    """
    Load checkpoint và chạy n_episodes greedy, in ra thống kê.

    Args:
        checkpoint_path: Đường dẫn file .pt hoặc thư mục chứa best_model.pt.
        n_episodes: Số episode đánh giá.
        render: In board sau mỗi bước nếu True.
    """
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    cfg = load_config(cfg_path)

    # Tạo env tạm để lấy obs_dim / num_actions
    env = OpenSpiel2048Env(seed=cfg.env.seed)
    obs_dim     = env.obs_dim
    num_actions = env.num_actions

    # Load model — network type is read from saved cfg (checkpoint) or current cfg (best)
    if os.path.isdir(checkpoint_path):
        q_net = load_best(checkpoint_path, obs_dim, num_actions, cfg)
        print(f"Loaded best_model from {checkpoint_path}/")
    else:
        q_net, meta = load_checkpoint(checkpoint_path, obs_dim, num_actions)
        print(f"Loaded checkpoint: episode={meta['episode']}, step={meta['global_step']}")

    print(f"Device: {DEVICE}")
    print(f"Running {n_episodes} greedy episodes...\n")

    results = []
    for i in range(n_episodes):
        stats = run_episode(q_net, env, num_actions, render=render)
        results.append(stats)
        print(f"  Episode {i+1:3d} | return={stats['return']:8.1f} | steps={stats['steps']:5d} | max_tile={stats['max_tile']}")

    returns   = [r["return"]   for r in results]
    max_tiles = [r["max_tile"] for r in results]

    print("\n── Summary ────────────────────────────────")
    print(f"  Return   : mean={np.mean(returns):.1f}  max={np.max(returns):.1f}  min={np.min(returns):.1f}")
    print(f"  Max tile : mean={np.mean(max_tiles):.0f}  best={np.max(max_tiles)}")
    print("───────────────────────────────────────────")


def run_expectimax(n_episodes: int = 10, depth: int = 1, render: bool = False, seed: int = 42):
    """Chạy thuần ExpectiMax (không dùng DQN), in thống kê."""
    guide = ExpectiMaxGuide(depth=depth)
    env   = OpenSpiel2048Env(seed=seed)

    print(f"ExpectiMax depth={depth} | {n_episodes} episodes\n")

    results = []
    for i in range(n_episodes):
        obs  = env.reset(seed=seed + i)
        done = False
        total_return = 0.0
        steps = 0

        while not done and steps < 10_000:
            legal = env.legal_actions()
            if not legal:
                break
            action = guide.best_action(env.state, legal)
            obs, reward, done, _ = env.step(action)
            total_return += reward
            steps += 1

            if render:
                print(env.board)
                print(f"  reward={reward:.0f}  total={total_return:.0f}\n")

        max_tile = int(env.board.max())

        results.append({"return": total_return, "steps": steps, "max_tile": max_tile})
        print(f"  Episode {i+1:3d} | return={total_return:8.1f} | steps={steps:5d} | max_tile={max_tile}")

    returns   = [r["return"]   for r in results]
    max_tiles = [r["max_tile"] for r in results]
    print("\n── Summary ────────────────────────────────")
    print(f"  Return   : mean={np.mean(returns):.1f}  max={np.max(returns):.1f}  min={np.min(returns):.1f}")
    print(f"  Max tile : mean={np.mean(max_tiles):.0f}  best={np.max(max_tiles)}")
    print("───────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="DQN 2048 Inference")
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to .pt checkpoint file or checkpoints/ directory")
    parser.add_argument("--episodes",   type=int,  default=10, help="Number of episodes (default: 10)")
    parser.add_argument("--render",     action="store_true",   help="Print board after each step")
    parser.add_argument("--expectimax", action="store_true",   help="Run pure ExpectiMax (no DQN)")
    parser.add_argument("--depth",      type=int,  default=1,  help="ExpectiMax depth (default: 1)")
    parser.add_argument("--seed",       type=int,  default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.expectimax:
        run_expectimax(n_episodes=args.episodes, depth=args.depth,
                       render=args.render, seed=args.seed)
    else:
        if args.checkpoint is None:
            parser.error("checkpoint is required when not using --expectimax")
        run_inference(args.checkpoint, n_episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()
