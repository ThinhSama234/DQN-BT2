"""
So sánh fair giữa ExpectiMax, DQN, và DDQN trên cùng tập seeds.

Cách dùng:
    # So sánh ExpectiMax vs một checkpoint
    python evaluate.py --expectimax --checkpoint checkpoints/best_model.pt

    # Chỉ chạy ExpectiMax
    python evaluate.py --expectimax --depth 1 --episodes 20

    # So sánh nhiều checkpoint
    python evaluate.py --checkpoint ckpt_dqn.pt --checkpoint ckpt_ddqn.pt

    # Đầy đủ
    python evaluate.py --expectimax --depth 2 --checkpoint dqn.pt --checkpoint ddqn.pt --episodes 50
"""

import argparse
import numpy as np
import torch

from expectimax import ExpectiMaxGuide
from environment_game import Game2048Env
from dqn_update import masked_greedy_action, DEVICE
from models.load_model import load_checkpoint


def _run_agent(get_action, env, seeds, max_steps=10_000):
    """Chạy agent trên danh sách seeds cố định, trả về list kết quả."""
    results = []
    for seed in seeds:
        obs  = env.reset(seed=seed)
        done = False
        total_return = 0.0
        steps = 0
        while not done and steps < max_steps:
            legal = env.legal_actions()
            if not legal:
                break
            action = get_action(obs, env, legal)
            obs, reward, done, _ = env.step(action)
            total_return += reward
            steps += 1
        max_tile = 0
        max_tile = int(env.board.max())
        results.append({"return": total_return, "steps": steps, "max_tile": max_tile})
    return results


def _print_results(name, results):
    returns   = [r["return"]   for r in results]
    max_tiles = [r["max_tile"] for r in results]
    tile_counts = {}
    for t in max_tiles:
        tile_counts[t] = tile_counts.get(t, 0) + 1

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Episodes : {len(results)}")
    print(f"  Return   : mean={np.mean(returns):8.1f}  max={np.max(returns):.1f}  std={np.std(returns):.1f}")
    print(f"  Max tile : mean={np.mean(max_tiles):6.0f}  best={np.max(max_tiles)}")
    sorted_tiles = sorted(tile_counts.items(), reverse=True)
    tile_str = "  ".join(f"{t}×{c}" for t, c in sorted_tiles[:6])
    print(f"  Tile dist: {tile_str}")
    return np.mean(returns)


def _print_comparison(all_results):
    print(f"\n{'═'*50}")
    print("  TỔNG KẾT SO SÁNH")
    print(f"{'═'*50}")
    ranked = sorted(all_results, key=lambda x: x[1], reverse=True)
    for rank, (name, mean_return, mean_tile) in enumerate(ranked, 1):
        print(f"  #{rank}  {name:<30} return={mean_return:8.1f}  avg_tile={mean_tile:.0f}")
    print(f"{'═'*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="So sánh fair ExpectiMax vs DQN/DDQN trên cùng seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--expectimax",  action="store_true",
                        help="Thêm ExpectiMax vào so sánh")
    parser.add_argument("--depth",       type=int, default=1, choices=[1, 2],
                        help="Depth ExpectiMax (default: 1)")
    parser.add_argument("--checkpoint",  action="append", default=[], metavar="PATH",
                        help="Checkpoint .pt cần đánh giá (có thể dùng nhiều lần)")
    parser.add_argument("--episodes",    type=int, default=20,
                        help="Số episode mỗi agent (default: 20)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Seed gốc (default: 42)")
    args = parser.parse_args()

    if not args.expectimax and not args.checkpoint:
        parser.error("Cần ít nhất --expectimax hoặc --checkpoint PATH")

    seeds = list(range(args.seed, args.seed + args.episodes))
    env   = Game2048Env(seed=args.seed)
    num_actions = env.num_actions
    obs_dim     = env.obs_dim

    print(f"Episodes: {args.episodes}  |  Seeds: {seeds[0]}..{seeds[-1]}")

    all_summary = []

    # ── ExpectiMax ────────────────────────────────────────────────────────────
    if args.expectimax:
        guide = ExpectiMaxGuide(depth=args.depth)
        name  = f"ExpectiMax depth={args.depth}"

        def get_expectimax(obs, env, legal):
            return guide.best_action(env.state, legal)

        results = _run_agent(get_expectimax, env, seeds)
        mean_r  = _print_results(name, results)
        all_summary.append((name, mean_r, np.mean([r["max_tile"] for r in results])))

    # ── Checkpoints (DQN / DDQN / ...) ───────────────────────────────────────
    for ckpt_path in args.checkpoint:
        q_net, meta = load_checkpoint(ckpt_path, obs_dim, num_actions)
        q_net.eval()

        ep   = meta.get("episode") or "?"
        step = meta.get("global_step") or "?"
        cfg  = meta.get("cfg")
        net_type = cfg.network.type if cfg else "?"
        name = f"{net_type} ep={ep} ({ckpt_path})"

        def make_action_fn(net):
            def get_action(obs, env, legal):
                return masked_greedy_action(net, obs, legal, num_actions,
                                            epsilon=0.0, device=DEVICE)
            return get_action

        results = _run_agent(make_action_fn(q_net), env, seeds)
        mean_r  = _print_results(name, results)
        all_summary.append((name, mean_r, np.mean([r["max_tile"] for r in results])))

    if len(all_summary) > 1:
        _print_comparison(all_summary)


if __name__ == "__main__":
    main()
