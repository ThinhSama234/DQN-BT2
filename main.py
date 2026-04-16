"""
Điểm vào duy nhất cho toàn bộ project DQN-2048.

Cách dùng:
    python main.py train
    python main.py search
    python main.py inference checkpoints/best_model.pt
    python main.py visualize

Trong Colab:
    !python main.py train --episodes 500
    !python main.py inference checkpoints/best_model.pt --episodes 5 --render
"""

import argparse
import sys
import os


def cmd_train(args):
    """Train DQN agent."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))
    from training.load_models import main as log_setup, cfg

    if args.episodes:
        cfg.training.num_episodes = args.episodes

    if args.guide is not None:
        cfg.guide.enabled = args.guide

    if args.double_dqn is not None:
        cfg.training.use_double_dqn = args.double_dqn

    import training.train as train_module
    train_module.NUM_EPISODES = cfg.training.num_episodes
    train_module.main(resume_from=args.resume, output_dir=args.output_dir)


def cmd_search(args):
    """Grid / random search hyperparameter."""
    from training.grid_search import run_search
    run_search(
        mode=args.mode,
        n_trials=args.trials,
        num_episodes=args.episodes or 300,
        seed=args.seed,
        space=args.space,
    )


def cmd_inference(args):
    """Chạy model đã train."""
    from inference.inference import run_inference
    run_inference(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes or 10,
        render=args.render,
    )


def cmd_visualize(args):
    """Vẽ lại plot từ log JSON (nếu có) hoặc hướng dẫn."""
    results_path = os.path.join("grid_search_results", "results.json")
    if os.path.exists(results_path):
        import json
        from visualize.visualize import plot_eval
        with open(results_path) as f:
            data = json.load(f)
        evals = [
            (i + 1, r["metrics"].get("mean_eval_return", 0))
            for i, r in enumerate(data)
            if "error" not in r["metrics"]
        ]
        plot_eval(evals, save_path="plots/search_eval.png")
        print("Saved → plots/search_eval.png")
    else:
        print("Chưa có kết quả search. Chạy 'python main.py search' trước.")
        print("Training plots được lưu tự động tại plots/ sau khi train.")


# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="DQN 2048 — entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py train
  python main.py train --episodes 500
  python main.py search --trials 10 --episodes 150
  python main.py search --mode grid --episodes 100
  python main.py inference checkpoints/best_model.pt
  python main.py inference checkpoints/best_model.pt --episodes 5 --render
  python main.py visualize
        """,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # train
    p_train = sub.add_parser("train", help="Train DQN agent")
    p_train.add_argument("--episodes", type=int, default=None,
                         help="Override số episode (mặc định: lấy từ config.yaml)")
    p_train.add_argument("--resume",     type=str, default=None, metavar="CHECKPOINT",
                         help="Tiếp tục train từ checkpoint .pt")
    p_train.add_argument("--output-dir", type=str, default="checkpoints", metavar="DIR",
                         help="Thư mục lưu checkpoint (mặc định: checkpoints/). "
                              "Dùng /drive/MyDrive/... trên Colab")
    p_train.add_argument("--guide", action=argparse.BooleanOptionalAction, default=None,
                         help="Bật/tắt ExpectiMax guide (mặc định: theo config.yaml)")
    p_train.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=None,
                         help="Dùng Double DQN + N-step (mặc định: theo config.yaml). "
                              "--no-double-dqn = vanilla DQN")

    # search
    p_search = sub.add_parser("search", help="Hyperparameter grid/random search")
    p_search.add_argument("--mode",     choices=["grid", "random"], default="random")
    p_search.add_argument("--trials",   type=int, default=20)
    p_search.add_argument("--episodes", type=int, default=300)
    p_search.add_argument("--seed",  type=int, default=0)
    p_search.add_argument("--space", choices=["wide", "narrow", "net"], default="wide",
                          help="wide=toàn bộ | narrow=thu hẹp | net=so sánh kiến trúc")

    # inference
    p_infer = sub.add_parser("inference", help="Chạy model đã train")
    p_infer.add_argument("checkpoint", help="Đường dẫn .pt hoặc thư mục checkpoints/")
    p_infer.add_argument("--episodes", type=int, default=10)
    p_infer.add_argument("--render",   action="store_true", help="In board từng bước")

    # visualize
    sub.add_parser("visualize", help="Vẽ plot từ kết quả search")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "train":     cmd_train,
        "search":    cmd_search,
        "inference": cmd_inference,
        "visualize": cmd_visualize,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
