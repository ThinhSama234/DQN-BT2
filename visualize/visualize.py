import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Màu chuẩn của game 2048 cho từng giá trị ô
_TILE_COLORS = {
    0:    "#cdc1b4", 2:    "#eee4da", 4:    "#ede0c8",
    8:    "#f2b179", 16:   "#f59563", 32:   "#f67c5f",
    64:   "#f65e3b", 128:  "#edcf72", 256:  "#edcc61",
    512:  "#edc850", 1024: "#edc53f", 2048: "#edc22e",
}
_TEXT_DARK  = "#776e65"
_TEXT_LIGHT = "#f9f6f2"


def moving_average(x, w: int = 20):
    """Tính moving average với cửa sổ w."""
    if len(x) < w:
        return np.asarray(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(
    episode_returns: list,
    episode_lengths: list,
    loss_history: list,
    save_path: str = None,
):
    """
    Vẽ 3 subplot: training return, episode length, DQN loss.

    Args:
        episode_returns: Return mỗi episode.
        episode_lengths: Số bước mỗi episode.
        loss_history: MSE loss mỗi update step.
        save_path: Nếu truyền vào, lưu file thay vì show. VD: "plots/training.png"
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Return
    ax = axes[0]
    ax.plot(episode_returns, alpha=0.3, color="steelblue", label="episode return")
    ma = moving_average(episode_returns, 20)
    ax.plot(range(len(ma)), ma, color="steelblue", linewidth=2, label="moving avg (20)")
    ax.set_title("Training return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()

    # Episode length
    ax = axes[1]
    ax.plot(episode_lengths, alpha=0.5, color="darkorange")
    ma_len = moving_average(episode_lengths, 20)
    ax.plot(range(len(ma_len)), ma_len, color="darkorange", linewidth=2)
    ax.set_title("Episode length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")

    # Loss
    ax = axes[2]
    ax.plot(loss_history, alpha=0.6, color="tomato")
    ax.set_title("DQN loss (MSE)")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_eval(eval_returns: list, save_path: str = None):
    """
    Vẽ greedy evaluation return theo episode.

    Args:
        eval_returns: List of (episode, return).
        save_path: Nếu truyền vào thì lưu file.
    """
    if not eval_returns:
        return

    eval_eps, eval_vals = zip(*eval_returns)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eval_eps, eval_vals, marker="o", color="mediumseagreen", linewidth=2)
    ax.set_title("Greedy evaluation return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_board(board, title: str = "", save_path: str = None):
    """
    Vẽ bàn cờ 2048 dạng lưới màu (giống giao diện game).

    Args:
        board: np.ndarray shape (4, 4) các giá trị ô (0, 2, 4, ...).
        title: Tiêu đề hình.
        save_path: Nếu truyền vào thì lưu file.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    for r in range(4):
        for c in range(4):
            val   = int(board[r, c])
            color = _TILE_COLORS.get(val, "#3c3a32")
            rect  = plt.Rectangle([c, 3 - r], 1, 1, color=color, linewidth=2, edgecolor="#bbada0")
            ax.add_patch(rect)
            if val > 0:
                text_color = _TEXT_LIGHT if val > 4 else _TEXT_DARK
                ax.text(c + 0.5, 3 - r + 0.5, str(val),
                        ha="center", va="center",
                        fontsize=16 if val < 1000 else 12,
                        fontweight="bold", color=text_color)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path: str = None):
    """Lưu file nếu save_path được truyền, ngược lại hiển thị."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
        plt.close(fig)
    else:
        plt.show()
