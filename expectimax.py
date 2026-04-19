"""
ExpectiMax guide cho 2048 — thuần numpy, không dùng pyspiel.

2048 là stochastic game: sau mỗi lượt player, game đặt tile ngẫu nhiên.
ExpectiMax xử lý đúng bằng cách xen kẽ Max node và Chance node.

Cây tìm kiếm:
    Max node   (lượt player): chọn action maximize expected value
    Chance node (tile đặt): average có trọng số xác suất

Actions: 0=Up  1=Right  2=Down  3=Left  (khớp Game2048Env).
"""

import numpy as np


# ── Board operations (standalone, dùng cho cả search và env) ─────────────────

def _slide(row: np.ndarray):
    """Slide và merge 1D row sang trái. Trả về (new_row, score)."""
    tiles = row[row != 0]
    result = np.zeros(4, dtype=np.int32)
    score, pos, i = 0, 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            val = int(tiles[i]) * 2
            result[pos] = val
            score += val
            i += 2
        else:
            result[pos] = int(tiles[i])
            i += 1
        pos += 1
    return result, score


def board_apply(action: int, board: np.ndarray):
    """Apply action lên bản copy của board. Trả về (new_board, score, changed)."""
    b = board.copy()
    score = 0
    if action == 0:    # Up
        for j in range(4):
            b[:, j], s = _slide(b[:, j]); score += s
    elif action == 2:  # Down
        for j in range(4):
            col, s = _slide(b[::-1, j]); b[::-1, j] = col; score += s
    elif action == 3:  # Left
        for i in range(4):
            b[i], s = _slide(b[i]); score += s
    elif action == 1:  # Right
        for i in range(4):
            row, s = _slide(b[i, ::-1]); b[i, ::-1] = row; score += s
    changed = not np.array_equal(b, board)
    return b, score, changed


def board_legal_actions(board: np.ndarray) -> list:
    return [a for a in range(4) if board_apply(a, board)[2]]


# ── Heuristic ─────────────────────────────────────────────────────────────────

def _heuristic(board: np.ndarray) -> float:
    """Board quality: empty cells + monotonicity + smoothness + corner bonus."""
    log_b = np.where(board > 0, np.log2(np.maximum(board, 1).astype(float)), 0.0)

    score_empty = float(np.sum(board == 0)) * 100.0

    # Monotonicity — vectorized
    score_mono = 0.0
    for rows in [log_b, log_b.T]:
        d = np.diff(rows, axis=1)
        pos = np.sum(np.maximum(d, 0), axis=1)
        neg = np.sum(np.maximum(-d, 0), axis=1)
        score_mono += float(np.sum(np.maximum(pos, neg)))
    score_mono *= 47.0

    # Smoothness — vectorized, không dùng Python loop
    nz_h = (board[:, :-1] > 0) & (board[:, 1:] > 0)
    nz_v = (board[:-1, :] > 0) & (board[1:, :] > 0)
    score_smooth = -(np.sum(np.abs(log_b[:, :-1] - log_b[:, 1:]) * nz_h)
                   + np.sum(np.abs(log_b[:-1, :] - log_b[1:, :]) * nz_v)) * 10.0

    max_val = log_b.max()
    corners = [log_b[0, 0], log_b[0, 3], log_b[3, 0], log_b[3, 3]]
    score_corner = 35.0 * max_val if max_val in corners else 0.0

    return score_empty + float(score_mono) + float(score_smooth) + score_corner


# ── ExpectiMax ────────────────────────────────────────────────────────────────

class ExpectiMaxGuide:
    """
    Depth-limited ExpectiMax search trên numpy board.

    depth=1 → player → chance → eval   (~120 nodes/move, nhanh)
    depth=2 → player → chance → player → chance → eval   (~4000 nodes/move)
    depth=3 → ~130k nodes/move, mạnh nhưng chậm
    """

    def __init__(self, depth: int = 1):
        self.depth = depth

    def best_action(self, board: np.ndarray, legal_actions_list: list) -> int:
        """Trả về action tốt nhất. board là numpy (4,4) int array."""
        best_val = float("-inf")
        best_act = legal_actions_list[0]
        for action in legal_actions_list:
            new_board, _, _ = board_apply(action, board)
            val = self._chance(new_board, self.depth - 1)
            if val > best_val:
                best_val = val
                best_act = action
        return best_act

    def _chance(self, board: np.ndarray, depth: int) -> float:
        """Chance node: average có trọng số qua tất cả tile placement khả thi."""
        empties = np.argwhere(board == 0)
        if len(empties) == 0:
            return _heuristic(board)
        n = len(empties)
        total = 0.0
        for r, c in empties:
            for tile, prob in [(2, 0.9 / n), (4, 0.1 / n)]:
                b = board.copy()
                b[r, c] = tile
                total += prob * self._max(b, depth)
        return total

    def _max(self, board: np.ndarray, depth: int) -> float:
        """Max node: player chọn action có expected value cao nhất."""
        if depth == 0:
            return _heuristic(board)
        legal = board_legal_actions(board)
        if not legal:
            return _heuristic(board)
        best = float("-inf")
        for action in legal:
            new_board, _, _ = board_apply(action, board)
            best = max(best, self._chance(new_board, depth - 1))
        return best


# ── Schedule ──────────────────────────────────────────────────────────────────

def guide_prob_by_step(step: int, cfg) -> float:
    """Xác suất dùng guide tại step hiện tại. Giảm tuyến tính 1.0 → min_prob."""
    if not getattr(cfg, "guide", None) or not cfg.guide.enabled:
        return 0.0
    frac = min(1.0, step / max(1, cfg.guide.decay_steps))
    return max(cfg.guide.min_prob, 1.0 - frac)
