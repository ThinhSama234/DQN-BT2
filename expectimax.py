"""
ExpectiMax guide cho cold-start training.

2048 là stochastic game: sau mỗi lượt player, game đặt tile ngẫu nhiên
→ dùng ExpectiMax (không phải Minimax) để xử lý chance node đúng cách.

Cây tìm kiếm:
    Max node  (lượt player): chọn action maximize expected value
    Chance node (tile placement): average có trọng số theo xác suất

Tích hợp vào training:
    - guide_prob bắt đầu = 1.0, giảm tuyến tính về 0 sau decay_steps bước
    - Khi guide active: action = expectimax.best_action(state, legal)
    - Experience vẫn lưu vào replay buffer bình thường
    → Network học từ trajectory chất lượng cao trong giai đoạn đầu
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from helper import state_return, legal_actions as _legal_actions


class ExpectiMaxGuide:
    """
    Depth-limited ExpectiMax search.

    depth=1 → player → chance → eval   (~120 nodes/move, nhanh)
    depth=2 → player → chance → player → chance → eval   (~4000 nodes/move)
    """

    def __init__(self, depth: int = 1):
        self.depth = depth

    # ── Heuristic lá cây ──────────────────────────────────────────────────────
    def _heuristic(self, state) -> float:
        """Score tích lũy của game — proxy tốt cho chất lượng board."""
        return state_return(state)

    # ── Đệ quy chính ──────────────────────────────────────────────────────────
    def _expectimax(self, state, depth: int) -> float:
        if state.is_terminal():
            return self._heuristic(state)

        if state.is_chance_node():
            # Chance node: trung bình có trọng số xác suất (không giảm depth)
            value = 0.0
            for action, prob in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                value += prob * self._expectimax(child, depth)
            return value

        # Max node
        if depth == 0:
            return self._heuristic(state)

        legal = _legal_actions(state)
        if not legal:
            return self._heuristic(state)

        best = float("-inf")
        for action in legal:
            child = state.clone()
            child.apply_action(action)
            best = max(best, self._expectimax(child, depth - 1))
        return best

    # ── Public API ────────────────────────────────────────────────────────────
    def best_action(self, state, legal_actions_list: list) -> int:
        """Trả về action tốt nhất theo ExpectiMax từ state hiện tại."""
        best_val = float("-inf")
        best_act = legal_actions_list[0]
        for action in legal_actions_list:
            child = state.clone()
            child.apply_action(action)
            val = self._expectimax(child, self.depth - 1)
            if val > best_val:
                best_val = val
                best_act = action
        return best_act


# ── Schedule ──────────────────────────────────────────────────────────────────

def guide_prob_by_step(step: int, cfg) -> float:
    """
    Xác suất dùng guide tại step hiện tại.
    Giảm tuyến tính từ 1.0 → min_prob trong decay_steps bước.
    """
    if not getattr(cfg, "guide", None) or not cfg.guide.enabled:
        return 0.0
    frac = min(1.0, step / max(1, cfg.guide.decay_steps))
    return max(cfg.guide.min_prob, 1.0 - frac)
