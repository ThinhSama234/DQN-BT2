import numpy as np


class Game2048Env:
    """
    Fast numpy 2048 environment.

    Observation: one-hot (4,4,18) flattened → 288 floats.
      obs[r*72 + c*18 + k] = 1 nếu cell(r,c) có giá trị 2^k (k=0 → empty).
    Actions: 0=Up  1=Right  2=Down  3=Left  (khớp OpenSpiel 2048).
    """

    ROWS = 4
    COLS = 4
    N_CH = 18  # channels: 0=empty, k=tile 2^k

    def __init__(self, seed: int = 42, reward_shaping_cfg=None):
        self.num_actions = 4
        self.obs_dim     = self.ROWS * self.COLS * self.N_CH  # 288
        self.rng         = np.random.default_rng(seed)
        self._shaping    = reward_shaping_cfg
        self.board       = np.zeros((4, 4), dtype=np.int32)
        self.score       = 0
        self._done       = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self._done = False
        self._spawn()
        self._spawn()
        return self._encode()

    def step(self, action: int):
        new_board, gained, changed = self._apply(action)
        if not changed:
            next_legal = self.legal_actions()
            return self._encode(), 0.0, self._done, {"legal_actions": next_legal}

        self.board  = new_board
        self.score += gained
        reward      = float(gained)

        if self._shaping and self._shaping.enabled:
            reward = self._shape(reward)

        self._spawn()
        next_legal = [a for a in range(4) if self._apply(a)[2]]
        self._done = len(next_legal) == 0
        obs = self._encode() if not self._done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, self._norm_reward(reward), self._done, {"legal_actions": next_legal}

    def legal_actions(self):
        return [a for a in range(4) if self._apply(a)[2]]

    @property
    def state(self):
        """Trả về board numpy — tương thích với code cũ dùng env.state."""
        return self.board

    # ── Internals ─────────────────────────────────────────────────────────────

    def _spawn(self):
        empties = np.argwhere(self.board == 0)
        if len(empties) == 0:
            return
        r, c = empties[self.rng.integers(len(empties))]
        self.board[r, c] = 2 if self.rng.random() < 0.9 else 4

    def _encode(self) -> np.ndarray:
        obs = np.zeros((4, 4, self.N_CH), dtype=np.float32)
        obs[:, :, 0] = (self.board == 0).astype(np.float32)
        rows, cols = np.where(self.board > 0)
        if len(rows):
            k = np.log2(self.board[rows, cols]).astype(np.int32)
            valid = k < self.N_CH
            obs[rows[valid], cols[valid], k[valid]] = 1.0
        return obs.reshape(-1)

    @staticmethod
    def _slide(row: np.ndarray):
        """Slide và merge 1 hàng sang trái. Trả về (new_row, score)."""
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

    def _apply(self, action: int):
        """Apply action lên bản copy của board. Trả về (new_board, score, changed)."""
        b = self.board.copy()
        score = 0

        if action == 0:    # Up   — slide từng cột lên = slide left trên cột
            for j in range(4):
                b[:, j], s = self._slide(b[:, j])
                score += s
        elif action == 2:  # Down — slide từng cột xuống = slide left trên cột đảo ngược
            for j in range(4):
                col, s = self._slide(b[::-1, j])
                b[::-1, j] = col
                score += s
        elif action == 3:  # Left
            for i in range(4):
                b[i], s = self._slide(b[i])
                score += s
        elif action == 1:  # Right — slide left trên hàng đảo ngược
            for i in range(4):
                row, s = self._slide(b[i, ::-1])
                b[i, ::-1] = row
                score += s

        changed = not np.array_equal(b, self.board)
        return b, score, changed

    # ── Reward shaping (optional) ─────────────────────────────────────────────

    @staticmethod
    def _norm_reward(r: float) -> float:
        """log2(r+1) nén reward range từ [4,131072] → [2.3,17], ổn định Q-value."""
        return float(np.log2(r + 1)) if r > 0 else 0.0

    def _shape(self, base_reward: float) -> float:
        cfg   = self._shaping
        bonus = 0.0
        empty = int(np.sum(self.board == 0))
        bonus += cfg.empty_cells_weight * empty
        max_val = int(self.board.max())
        if max_val > 1:
            corners = [self.board[0,0], self.board[0,3], self.board[3,0], self.board[3,3]]
            if max_val in corners:
                bonus += cfg.corner_weight * float(np.log2(max_val))
        if cfg.monotonicity_weight > 0:
            bonus += cfg.monotonicity_weight * self._monotonicity()
        return base_reward + bonus

    def _monotonicity(self) -> float:
        mono = 0
        for r in range(4):
            row = self.board[r]
            mono += sum(row[i] >= row[i+1] for i in range(3))
        for c in range(4):
            col = self.board[:, c]
            mono += sum(col[i] >= col[i+1] for i in range(3))
        return mono / 24.0


# Alias để tương thích với import cũ
OpenSpiel2048Env = Game2048Env
