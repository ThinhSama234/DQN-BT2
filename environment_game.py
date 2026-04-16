import numpy as np
import pyspiel
from helper import auto_resolve_chance_nodes, extract_obs, legal_actions, state_return, parse_board_numbers


class OpenSpiel2048Env():
    def __init__(self, seed=42, reward_shaping_cfg=None):
        self.game        = pyspiel.load_game('2048')
        self.player_id   = 0
        self.num_actions = self.game.num_distinct_actions()
        self.obs_dim     = self.game.observation_tensor_size()
        self.rng         = np.random.default_rng(seed)
        self.state       = None
        self._shaping    = reward_shaping_cfg   # None hoặc RewardShapingConfig
    def reset(self, seed = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.game.new_initial_state()
        auto_resolve_chance_nodes(self.state, self.rng)
        return extract_obs(self.state, self.player_id)
    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset before run action")
        if self.state.is_terminal():
            raise RuntimeError("Episode ended. Please Reset Game")
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action {action}, legal action {legal}")
        prev_return = state_return(self.state)
        self.state.apply_action(int(action))
        auto_resolve_chance_nodes(self.state, self.rng)
        next_obs = extract_obs(self.state, self.player_id) if not self.state.is_terminal() else np.zeros(self.obs_dim, dtype = np.float32)
        new_return = state_return(self.state)
        reward = new_return - prev_return
        done   = self.state.is_terminal()

        if self._shaping is not None and self._shaping.enabled and not done:
            reward = self._shape(reward)

        info = {"legal_actions": self.legal_actions()}
        return next_obs, float(reward), done, info

    # ── Reward Shaping ────────────────────────────────────────────────────────
    def _shape(self, base_reward: float) -> float:
        board = parse_board_numbers(self.state)
        if board is None:
            return base_reward

        cfg   = self._shaping
        bonus = 0.0

        # Bonus ô trống — nhiều ô trống = nhiều lựa chọn hơn
        empty  = int(np.sum(board == 0))
        bonus += cfg.empty_cells_weight * empty

        # Bonus max tile ở góc
        max_val = int(board.max())
        if max_val > 1:
            corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
            if max_val in corners:
                bonus += cfg.corner_weight * float(np.log2(max_val))

        # Bonus đơn điệu — khuyến khích sắp xếp từ lớn → nhỏ
        if cfg.monotonicity_weight > 0:
            bonus += cfg.monotonicity_weight * self._monotonicity(board)

        return base_reward + bonus

    @staticmethod
    def _monotonicity(board: np.ndarray) -> float:
        """Tỉ lệ cặp liền kề đơn điệu (hàng + cột), trong [0, 1]."""
        mono = 0
        for r in range(4):
            row = board[r]
            mono += sum(row[i] >= row[i + 1] for i in range(3))
        for c in range(4):
            col = board[:, c]
            mono += sum(col[i] >= col[i + 1] for i in range(3))
        return mono / 24.0
    
    def legal_actions(self):
        if self.state is None or self.state.is_terminal():
            return []
        return legal_actions(self.state, self.player_id)
    
if __name__ == "__main__":
    env = OpenSpiel2048Env()
    obs = env.reset()
    print(obs)