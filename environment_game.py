import numpy as np
import pyspiel
from helper import auto_resolve_chance_nodes, extract_obs, legal_actions, state_return
class OpenSpiel2048Env():
    def __init__(self, seed = 42):
        # load lib
        self.game = pyspiel.load_game('2048')
        self.player_id = 0
        self.num_actions = self.game.num_distinct_actions()
        self.obs_dim = self.game.observation_tensor_size()
        self.rng = np.random.default_rng(seed)
        self.state = None
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
        done = self.state.is_terminal()
        info = {"legal_actions": self.legal_actions()}
        return next_obs, float(reward), done, info
    
    def legal_actions(self):
        if self.state is None or self.state.is_terminal():
            return []
        return legal_actions(self.state, self.player_id)
    
if __name__ == "__main__":
    env = OpenSpiel2048Env()
    obs = env.reset()
    print(obs)