import pyspiel
from helper import auto_resolve_chance_nodes, extract_obs
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
    
if __name__ == "__main__":
    env = OpenSpiel2048Env()
    obs = env.reset()
    print(obs)