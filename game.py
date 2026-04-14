import pyspiel
class Game():
    def __init__(self):
        """Initialize the game""" 
        self.game = pyspiel.load_game("2048")
        self.state = self.game.new_initial_state()