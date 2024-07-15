import src.player as Player

class Bot(Player):
    def __init__(self, name, chips, model, bounty):
        super().__init__(name, chips, bounty)
        
        self.model = model
        
        def decide(self, game_state):
            pass
        #entscheidungsfindung über das neuronale netzwerk implementieren