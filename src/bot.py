import src.player as Player
import numpy as np

class Bot(Player):
    def __init__(self, name, chips, model, bounty):
        super().__init__(name, chips, bounty)
        
        self.model = model
        
        def decide(self, game_state):

            state_vector = self.convert_state_to_vector(game_state)

            prediction = self.model.predict(state_vector.reshape(1, -1))

            action = np.argmax(prediction)

            return action
        

        def convert_state_to_vector(self, game_state):
            state_vector = np.array([
                game_state.player_chips,
                game_state.community_cards,
                game_state.player_hands,
                game_state.pot_size,
                game_state.current_bet,
                game_state.position,
                game_state.bounty
            ])
            
            return state_vector