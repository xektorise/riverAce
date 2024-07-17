from player import Player
from card import Card
import numpy as np

class Bot(Player):
    def __init__(self, name, chips, model, bounty):
        super().__init__(name, chips, bounty)
        self.model = model
    
    def decide(self, game_state):
        if self.model:
            state_vector = self.convert_state_to_vector(game_state)
            prediction = self.model.predict(state_vector.reshape(1, -1))
            action = np.argmax(prediction)
            return action
        else:
            return self.strategy(game_state)
    
    def convert_state_to_vector(self, game_state):
        player_chips = game_state['player_chips'][self.name]
        community_cards = self._convert_cards_to_vector(game_state['community_cards'])
        player_hands = self._convert_cards_to_vector(self.hand.hole_cards)
        pot_size = game_state['pot_size']
        current_bet = game_state['current_bet']
        position = game_state['player_positions'][self.name]
        bounty = self.bounty
        
        state_vector = np.concatenate([
            np.array([player_chips, pot_size, current_bet, position, bounty]),
            community_cards,
            player_hands
        ])
        
        return state_vector
    
    def _convert_cards_to_vector(self, cards):
        rank_order = {rank: index for index, rank in enumerate(Card.ranks, start=2)}
        suit_order = {suit: index for index, suit in enumerate(Card.suits, start=1)}
        
        vector = np.array([
            [rank_order[card.rank], suit_order[card.suit]] for card in cards
        ]).flatten()
        
        return vector
    
    def strategy(self, game_state):
        # extract game information
        current_bet = game_state['current_bet']
        community_cards = game_state['community_cards']
        player_hands = self.hand.hole_cards
        
        # imple pre-flop strategy
        if not community_cards:  # pre-flop
            hand_strength = self.evaluate_starting_hand(player_hands)
            if hand_strength >= 8:
                return 'raise'
            elif hand_strength >= 5:
                return 'call'
            else:
                return 'fold'
        
        # simple post-flop strategy
        hand_strength = self.evaluate_hand_strength(player_hands + community_cards)
        if hand_strength >= 8:
            return 'raise'
        elif hand_strength >= 5:
            return 'call'
        else:
            return 'fold'
    
    def evaluate_starting_hand(self, hand):
        ranks = [card.rank for card in hand]
        if ranks[0] == ranks[1]:  # pair
            return 10
        elif 'A' in ranks or 'K' in ranks:  # high cards
            return 7
        else:
            return 3
    
    def evaluate_hand_strength(self, hand):
        ranks = [card.rank for card in hand]
        if len(set(ranks)) < len(ranks):  # at least one pair
            return 10
        elif 'A' in ranks or 'K' in ranks or 'Q' in ranks:  # high cards
            return 7
        else:
            return 3

# Example usage
# Assuming you have a trained model and a game_state dictionary
# model = your_trained_model
game_state = {
    'pot_size': 150,
    'community_cards': [Card('2', 'Hearts'), Card('7', 'Clubs'), Card('A', 'Spades')],
    'current_bet': 50,
    'player_chips': {'AI Bot': 1000},
    'player_positions': {'AI Bot': 1}
}

bot = Bot(name="AI Bot", chips=1000, model=None, bounty=50)
bot.add_hole_cards([Card('A', 'Hearts'), Card('K', 'Hearts')])
action = bot.decide(game_state)
print(f"Bot action: {action}")
