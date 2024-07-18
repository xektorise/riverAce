from player import Player
from card import Card
import numpy as np

class Bot(Player):
    def __init__(self, name, chips, model, bounty):
        """Initialize the bot with a name, chips, model, and bounty."""
        super().__init__(name, chips, bounty)
        self.model = model
    
    def decide(self, game_state):
        """Decide on an action based on the game state."""
        if self.model:
            state_vector = self.convert_state_to_vector(game_state)
            prediction = self.model.predict(state_vector.reshape(1, -1))
            action = np.argmax(prediction)
            return self.action_from_index(action)
        else:
            return self.strategy(game_state)
    
    def convert_state_to_vector(self, game_state):
        """Convert the game state to a feature vector for the model."""
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
        """Convert a list of Card objects to a vector of ranks and suits."""
        rank_order = {rank: index for index, rank in enumerate(Card.ranks, start=2)}
        suit_order = {suit: index for index, suit in enumerate(Card.suits, start=1)}
        
        vector = np.zeros(10)  # assuming a maximum of 5 cards (rank and suit pairs)
        for i, card in enumerate(cards):
            if i < 5:
                vector[2 * i] = rank_order[card.rank]
                vector[2 * i + 1] = suit_order[card.suit]
        
        return vector
    
    def strategy(self, game_state):
        """Fallback strategy when no model is provided."""
        current_bet = game_state['current_bet']
        community_cards = game_state['community_cards']
        player_hands = self.hand.hole_cards
        
        if not community_cards:  # Pre-flop strategy
            hand_strength = self.evaluate_starting_hand(player_hands)
            if hand_strength >= 8:
                return 'raise'
            elif hand_strength >= 5:
                return 'call'
            else:
                return 'fold'
        
        # post-flop strategy
        hand_strength = self.evaluate_hand_strength(player_hands + community_cards)
        if hand_strength >= 8:
            return 'raise'
        elif hand_strength >= 5:
            return 'call'
        else:
            return 'fold'
    
    def evaluate_starting_hand(self, hand):
        """Evaluate the strength of the starting hand."""
        ranks = [card.rank for card in hand]
        if ranks[0] == ranks[1]:  # pair
            return 10
        elif 'A' in ranks or 'K' in ranks:  # high cards
            return 7
        else:
            return 3
    
    def evaluate_hand_strength(self, hand):
        """Evaluate the strength of the hand after the flop."""
        ranks = [card.rank for card in hand]
        if len(set(ranks)) < len(ranks):  # at least one pair
            return 10
        elif 'A' in ranks or 'K' in ranks or 'Q' in ranks:  # high cards
            return 7
        else:
            return 3

    def action_from_index(self, index):
        """Convert model output index to action."""
        actions = ['fold', 'call', 'raise']
        return actions[index]