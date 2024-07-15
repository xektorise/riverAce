import src.card as Card
import random

class Deck:
    def __init__(self):
        suits = ['H', 'D', 'C', 'S']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]
        
        self.shuffle()
    
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self, num_cards):
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        
        return dealt_cards
    

    def reset(self):
        suits = ['H', 'D', 'C', 'S']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]
        self.shuffle()
