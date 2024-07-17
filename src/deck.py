import random
from card import Card

class Deck:
    def __init__(self):
        self.reset()
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self, num_cards):
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards
    
    def reset(self):
        self.cards = [Card(rank, suit) for suit in Card.suits for rank in Card.ranks]
        self.shuffle()
    
    def __str__(self):
        return ', '.join(str(card) for card in self.cards)
    
    def __repr__(self):
        return f"Deck(cards={self.cards})"
