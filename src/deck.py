import random
from card import Card

class Deck:
    def __init__(self):
        """Initialize a new deck and shuffle it."""
        self.reset()
    
    def shuffle(self):
        """Shuffle the deck of cards."""
        random.shuffle(self.cards)
    
    def deal(self, num_cards):
        """Deal a specified number of cards from the deck.

        Args:
            num_cards (int): The number of cards to deal.

        Returns:
            list: A list of dealt cards.
        
        Raises:
            ValueError: If num_cards is greater than the number of cards left in the deck.
        """
        if num_cards > len(self.cards):
            raise ValueError("Cannot deal more cards than are in the deck.")
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards
    
    def reset(self):
        """Reset the deck to a full set of 52 cards and shuffle it."""
        self.cards = [Card(rank, suit) for suit in Card.suits for rank in Card.ranks]
        self.shuffle()
    
    def __str__(self):
        """Return a string representation of the deck."""
        return ', '.join(str(card) for card in self.cards)
    
    def __repr__(self):
        """Return a detailed string representation of the deck for debugging."""
        return f"Deck(cards={self.cards})"