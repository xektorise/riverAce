import random
from typing import List
from card import Card

class Deck:
    FULL_DECK_SIZE = 52

    def __init__(self):
        """Initialize a new deck and shuffle it."""
        self.reset()
    
    def shuffle(self, seed: int = None) -> None:
        """
        Shuffle the deck of cards.
        
        Args:
            seed (int, optional): Seed for the random shuffle. Useful for testing.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)
    
    def deal(self, num_cards: int) -> List[Card]:
        """
        Deal a specified number of cards from the deck.

        Args:
            num_cards (int): The number of cards to deal.

        Returns:
            List[Card]: A list of dealt cards.
        
        Raises:
            ValueError: If num_cards is greater than the number of cards left in the deck.
        """
        if num_cards > len(self.cards):
            raise ValueError("Cannot deal more cards than are in the deck.")
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards
    
    def reset(self) -> None:
        """Reset the deck to a full set of 52 cards and shuffle it."""
        self.cards = [Card(rank, suit) for suit in Card.suits for rank in Card.ranks]
        self.shuffle()
    
    def __str__(self) -> str:
        """Return a string representation of the deck."""
        return ', '.join(str(card) for card in self.cards)
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the deck for debugging."""
        return f"Deck(cards={self.cards})"
    
    def __len__(self) -> int:
        """Return the number of cards currently in the deck."""
        return len(self.cards)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()