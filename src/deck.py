import random
from typing import List
from card import Card, Rank, Suit

class Deck:
    FULL_DECK_SIZE = 52

    def __init__(self) -> None:
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
    
    def probability_of_card(self, rank: Rank = None, suit: Suit = None) -> float:
        """
        Calculate the probability of drawing a card with a specific rank or suit.
        
        Args:
            rank (Rank, optional): The rank of the card.
            suit (Suit, optional): The suit of the card.
        
        Returns:
            float: The probability of drawing the specified card.
        """
        remaining_cards = len(self.cards)
        if rank and suit:
            matching_cards = sum(1 for card in self.cards if card.rank == rank and card.suit == suit)
        elif rank:
            matching_cards = sum(1 for card in self.cards if card.rank == rank)
        elif suit:
            matching_cards = sum(1 for card in self.cards if card.suit == suit)
        else:
            return 1.0
        return matching_cards / remaining_cards
    
    def peek(self, num_cards: int = 1) -> List[Card]:
        """Look at the top cards of the deck without removing them"""
        if num_cards > len(self.cards):
            raise ValueError("Cannot peek more cards than are in the deck.")
        return self.cards[:num_cards]
    
    def reset(self) -> None:
        """Reset the deck to a full set of 52 cards and shuffle it."""
        self.cards = [Card(rank, suit) for suit in Suit for rank in Rank]
        self.shuffle()
    
    
    def get_dealt_cards(self) -> List[Card]:
        """Return a list of cards that have been dealt."""
        all_cards = set([Card(rank, suit) for suit in Suit for rank in Rank])
        return list(all_cards - set(self.cards))
    
    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self.cards) == 0
    
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