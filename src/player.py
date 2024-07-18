from src.hand import Hand

class Player:
    def __init__(self, name: str, chips: int, bounty: int):
        """Initialize a player with a name, chips, and bounty."""
        self.name = name
        self.chips = chips
        self.bounty = bounty
        self.hand = Hand()
        self.has_folded = False
        self.current_bet = 0
    
    def __str__(self) -> str:
        """Return a string representation of the player."""
        return f"Player(name={self.name}, chips={self.chips}, bounty={self.bounty}, has_folded={self.has_folded})"
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the player for debugging."""
        return (f"Player(name='{self.name}', chips={self.chips}, bounty={self.bounty}, "
                f"hand={self.hand}, has_folded={self.has_folded}, current_bet={self.current_bet})")
    
    def bet(self, amount: int) -> int:
        """Place a bet of a given amount.
        
        Args:
            amount (int): The amount to bet.

        Returns:
            int: The actual amount bet.
        
        Raises:
            ValueError: If the bet amount is not positive or exceeds the player's chips.
        """
        if amount <= 0:
            raise ValueError("Bet amount must be positive")
        
        if self.chips >= amount:
            self.chips -= amount
            self.current_bet += amount
            return amount
        else:
            raise ValueError("Not enough chips")
    
    def call(self, amount: int) -> int:
        """Call a bet by matching the current highest bet.
        
        Args:
            amount (int): The amount to call.

        Returns:
            int: The actual amount called.
        """
        call_amount = amount - self.current_bet
        return self.bet(call_amount)
    
    def fold(self):
        """Fold the current hand."""
        self.hand.clear()
        self.has_folded = True
        self.current_bet = 0
    
    def check(self) -> int:
        """Check (pass the action without betting).
        
        Returns:
            int: The amount bet (always 0 for a check).
        """
        return 0
    
    def reset_fold(self):
        """Reset the fold status for a new round."""
        self.has_folded = False
        self.current_bet = 0
    
    def raise_bet(self, pot_size: int, custom_amount: int = None) -> int:
        """Raise the bet.
        
        Args:
            pot_size (int): The current size of the pot.
            custom_amount (int, optional): A custom raise amount. If not provided, defaults to double the pot size.

        Returns:
            int: The actual amount raised.
        """
        if custom_amount is not None:
            raise_amount = custom_amount
        else:
            raise_amount = 2 * pot_size

        return self.bet(raise_amount)
    
    def all_in(self) -> int:
        """Go all-in with the remaining chips.
        
        Returns:
            int: The amount of chips put into the pot.
        """
        all_in_amount = self.chips
        self.current_bet += self.chips
        self.chips = 0
        return all_in_amount

    def add_hole_cards(self, cards):
        """Add hole cards to the player's hand."""
        self.hand.add_hole_cards(cards)
