import json
from typing import List
from enum import Enum
from src.hand import Hand
from src.card import Card

class PLAYERSTATE(Enum):
    ACTIVE = 1
    ALL_IN = 2
    FOLDED = 3
    BUST = 4

class Player:
    def __init__(self, name: str, chips: int, bounty: int):
        """Initialize a player with a name, chips, and bounty."""
        self.name = name
        self.chips = chips
        self.bounty = bounty
        self.hand = Hand()
        self.has_folded = False
        self.current_bet = 0
        self.total_bet = 0
        self.position: Optional[str] = None
        self.hand_history: List[Hand] = []
        self.action_history: List[str] = []
        self.tilt_factor: float = 1.0
        self.state = PLAYERSTATE.ACTIVE
        self.stats = {
            'hands_played' : 0,
            'hands_won' : 0,
            'total_profit' : 0,
            'biggest_pot_win' : 0,
        }
    
    def __str__(self) -> str:
        """Return a string representation of the player."""
        return f"Player(name={self.name}, chips={self.chips}, bounty={self.bounty}, has_folded={self.has_folded})"
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the player for debugging."""
        return (f"Player(name='{self.name}', chips={self.chips}, bounty={self.bounty}, "
                f"hand={self.hand}, has_folded={self.has_folded}, current_bet={self.current_bet}, "
                f"total_bet={self.total_bet})")
    
    def __eq__(self, other):
        """Check if two players are equal."""
        if isinstance(other, Player):
            return self.name == other.name and self.chips == other.chips and self.bounty == other.bounty
        return False
    
    def bet(self, amount: int) -> int:
        """Place a bet of a given amount."""
        if amount <= 0:
            raise ValueError("Bet amount must be positive")
        
        if self.chips >= amount:
            self.chips -= amount
            self.current_bet += amount
            self.total_bet += amount
            return amount
        else:
            raise ValueError("Not enough chips")
    
    def call(self, amount: int) -> int:
        """Call a bet by matching the current highest bet."""
        call_amount = amount - self.current_bet
        return self.bet(call_amount)
    
    def fold(self) -> None:
        """Fold the current hand."""
        self.hand.clear()
        self.has_folded = True
    
    def check(self) -> int:
        """Check (pass the action without betting)."""
        return 0
    
    def reset_for_new_round(self) -> None:
        """Reset the player's state for a new round."""
        self.has_folded = False
        self.current_bet = 0
        self.total_bet = 0
        self.hand.clear()
    
    def raise_bet(self, pot_size: int, current_bet: int, custom_amount: int = None) -> int:
        """Raise the bet."""
        if custom_amount is not None:
            raise_amount = max(custom_amount, 2 * current_bet)
        else:
            raise_amount = max(2 * current_bet, pot_size)

        return self.bet(raise_amount - self.current_bet)
    
    def all_in(self) -> int:
        """Go all-in with the remaining chips."""
        all_in_amount = self.chips
        self.current_bet += self.chips
        self.total_bet += self.chips
        self.chips = 0
        return all_in_amount

    def add_hole_cards(self, cards: List[Card]) -> None:
        """Add hole cards to the player's hand."""
        self.hand.add_hole_cards(cards)

    def update_hand_with_community_cards(self, community_cards: List[Card]) -> None:
        """Update the player's hand with community cards."""
        self.hand.add_community_cards(community_cards)

    def get_total_bet(self) -> int:
        """Get the total amount bet by the player in the current round."""
        return self.total_bet
    
    def set_position(self, position: str) -> None:
        """Set players position at the table."""
        self.position = position
        
    def add_hand_to_history(self) -> None:
        self.hand_history.append(self.hand.copy())
    
    def update_stats(self, won: bool, profit: int, pot_size: int) -> None:
        """Update player statistics after a hand"""
        self.stats['hands_played'] += 1
        if won:
            self.stats['hands_won'] += 1
        self.stats['total_profit'] += profit
        self.stats['biggest_pot_won'] = max(self.stats['biggest_pot_won'], pot_size)
    
    def record_action(self, action: str) -> None:
        """Record an action taken by each player"""
        self.action_history.append(action)
        
    def clear_action_history(self) -> None:
        """Clear action history for a new hand"""
        self.action_history.clear()
    
    def adjust_tilt_factor(self, factor: float) -> None:
        """Adjust player's tilt factor"""
        self.tilt_factor *= factor
        self.tilt_factor = max(0.5, min(2,0, self.tilt_factor))
    
    def update_state(self) -> None:
        """Update the player's state based on their current situation"""
        if self.chips == 0 and self.current_bet > 0:
            self.state = PLAYERSTATE.ALL_IN
        elif self.has_folded:
            self.state = PLAYERSTATE.FOLDED
        elif self.chips == 0 and self.current_bet == 0:
            self.state = PLAYERSTATE.BUST
        else:
            self.state = PLAYERSTATE.ACTIVE
    
    def decide(self, game_state: dict, options: List[str]) -> str:
        """
        Make a decision based on the current game state and available options.
        This method will be overridden in a Bot subclass for AI players.
        """
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def to_dict(self) -> dict:
        """Convert the player object to a dictionary for serialization."""
        return {
            'name' : self.name,
            'chips' : self.chips,
            'bounty' : self.bounty,
            'has_folded' : self.has_folded,
            'current_bet' : self.current_bet,
            'total_bet' : self.total_bet,
            'position' : self.position,
            'stats' : self.stats,
            'tilt_factor' : self.tilt_factor,
            'state' : self.state.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Player':
        """Create a Player Object from dictionary"""
        player = cls(data['name'], data['chips'], data['bounty'])
        player.has_folded = data['has_folded']
        player.current_bet = data['current_bet']
        player.total_bet = data['total_bet']
        player.position = data['position']
        player.stats = data['stats']
        player.tilt_factor = data['tilt_factor']
        player.state = PLAYERSTATE(data['state'])
        return player
        