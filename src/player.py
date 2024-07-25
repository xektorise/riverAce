import json
from typing import List
from enum import Enum, auto
from hand import Hand
from card import Card

class PlayerState(Enum):
    ACTIVE = 1
    ALL_IN = 2
    FOLDED = 3
    BUST = 4

class Position(Enum):
    SMALL_BLIND = auto()
    BIG_BLIND = auto()
    UNDER_THE_GUN = auto()
    MIDDLE_POSITION = auto()
    CUTOFF = auto()
    BUTTON = auto()


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
        self.position: Position = None
        self.hand_history: List[Hand] = []
        self.action_history: List[str] = []
        self.tilt_factor: float = 1.0
        self.state = PlayerState.ACTIVE
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
        if amount <= 0:
            raise ValueError("Bet amount must be positive")
        if amount > self.chips:
            raise ValueError(f"Bet amount {amount} exceeds available chips {self.chips}")
        self.chips -= amount
        self.current_bet += amount
        self.total_bet += amount
        return amount
    
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
        all_in_amount = self.chips
        self.current_bet += self.chips
        self.total_bet += self.chips
        self.chips = 0
        self.state = PlayerState.ALL_IN
        return all_in_amount
    

    def post_blind(self, amount: int) -> None:
        """Post a blind bet."""
        self.bet(amount)

    def win_pot(self, amount: int) -> None:
        """Handle winning a pot."""
        self.add_chips(amount)
        self.update_stats(won=True, profit=amount, pot_size=amount)


    def add_hole_cards(self, cards: List[Card]) -> None:
        """Add hole cards to the player's hand."""
        self.hand.add_hole_cards(cards)

    def update_hand_with_community_cards(self, community_cards: List[Card]) -> None:
        """Update the player's hand with community cards."""
        self.hand.add_community_cards(community_cards)

    def get_total_bet(self) -> int:
        """Get the total amount bet by the player in the current round."""
        return self.total_bet
    
    def get_current_position(self) -> Position:
        return self.position
    
    def set_position(self, position: Position) -> None:
        """Set players position at the table."""
        self.position = position
    
    def get_position_category(self) -> str:
        """Get a simplified category for player's position"""
        if self.position in (Position.SMALL_BLIND, Position.BIG_BLIND, Position.UNDER_THE_GUN):
            return "early"
        if self.position == Position.MIDDLE_POSITION:
            return "middle"
        if self.position in (Position.CUTOFF, Position.BUTTON):
            return "late"
        else:
            return "unknown"
        
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
        self.tilt_factor = max(0.5, min(2.0, self.tilt_factor))

    def reset_tilt_factor(self) -> None:
        self.tilt_factor = 1.0
    
    def update_state(self) -> None:
        """Update the player's state based on their current situation"""
        if self.chips == 0 and self.current_bet > 0:
            self.state = PlayerState.ALL_IN
        elif self.has_folded:
            self.state = PlayerState.FOLDED
        elif self.chips == 0 and self.current_bet == 0:
            self.state = PlayerState.BUST
        else: 
            self.state = PlayerState.ACTIVE
    
    def calculate_spr(self, pot_size: int) -> float:
        """Calculate pot-to-stack ratio"""
        return self.chips / pot_size if pot_size > 0 else float('inf')
    
    def calculate_m_ratio(self, total_blinds: int) -> float:
        """Calculate the player's M-ratio (chip stack / total blinds)"""
        return self.chips / total_blinds if total_blinds > 0 else float('inf')
    
    def calculate_effective_stack(self, opponent_chips: int) -> int:
        """Calculate the effective stack(the smaller of this player's and opponent's stack)"""
        return min(self.chips, opponent_chips)

    
    def calculate_aggression_factor(self) -> float:
        """Calculate player's aggression factor"""
        bets_and_raises = self.action_history.count('bet') + self.action_history.count('raise')
        calls = self.action_history.count('call')
        return bets_and_raises / calls if calls > 0 else float('inf')
    
    def calculate_roi(self) -> float:
        """Calculate the player's roi (return of investment)"""
        total_buyins = self.stats['hands_played'] * (self.chips + self.bounty)
        return (self.stats['total_profit'] / total_buyins) * 100 if total_buyins > 0 else 0
    
    def calculate_vpip(self) -> float:
        """Calculate the player's vpip (voluntarily put money in pot)"""
        voluntary_actions = sum(1 for action in self.action_history if action in ['bet', 'raise', 'call'])
        return (voluntary_actions / len(self.action_history)) * 100 if self.action_history else 0
    
    def calculate_pfr(self) -> float:
        """Calculate pfr (pre-flop raise) percentage"""
        preflop_raises = sum(1 for action in self.action_history if action == 'raise')
        return (preflop_raises / self.stats['hands_played']) * 100 if self.stats['hands_played'] > 0 else 0
    
    def calculate_pot_odds(self, pot_size: int, current_bet: int) -> float:
        """Calculate pot odds"""

        call_amount = current_bet - self.current_bet

        if call_amount == 0:
            return 0.0
        
        return call_amount / (pot_size + call_amount)
    
    def get_range(self) -> str:
        """Get player's estimated range based on position and actions"""
        position_category =self.get_position_category()
        if position_category == "early":
            return "Premium hands only"
        elif position_category == "middle":
            return "Strong to medium hands"
        elif position_category == "late":
            return "wide range of hands"
        else:
            return "Unknown range"
    
    def add_chips(self, amount: int) -> None:
        """Add chips to the player's stack after winning a pot"""
        self.chips += amount

    def get_stack_size(self) -> int:
        """Get the current player's stack size"""
        return self.chips
    
    def get_win_rate(self) -> float:
        """Get the player's winrate"""
        return (self.stats['hands_won'] / self.stats['hands_played']) * 100 if self.stats['hands_played'] > 0 else 0
    
    def get_current_state(self) -> PlayerState:
        """Get player's current state"""
        return self.state
    

    
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
            'position' : self.position if self.position else None,
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
        player.position = Position[data['position']] if data['position'] else None
        player.stats = data['stats']
        player.tilt_factor = data['tilt_factor']
        player.state = PlayerState(data['state'])
        return player
        