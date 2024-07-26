from dataclasses import dataclass
from enum import Enum, auto



class Suit(Enum):
    HEARTS = 'H'
    DIAMONDS = 'D'
    CLUBS = 'C'
    SPADES = 'S'

class Rank(Enum):
    TWO = '2'
    THREE = '3'
    FOUR = '4'
    FIVE = '5'
    SIX = '6'
    SEVEN = '7'
    EIGHT = '8'
    NINE = '9'
    TEN = 'T'
    JACK = 'J'
    QUEEN = 'Q'
    KING = 'K'
    ACE = 'A'
    

@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit
    
    rank = list(Rank)
    suit = list(Suit)
    rank_values = {r.value: i + 2 for i, r in enumerate(Rank)}

    def __post_init__(self):
        if not isinstance(self.rank, Rank) or not isinstance(self.suit, Suit):
            raise ValueError(f"Invalid rank or suit: {self.rank}, {self.suit}")
        
        # store indices for faster comparisons
        object.__setattr__(self, 'rank_index', list(Rank).index(self.rank))
        object.__setattr__(self, 'suit_index', list(Suit).index(self.suit))

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __repr__(self):
        return f"Card(rank='{self.rank}', suit='{self.suit}')"

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False

    def __lt__(self, other):
        if isinstance(other, Card):
            if self.rank == other.rank:
                return self.suit_index < other.suit_index
            return self.rank_index < other.rank_index
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __hash__(self):
        return hash((self.rank, self.suit))

    def get_value(self):
        return self.rank_values[self.rank.value]
    
    def to_poker_notation(self):
        return f"{self.rank.value}{self.suit.value}"
    
    def is_face_card(self):
        return self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]

    def is_ace(self):
        return self.rank == Rank.ACE
    
    @classmethod
    def from_string(cls, card_string: str):
        if len(card_string) != 2:
            raise ValueError("Card string must be 2 characters long")
        rank, suit = card_string[0], card_string[1]
        return cls(Rank(rank), Suit(suit))

    @property
    def color(self):
        return 'Red' if self.suit in [Suit.HEARTS, Suit.DIAMONDS] else 'Black'