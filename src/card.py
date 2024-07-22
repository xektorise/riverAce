from dataclasses import dataclass

@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                   '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    def __post_init__(self):
        if self.rank not in self.ranks:
            raise ValueError(f"Invalid rank: {self.rank}")
        if self.suit not in self.suits:
            raise ValueError(f"Invalid suit: {self.suit}")
        
        # store indices for faster comparisons
        object.__setattr__(self, 'rank_index', self.ranks.index(self.rank))
        object.__setattr__(self, 'suit_index', self.suits.index(self.suit))

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
        return self.rank_values[self.rank]

    @property
    def color(self):
        return 'Red' if self.suit in ['Hearts', 'Diamonds'] else 'Black'