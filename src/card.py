class Card:
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    
    def __init__(self, rank, suit):
        if rank not in Card.ranks:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in Card.suits:
            raise ValueError(f"Invalid suit: {suit}")
        
        self.rank = rank
        self.suit = suit
    
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
                return Card.suits.index(self.suit) < Card.suits.index(other.suit)
            return Card.ranks.index(self.rank) < Card.ranks.index(other.rank)
        return False
    
    def __le__(self, other):
        return self < other or self == other
    
    def __gt__(self, other):
        return not self <= other
    
    def __ge__(self, other):
        return not self < other