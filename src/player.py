from src.hand import Hand

class Player:
    def __init__(self, name, chips, bounty):

        self.name = name
        self.chips = chips
        self.bounty = bounty
        self.hand = Hand()
        self.has_folded = False
        self.current_bet = 0
    
    def __str__(self):
        return f"Player(name={self.name}, chips={self.chips}, bounty={self.bounty}, has_folded={self.has_folded})"
    
    def __repr__(self):
        return f"Player(name='{self.name}', chips={self.chips}, bounty={self.bounty}, hand={self.hand}, has_folded={self.has_folded}, current_bet={self.current_bet})"
    
    def bet(self, amount):

        if amount <= 0:
            print("Bet amount must be positive")
            return 0
        
        if self.chips >= amount:
            self.chips -= amount
            self.current_bet += amount
            return amount
        else:
            print("Not enough chips")
            return 0
    
    def call(self, amount):
        call_amount = amount - self.current_bet
        return self.bet(call_amount)
    
    def fold(self):

        self.hand.clear()
        self.has_folded = True
        self.current_bet = 0
    
    def check(self):

        return 0
    
    def reset_fold(self):

        self.has_folded = False
        self.current_bet = 0
    
    def raise_bet(self, pot_size, custom_amount = None):
        
        if custom_amount is not None:
            raise_amount = custom_amount
        
        else:
            raise_amount = 2 * pot_size

        return self.bet(raise_amount)
    
    def all_in(self):
        
        all_in_amount = self.chips
        self.current_bet += self.chips
        self.chips = 0
        return all_in_amount