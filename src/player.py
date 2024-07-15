from hand import Hand

class Player:
    def __init__(self, name, chips, bounty):
        self.name = name
        self.chips = chips
        self.bounty = bounty
        self.hand = Hand()
        self.has_folded = False
        self.current_bet = 0
        
    def bet(self, amount):
        if self.chips >= amount:
            self.chips -= amount
            self.current_bet += amount
            return amount
        else:
            print("Not enough chips")
            return 0
        
    def fold(self):
        self.hand.clear()
        self.has_folded = True
        self.current_bet = 0
    
    def check(self):
        return 0
        
    def reset_fold(self):
        self.has_folded = False
        self.current_bet = 0
        
    def raise_bet(self, amount):
        return self.bet(amount)
