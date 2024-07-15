from game import Game
from player import Player

class Tournament:
    def __init__(self, players, small_blind, big_blind):
        self.players = players
        self.tables = []
        self.small_blind = small_blind
        self.big_blind = big_blind
    
    
    def create_tables(self):
        self.tables = []
        num_tables = (len(self.players + 7) // 8)
        
        for i in range(num_tables):
            table_players = self.players[i*8:(i+1)*8]
            self.tables.append(Game(table_players, self.small_blind, self.big_blind))
    
    
    def play_tournament(self):
        while len(self.players) > 8:
            self.play_round()
            self.consolidate_tables()
            
            
        final_table = Game(self.players, self.big_blind, self.small_blind)
        while len([player for player in self.players if player.chips > 0]) > 1:
            final_table.play_round()
    
    def play_round(self):
        for table in self.tables:
            table.play_round()
            table.remove_player()
            
    
    
    def consolidate_tables(self):
        remaining_players = []
        for table in self.tables:
            remaining_players.extend(table.players)
        self.players = remaining_players
        self.create_tables()