import unittest
import sys
import os
from unittest.mock import MagicMock

# add the src directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.card import Card
from src.player import Player
from src.game import Game

class TestGame(unittest.TestCase):
    
    def setUp(self):
        self.players = [
            Player('Roman', 5000, 100),
            Player('Alice', 5000, 100),
            Player('Bob', 5000, 100)
        ]
        
        self.game = Game(self.players, small_blind=10, big_blind=20)
    
    def test_deal_hands(self):
        self.game.deal_hands()
        for player in self.players:
            self.assertEqual(len(player.hand.hole_cards), 2)
    
    
    def test_post_blinds(self):
        self.game.post_blinds()
        self.assertEqual(self.players[1].current_bet, 10)
        self.assertEqual(self.players[2].current_bet, 20)
        self.assertEqual(self.players[1].chips, 4990)
        self.assertEqual(self.players[2].chips, 4980)
        self.assertEqual(self.game.pot, 0)
        self.assertEqual(self.game.current_bet, 20)
    
    
    def test_collect_bet(self):
        self.game.post_blinds()
        print("Pot before collecting bets:", self.game.pot)
        self.game.collect_bets()
        print("Pot after collecting bets:", self.game.pot)
        self.assertEqual(self.game.pot, 30)
        for player in self.players:
            self.assertEqual(player.current_bet, 0)
    
    def test_betting_rounds_all_check(self):
        for player in self.players:
            player.decide = MagicMock(return_value = 'check')
        
        self.game.betting_rounds()
        
        for player in self.players:
            self.assertEqual(player.current_bet, 0)
            self.assertFalse(player.has_folded)
        
if __name__ == '__main__':
    unittest.main()
