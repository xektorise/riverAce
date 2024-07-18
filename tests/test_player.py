import unittest
import os
import sys

# add the src directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.hand import Hand
from src.player import Player

class TestPlayer(unittest.TestCase):
    
    def test_player_initialization(self):
        player = Player('Roman', 5000, 100)
        self.assertEqual(player.name, 'Roman')
        self.assertEqual(player.chips, 5000)
        self.assertEqual(player.bounty, 100)
        self.assertFalse(player.has_folded)
        self.assertIsInstance(player.hand, Hand)
        self.assertEqual(player.current_bet, 0)
    
    def test_player_bet(self):
        player = Player('Roman', 5000, 100)
        bet_amount = player.bet(200)
        self.assertEqual(bet_amount, 200)
        self.assertEqual(player.chips, 4800)
        self.assertEqual(player.current_bet, 200)
        
        with self.assertRaises(ValueError):
            player.bet(4900)
    
    def test_player_call(self):
        player = Player('Roman', 5000, 100)
        player.current_bet = 100
        call_amount = player.call(300)
        self.assertEqual(call_amount, 200)
        self.assertEqual(player.chips, 4800)
        self.assertEqual(player.current_bet, 300)
    
    def test_player_fold(self):
        player = Player('Roman', 5000, 100)
        player.fold()
        self.assertTrue(player.has_folded)
        self.assertEqual(player.current_bet, 0)
        self.assertEqual(len(player.hand.hole_cards), 0)
        self.assertEqual(len(player.hand.community_cards), 0)
    
    def test_player_check(self):
        player = Player('Roman', 5000, 100)
        self.assertEqual(player.check(), 0)
    
    def test_player_raise_bet(self):
        player = Player('Roman', 5000, 100)
        raise_amount = player.raise_bet(400)
        self.assertEqual(raise_amount, 800)
        self.assertEqual(player.chips, 4200)
        self.assertEqual(player.current_bet, 800)
        
        player.current_bet = 0
        player.chips = 1000
        custom_raise_amount = player.raise_bet(300, 500)
        self.assertEqual(custom_raise_amount, 500)
        self.assertEqual(player.chips, 500)
        self.assertEqual(player.current_bet, 500)
    
    def test_player_all_in(self):
        player = Player('Roman', 5000, 100)
        all_in_amount = player.all_in()
        self.assertEqual(all_in_amount, 5000)
        self.assertEqual(player.current_bet, 5000)
        self.assertEqual(player.chips, 0)
    
if __name__ == '__main__':
    unittest.main()
