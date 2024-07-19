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
    
    
    def test_collect_bets(self):
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
        
        self.game.betting_round()
        
        for player in self.players:
            self.assertEqual(player.current_bet, 0)
            self.assertFalse(player.has_folded)
    
    
    def test_deal_community_cards(self):
        self.game.deal_community_cards(3)
        self.assertEqual(len(self.game.community_cards), 3)
        self.game.deal_community_cards(2)
        self.assertEqual(len(self.game.community_cards), 5)
    
    def test_move_dealer(self):
        initial_dealer_index = self.game.dealer_index
        self.game.move_dealer()
        self.assertEqual(self.game.dealer_index, (initial_dealer_index + 1) % len(self.players))
    
    def test_determine_winner(self):
        self.game.deal_hands()
        
        self.players[0].hand.hole_cards = [Card('A', 'Clubs'), Card('K', 'Hearts')]
        self.players[1].hand.hole_cards = [Card('6', 'Diamonds'), Card('7', 'Diamonds')]
        self.players[2].hand.hole_cards = [Card('Q', 'Diamonds'), Card('J', 'Clubs')]
        
        for player in self.players:
            player.bet(300)
        
        self.game.pot = 900
        
        self.game.community_cards = [Card('10', 'Hearts'), Card('A', 'Diamonds'), Card('7', 'Clubs'), Card('2', 'Spades'), Card('K', 'Spades')]
        
        self.players[0].hand.evaluate = MagicMock(return_value = 9)
        self.players[1].hand.evaluate = MagicMock(return_value = 5)
        self.players[2].hand.evaluate = MagicMock(return_value = 8)
        
        self.game.determine_winner()
        
        self.assertEqual(self.players[0].chips, 5600)
        self.assertEqual(self.players[1].chips, 4700)
        self.assertEqual(self.players[2].chips, 4700)
            
    
    def test_split_pot(self):
        self.game.pot = 90
        winners = [self.players[0], self.players[1]]
        self.game.split_pot(winners)
        
        self.assertEqual(self.players[0].chips, 5045)
        self.assertEqual(self.players[1].chips, 5045)
    
    
    def test_remove_player(self):
        self.game.remove_player(self.players[1])
        self.assertEqual(len(self.game.players), 2)
        self.assertNotIn(self.players[1], self.game.players)
    
    def test_hande_side_pots(self):
        self.players[0].current_bet = 200
        self.players[1].current_bet = 300
        self.players[2].current_bet = 400
        
        all_in_player = self.players[0]
        all_in_amount = 200
        
        self.game.handle_side_pots(all_in_player, all_in_amount)
        
        self.assertEqual(len(self.game.side_pots), 2)
        self.assertEqual(self.game.side_pots[0], (100, self.players[1]))
        self.assertEqual(self.game.side_pots[1], (200, self.players[2]))
    
    
if __name__ == '__main__':
    unittest.main()
