import unittest

import sys
import os

# add the src directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.hand import Hand
from src.card import Card

class TestHand(unittest.TestCase):
    def test_straight_flush(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Hearts'), Card('K', 'Hearts')])
        hand.add_community_cards([Card('Q', 'Hearts'), Card('J', 'Hearts'), Card('10', 'Hearts')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'Straight Flush')
        self.assertEqual(best_hand, [10, 11, 12, 13, 14])
    
    
    def test_four_of_a_kind(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Hearts'), Card('A', 'Spades')])
        hand.add_community_cards([Card('A', 'Diamonds'), Card('A', 'Clubs'), Card('2', 'Hearts')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'Four of a Kind')
        self.assertEqual(best_hand, [14, 2])
    
    def test_full_house(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Spades'), Card('2', 'Diamonds')])
        hand.add_community_cards([Card('2', 'Spades'), Card('2', 'Hearts'), Card('A', 'Clubs')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'Full House')
        self.assertEqual(best_hand, [2, 14])
        
        
    def test_flush(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Spades'), Card('K', 'Spades')])
        hand.add_community_cards([Card('Q', 'Spades'), Card('9', 'Spades'), Card('3', 'Spades')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'Flush')
        self.assertEqual(best_hand, [14, 13, 12, 9, 3])
        
    def test_straight(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Spades'), Card('K', 'Spades')])
        hand.add_community_cards([Card('Q', 'Diamonds'), Card('J', 'Clubs'), Card('10', 'Spades')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'Straight')
        self.assertEqual(best_hand, [10, 11, 12, 13, 14])
        
    def test_three_of_a_kind(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Spades'), Card('A', 'Spades')])
        hand.add_community_cards([Card('Q', 'Diamonds'), Card('A', 'Clubs'), Card('10', 'Spades')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'Three of a Kind')
        self.assertEqual(best_hand, [14, 12, 10])


    def test_two_pair(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Spades'), Card('K', 'Spades')])
        hand.add_community_cards([Card('Q', 'Diamonds'), Card('A', 'Clubs'), Card('K', 'Diamonds')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'Two Pair')
        self.assertEqual(best_hand, [14, 13, 12])
        
    def test_one_pair(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Spades'), Card('K', 'Spades')])
        hand.add_community_cards([Card('Q', 'Diamonds'), Card('A', 'Clubs'), Card('2', 'Diamonds')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'One Pair')
        self.assertEqual(best_hand, [14, 13, 12, 2])

    def test_high_card(self):
        hand = Hand()
        hand.add_hole_cards([Card('A', 'Spades'), Card('K', 'Spades')])
        hand.add_community_cards([Card('Q', 'Diamonds'), Card('6', 'Clubs'), Card('2', 'Diamonds')])
        hand_type, best_hand = hand.evaluate()
        self.assertEqual(hand_type, 'High Card')
        self.assertEqual(best_hand, [14, 13, 12, 6, 2])
        
if __name__ == '__main__':
    unittest.main()