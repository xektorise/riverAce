import unittest

import sys
import os

# add the src directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.card import Card

class TestCard(unittest.TestCase):
    
    def test_card_initialization(self):
        card = Card('A', 'Spades')
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.suit, 'Spades')
    
    def test_card_equality(self):
        card1 = Card('A', 'Spades')
        card2 = Card('A', 'Spades')
        card3 = Card('K', 'Spades')
        self.assertEqual(card1, card2)
        self.assertNotEqual(card1, card3)


if __name__ == '__main__':
    unittest.main()