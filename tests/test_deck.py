import unittest

import sys
import os

# add the src directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.deck import Deck
from src.card import Card

class TestDeck(unittest.TestCase):
    
    def test_deck_initialization(self):
        deck = Deck()
        self.assertEqual(len(deck.cards), 52)
    
    def test_deck_shuffle(self):
        deck = Deck()
        original_order = deck.cards[:]
        deck.shuffle()
        self.assertNotEqual(deck.cards, original_order)
    
    
    def test_deal_cards(self):
        deck = Deck()
        dealt_cards = deck.deal(5)
        self.assertEqual(len(dealt_cards), 5)
        self.assertEqual(len(deck.cards), 47)
    
    def test_deck_reset(self):
        deck = Deck()
        deck.deal(5)
        deck.reset()
        self.assertEqual(len(deck.cards), 52)

if __name__ == '__main__':
    unittest.main()