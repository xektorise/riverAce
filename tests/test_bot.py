import unittest
from unittest.mock import MagicMock
import sys
import os

# add the src directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.card import Card
from src.player import Player
from src.bot import Bot

class TestBot(unittest.TestCase):

    def setUp(self):
        self.bot = Bot('AI', 5000, None, 100)
        self.game_state = {
            'player_chips': {'AI': 5000, 'Player1': 5000, 'Player2': 5000},
            'community_cards': [Card('10', 'Hearts'), Card('A', 'Diamonds'), Card('7', 'Clubs')],
            'pot_size': 150,
            'current_bet': 50,
            'player_positions': {'AI': 0, 'Player1': 1, 'Player2': 2},
            'bounties': {'AI': 100, 'Player1': 100, 'Player2': 100}
        }

    def test_bot_decision_with_model(self):
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[[0.1, 0.3, 0.6]])  # Assume 'raise' is the highest probability
        self.bot.model = mock_model

        action = self.bot.decide(self.game_state)
        self.assertEqual(action, 'raise')

    def test_bot_decision_without_model(self):
        action = self.bot.decide(self.game_state)
        self.assertIn(action, ['call', 'raise', 'fold'])

if __name__ == '__main__':
    unittest.main()