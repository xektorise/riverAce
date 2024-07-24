import unittest
import pandas as pd
import numpy as np
from data_processor import DataPreprocessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.game_states, self.actions = self.preprocessor.generate_realistic_synthetic_data(100)
    
    def test_preprocess_game_state(self):
        game_state = self.game_states[0]
        result = self.preprocessor.preprocess_game_state(game_state)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 8)) # 8 features
        
    