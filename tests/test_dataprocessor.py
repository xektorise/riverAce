import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
import numpy as np
from src.data_processor import DataPreprocessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.game_states, self.actions = self.preprocessor.generate_realistic_synthetic_data(100)
    
    def test_preprocess_game_state(self):
        game_state = self.game_states[0]
        result = self.preprocessor.preprocess_game_state(game_state)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 8)) # 8 features
    
    def test_preprocess_batch(self):
        result = self.preprocessor.preprocess_batch(self.game_states)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 8)) # 100 samples , 8 features
    
    def test_encore_decode_actions(self):
        encoded = self.preprocessor.encode_actions(['fold', 'call', 'raise'])
        self.assertEqual(list(encoded), [0, 2, 4])
        decoded = self.preprocessor.decode_actions(encoded)
        self.assertEqual(decoded, ['fold', 'call', 'raise'])
    
    def test_create_dataframe(self):
        df = self.preprocessor.create_dataframe(self.game_states, self.actions)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(df.columns), 9)
    
    def test_split_data(self):
        x_train, x_test, y_train, y_test = self.preprocessor.split_data(self.game_states, self.actions)
        self.assertEqual(len(x_train) + len(x_test), 100)
        self.assertEqual(len(y_train) + len(y_test), 100)
    
    def test_validate_dataset(self):
        self.assertTrue(self.preprocessor.validate_dataset(self.game_states, self.actions))
    
    def test_get_feature_importance(self):
        importance = self.preprocessor.get_feature_importance(self.game_states, self.actions)
        self.assertEqual(len(importance), 8) #8 features
        self.assertTrue(all(0 <= v <= 1 for v in importance.values()))
    
    def test_selected_features(self):
        selected = self.preprocessor.select_features(self.game_states)
        self.assertIsInstance(selected, list)
        self.assertTrue(all(feature in self.preprocessor.feature_columns for feature in selected))
    
    def test_save_load_preprocessor(self):
        self.preprocessor.save_preprocessor("test_preprocessor.joblib")
        loaded_preprocessor = DataPreprocessor.load_preprocessor("test_preprocessor.joblib")
        self.assertIsInstance(loaded_preprocessor, DataPreprocessor)
        self.assertEqual(loaded_preprocessor.feature_columns, self.preprocessor.feature_columns)
        
        
    def test_calculate_hand_strength(self):
        # test complete hand
        strength = self.preprocessor.calculate_hand_strength(['AH', 'KH'], ['QH', 'JH', 'TH'])
        self.assertGreater(strength, 0)

        # test incomplete hand
        strength = self.preprocessor.calculate_hand_strength(['AH', 'KH'], ['QH'])
        self.assertEqual(strength, 0)

        
        strength = self.preprocessor.calculate_hand_strength(['AH', 'KH'], ['QH', 'JH', 'TH'])
        self.assertGreater(strength, 0)

if __name__ == '__main__':
    unittest.main()