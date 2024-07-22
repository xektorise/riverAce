import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'pot_size', 'num_community_cards', 'current_bet', 
            'player_chips', 'hand_strength', 'position', 'stack_to_pot_ratio', 'num_players'
        ]
        self.action_map = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all-in': 5}
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}

    def preprocess_game_state(self, game_state: Dict) -> np.array:
        try:
            features = [
                game_state['pot_size'],
                len(game_state['community_cards']),
                game_state['current_bet'],
                game_state['player_chips'],
                self.calculate_hand_strength(game_state['hole_cards'], game_state['community_cards']),
                self.position_to_numeric(game_state['position']),
                game_state['player_chips'] / (game_state['pot_size'] + 1),
                len(game_state['players'])
            ]
            return np.array(features).reshape(1, -1)
        except KeyError as e:
            raise ValueError(f"Missing key in game state: {e}")


    def preprocess_batch(self, game_states: List[Dict]) -> np.array:
        """
        Preprocess a batch of game states.
        """
        return np.array([self.preprocess_game_state(state)[0] for state in game_states])

    def fit_scaler(self, data: np.array):
        """
        Fit the scaler to the data.
        """
        self.scaler.fit(data)

    def transform_data(self, data: np.array) -> np.array:
        """
        Transform the data using the fitted scaler.
        """
        return self.scaler.transform(data)

    def fit_transform_data(self, data: np.array) -> np.array:
        """
        Fit the scaler to the data and transform it.
        """
        return self.scaler.fit_transform(data)

    @staticmethod
    def calculate_hand_strength(hole_cards: List[str], community_cards: List[str]) -> float:
        # Implement hand strength calculation
        # This is a placeholder - you'll need to implement actual hand strength logic
        return np.random.random()
    
    def normalize_hand_strength(self, strength: float) -> float:
        # Normalize hand strength to 0-1 range
        # This method should be adjusted based on your actual hand strength calculation
        return max(0, min(strength, 1))

    @staticmethod
    def position_to_numeric(position: int) -> float:
        # Convert position to a numeric value between 0 and 1
        # Assuming 8 players max
        return position / 8
    
    def encode_actions(self, actions: List[str]) -> np.array:
        """Encode categorical actions to numerical values"""
        action_map = {'fold' : 0, 'check' : 1, 'call' : 2, 'bet' : 3, 'raise' : 4, 'all-in' : 5}
        return np.array([action_map.get(action, -1) for action in actions])
    
    def decode_actions(self, encoded_actions: np.array) -> List[str]:
        """Decode numerical action values back to categorical strings"""
        return [self.reverse_action_map.get(action, 'unknown') for action in encoded_actions]

    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle missing data - here we're just dropping rows with any missing values
        # You might want to use a more sophisticated method depending on your needs
        return df.dropna()
    
    def get_feature_columns(self) -> List[str]:
        """Return the current feature columns"""
        return self.feature_columns
    
    def update_feature_columns(self, new_columns: List[str]):
        """Update the feature columns used in preprocessing"""
        self.feature_columns = new_columns

    def preprocess_single(self, game_state: Dict, action: str) -> np.array:
        """Preprocess a single game state and action."""
        features = self.preprocess_game_state(game_state)
        encoded_action = self.encode_actions([action])
        return np.concatenate([features.flatten(), encoded_action])
    
    def reset_scaler(self):
        """Reset the scaler to its inital state"""
        self.scaler = StandardScaler()

    def get_action_map(self) -> Dict[str, int]:
        """Return the current action map"""
        return self.action_map
    
    def validate_game_state(self, game_state: Dict) -> bool:
        """Validate that the game state contains all required keys"""
        required_keys = ['pot_size', 'community_cards', 'current_bet', 'player_chips', 'hole_cards', 'position', 'players']
        return all(key in game_state for key in required_keys)
    
    def normalize_features(self, features: np.array) -> np.array:
        """Normalize all features to a 0-1 range"""
        return (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))

    def create_dataframe(self, game_states: List[Dict], actions: List[int]) -> pd.DataFrame:
        """Create a pandas DataFrame from game states and actions."""
        features = self.preprocess_batch(game_states)
        df = pd.DataFrame(features, columns=self.feature_columns)
        df['action'] = actions
        return df

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded sucessfully to {filepath}")
            return self.handle_missing_data(df)
        except IOError:
            print(f"Error loading data from {filepath}")
            return pd.DataFrame()
        
    def save_data(self, data: pd.DataFrame, filepath: str):
        """Save data to a CSV file."""
        try:
            data.to_csv(filepath, index=False)
            print(f"Data saved sucessfully to {filepath}")
        except IOError:
            print(f"Error saving data to {filepath}")