import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import List, Dict, Tuple
from card import Card
from hand import Hand
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class DataPreprocessor:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
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
                self.position_to_numeric(game_state['position'], len(game_state['players'])),
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
        try:
            return np.array([self.preprocess_game_state(state)[0] for state in game_states])
        
        except Exception as e:
            self.logger.error(f"Error in preprocess_batch: {e}")
            raise

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
        try:
            
            hole_card_objects = [Card(card[0], card[1:]) for card in hole_cards]
            community_card_objects = [Card(card[0], card[1:]) for card in community_cards]

            hand = Hand()
            hand.add_hole_cards(hole_card_objects)
            hand.add_community_cards(community_card_objects)


            return hand.get_hand_strength()
        
        except Exception as e:
            print(f"Error calculating hand strength: {e}")
            return 0

    @staticmethod
    def position_to_numeric(position: int, num_players: int) -> float:
        """Convert position to a numeric value between 0 and 1

        Args:
            position (int): The player's position at the tables (0-indexed)
            num_players (int): The total number of players in the game

        Returns:
            float: A value between 0 and 1 representing the normalized position
        """
        
        if num_players < 2:
            raise ValueError("These must be atleast 2 players in the game")
        if position < 0 or position >= num_players:
            raise ValueError(f"Position must be between 0 and {num_players - 1}")
        
        return (position + 1) / num_players
    
    def encode_actions(self, actions: List[str]) -> np.array:
        """Encode categorical actions to numerical values"""
        action_map = {'fold' : 0, 'check' : 1, 'call' : 2, 'bet' : 3, 'raise' : 4, 'all-in' : 5}
        return np.array([action_map.get(action, -1) for action in actions])
    
    def decode_actions(self, encoded_actions: np.array) -> List[str]:
        """Decode numerical action values back to categorical strings"""
        return [self.reverse_action_map.get(action, 'unknown') for action in encoded_actions]

    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in the DataFrame
        
        This method uees a combination of techniques to handle missing data:
        
        1. For numerical columns, it imputes missing values with the median.
        2. For categorical colums, it imputes missing values with the mode (most frequent value)
        3. If a column has more than 50% missing value, it drops that column
        4. After imputation, if any rows have missing values, they will be dropped

        Args:
            df (pd.DataFrame): Input DataFrame with potentially mising values

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        
        inital_shape = df.shape
        self.logger.info(f"Inital dataframe shape: {inital_shape}")
        
        # first, drop columns with more than 50% missing values
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)
        
        columns_dropped = inital_shape[1] - df.shape[1]
        self.logger.info(f"Columns dropped due to >50% missing values: {columns_dropped}")
        
        # seperate numerical and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        
        # impute missing values in numerical columns with median
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace = True)
        
        # imput missing values in categorical columns with mode
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace = True)
        

        # after imputation, drop any rows that still have missing values
        df = df.dropna()
        
        rows_dropped = inital_shape[0] - df.shape[0]
        self.logger.info(f"Rows dropped due to missing values: {rows_dropped}")
        self.logger.info(f"DataFrame final shape: {df.shape}")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return the current feature columns"""
        return self.feature_columns
    
    def update_feature_columns(self, new_columns: List[str]):
        """Update the feature columns used in preprocessing"""
        self.feature_columns = new_columns
    
    def scale_features(self, data: np.array, columns_to_scale: List[int]):
        """Scale specific features of data"""
        data_scaled = data.copy()
        data_scaled[:, columns_to_scale] = self.scaler.fit_transform(data[:, columns_to_scale])
        return data_scaled

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
        try:
            
            features = self.preprocess_batch(game_states)
            df = pd.DataFrame(features, columns=self.feature_columns)
            df['action'] = actions
            return df
        
        except Exception as e:
            self.logger.info(f"Error in create_dataframe: {e}")
            raise
    
    def split_data(self, game_states: List[Dict], actions: List[int], test_size: float = 0.2) -> Tuple [np.array, np.array, np.array, np.array]:
        """Split data into training and validation sets"""
        try:
            
            x = self.preprocess_batch(game_states)
            y = np.array(actions)
            return train_test_split(x, y, test_size=test_size, random_state=42)
        
        except Exception as e:
            self.logger.info(f"Error in split_data: {e}")
            raise
    
    def validate_dataset(self, game_states: List[Dict], actions: List[int]) -> bool:
        """Validate entire dataset"""
        if len(game_states) != len(actions):
            self.logger.error("Number of game states does not match number of actions")
            return False

        if not all(self.validate_game_state(state) for state in game_states):
            self.logger.error("One or more game states are invalid")
            return False

        if not all(action in self.action_map.values() for action in actions):
            self.logger.error("One or more actions are invalid")
            return False

        return True
    
    def preprocess_complete_state(self, game_state: Dict, action: int) -> np.array:
        """Preprocess a complete game state including  the action"""
        features = self.preprocess_game_state(game_state)
        return np.concatenate([features.flatten(), [action]])
    
    
    def get_data_summary(self, game_states: List[Dict], actions: List[int]):
        """Get summary stats of the preprocessed data"""
        features = self.preprocess_batch(game_states)
        df = pd.DataFrame(features, columns=self.feature_columns)
        df['action'] = actions
        
        summary = {
            'num_samples' : len(df),
            'feature_means' : df[self.feature_columns].mean().to_dict(),
            'feature_stds' : df[self.feature_columns].std().to_dict(),
            'action_distribution' : df['action'].value_counts().to_dict()
        }
        
        return summary
    
    def visualize_feature_distribution(self, game_states: List[Dict]):
        """Visualize the distribution of features"""
        try:
            
            features = self.preprocess_batch(game_states)
            df = pd.DataFrame(features, columns = self.feature_columns)

            fig, axes = plt.subplots(4, 2, figsize=(15, 20))
            axes = axes.flatten()

            for i, col in enumerate(self.feature_columns):
                df[col].hist(ax=axes[i], bins = 50)
                axes[i].set_title(col)
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')

            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            self.logger.error(f"Error in visualize_feature_distribution: {e}")
            raise
            
    
    
    def visualize_correlation_matrix(self, game_states: List[Dict]):
        """Visualize the correlation matrix of features"""
        try:
            
            corr_matrix = self.check_feature_correlations(game_states)

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot =  True, cmap = 'coolwarm', vmin = -1, vmax = 1, center = 0)
            plt.title('Feature Correlation Matrix')
            plt.show()
        
        except Exception as e:
            self.logger.error(f"Error in visualize_correlation_matrix: {e}")
            raise
    
    def plot_action_distribution(self, actions: List[int]):
        """Plot the distribution of actions."""
        try:
            action_counts = pd.Series(actions).value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            action_counts.plot(kind='bar')
            plt.title('Distribution of Actions')
            plt.xlabel('Action')
            plt.ylabel('Count')
            plt.xticks(range(len(self.action_map)), self.action_map.keys(), rotation=0)
            plt.show()
        except Exception as e:
            self.logger.error(f"Error in plot_action_distribution: {e}")
            raise
        
        
    def check_feature_correlations(self, game_state: List[Dict]) -> pd.DataFrame:
        """Check correlation between features"""
        features = self.preprocess_batch(game_state)
        df = pd.DataFrame(features, columns = self.feature_columns)
        return df.corr()
    
    def get_feature_statistics(self, game_states: List[Dict]) -> pd.DataFrame:
        """Get basic statistics of the features"""
        features = self.preprocess_batch(game_states)
        df = pd.DataFrame(features, columns = self.feature_columns)
        return df.describe()
    
    def get_feature_importance(self, game_states: List[Dict], actions: List[int]) -> Dict[str, float]:
        """Get feature importance using a Random Forest classifier."""
        x = self.preprocess_batch(game_states)
        y = np.array(actions)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(x, y)
        return dict(zip(self.feature_columns, rf.feature_importances_))
    
    def get_most_correlated_features(self, game_states: List[Dict], threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """Get Pairs of features with correlation above a certain threshold"""
        corr_matrix = self.check_feature_correlations(game_states)
        
        correlated_features = []
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    correlated_features.append((self.feature_columns[i], self.feature_columns[j], corr_matrix.iloc[i, j]))
        return sorted(correlated_features, key=lambda x: abs(x[2]), reverse=True)
    
    def select_features(self, game_states: List[Dict], correlation_threshold: float = 0.9) -> List[str]:
        """Select features by remove highly correlated features."""
        corr_matrix = self.check_feature_correlations(game_states)
        
        features_to_drop = set()
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    features_to_drop.add(self.feature_columns[j])
        
        return [feature for feature in self.feature_columns if feature not in features_to_drop]
    
    
    def generate_realistic_synthetic_data(self, num_samples: int = 100) -> Tuple[List[Dict], List[int]]:
        """Generate more realistic synthetic data"""
        game_states = []
        actions = []

        for _ in range(num_samples):
            num_players = np.random.randint(2, 9)
            game_state = {
                'pot_size': np.random.randint(10, 1000),
                'community_cards': ['AH', 'KH', 'QH', 'JH', 'TH'][:np.random.randint(0, 6)],
                'current_bet': np.random.randint(0, 100),
                'player_chips': np.random.randint(100, 5000),
                'hole_cards': [np.random.choice(['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']) + 
                               np.random.choice(['H', 'D', 'C', 'S']) for _ in range(2)],
                'position': np.random.randint(0, num_players),
                'players': [f'p{i}' for i in range(num_players)]
            }

            game_states.append(game_state)
            actions.append(np.random.choice(list(self.action_map.values())))

        return game_states, actions
    
    
    def save_preprocessor(self, filepath: str):
        """Save the preprocessor to a file"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load_preprocessor(cls, filepath: str):
        """Load a preprocessor"""
        return joblib.load(filepath)
    

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