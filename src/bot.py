import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from player import Player
from neural_network import NeuralNetwork
from data_processor import DataPreprocessor
from typing import List, Dict

class Bot(Player):
    def __init__(self, name: str, chips: int, bounty: int):
        super().__init__(name, chips, bounty)
        self.preprocessor = DataPreprocessor()
        self.model = self._initialize_model()
        self.scaler = StandardScaler()
        self.action_map = {0: 'fold', 1: 'check', 2: 'call', 3: 'bet', 4: 'raise', 5: 'all-in'}

    def _initialize_model(self):
        # Initialize a RandomForestClassifier as our model
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def _preprocess_game_state(self, game_state: Dict) -> np.array:
        # Convert game state to a feature vector
        features = [
            game_state['pot_size'],
            len(game_state['community_cards']),
            game_state['current_bet'],
            self.chips,
            self.calculate_hand_strength(game_state['community_cards']),
            self.position_to_numeric(game_state['player_positions'][self.name]),
            # Add more features as needed
        ]
        return np.array(features).reshape(1, -1)

    def position_to_numeric(self, position: int) -> float:
        # Convert position to a numeric value between 0 and 1
        return position / len(self.players)

    def calculate_hand_strength(self, community_cards: List['Card']) -> float:
        # Implement hand strength calculation
        # This is a placeholder - you'll need to implement actual hand strength logic
        return np.random.random()

    def train(self, game_states: List[Dict], actions: List[int]):

        df = self.preprocessor.create_dataframe(game_states, actions)
        x = df[self.preprocessor.feature_columns]
        y = df['action']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Print the model's accuracy
        print(f"Model accuracy: {self.model.score(X_test_scaled, y_test)}")

    def decide(self, game_state: Dict, options: List[str]) -> str:
        processed_state = self.preprocessor.preprocess_game_state(game_state)
        # Preprocess the game state
        features = self._preprocess_game_state(game_state)

        # Scale the features
        scaled_features = self.scaler.transform(features)

        # Get the model's prediction
        action_index = self.model.predict(scaled_features)[0]

        # Map the prediction to an action
        predicted_action = self.action_map[action_index]

        # Ensure the predicted action is in the list of options
        if predicted_action in options:
            return predicted_action
        else:
            # If the predicted action is not available, choose randomly from available options
            return np.random.choice(options)

    def update_model(self, game_history: pd.DataFrame):
        # Use game history to update the model
        self.train(game_history)