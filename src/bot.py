import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from player import Player
from neural_network import NeuralNetwork
from data_processor import DataPreprocessor
from typing import List, Dict
from collections import defaultdict

class Bot(Player):
    def __init__(self, name: str, chips: int, bounty: int):
        super().__init__(name, chips, bounty)
        self.preprocessor = DataPreprocessor()
        self.model = self._initialize_model('random_forest')
        self.scaler = StandardScaler()
        self.action_map = self.preprocessor.action_map
        self.opponent_models = defaultdict(lambda: {'actions': [], 'vpip' : 0, 'pfr' : 0, 'aggression_factor' : 1.0})
        self.model_trained = False
        self.tournament_position = 0
        self.num_players = 0

    def _initialize_model(self, model_type ='random_forest'):
        if model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'neural_network':
            return NeuralNetwork (input_shape=(len(self.preprocessor.feature_columns),))


    def train(self, game_states: List[Dict], actions: List[int]):
        
        self.model_trained = True
        df = self.preprocessor.create_dataframe(game_states, actions)
        x = df[self.preprocessor.feature_columns]
        y = df['action']

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Scale the features
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Train the model
        self.model.fit(x_train_scaled, y_train)

        # Print the model's accuracy
        print(f"Model accuracy: {self.model.score(x_test_scaled, y_test)}")

    def decide(self, game_state: Dict, options: List[str]) -> str:
        self.update_opponent_model(game_state)
        self.adjust_strategy(game_state)

        features = self.preprocessor.preprocess_game_state(game_state)
        scaled_features = self.preprocessor.transform_data(features)

        hand_strength = features[0][4]
        pot_odds = self.calculate_pot_odds(game_state['pot_size'], game_state['current_bet'])

        if self.model_trained:
            action_probs = self.model.predict_proba(scaled_features)[0]
            predicted_action = self.action_map[np.argmax(action_probs)]

            if predicted_action in options:
                return predicted_action

        # fallback to heuristic decision-making
        if hand_strength > 0.8 * self.aggression_factor:
            if 'raise' in options:
                return 'raise'
            elif 'bet' in options:
                return 'bet'
            else:
                return 'call'
        
        if hand_strength > 0.5 * self.aggression_factor:
            if pot_odds < hand_strength:
                if 'call' in options:
                    return 'call'
                elif 'check' in options:
                    return 'check'
            else:
                if 'check' in options:
                    return 'check'
                else:
                    return 'fold'
        else:
            if 'check' in options:
                return 'check'
            else:
                return 'fold'

    def update_model(self, game_history: List[Dict]):
        if not game_history:
            return
        
        game_states = [gh['state'] for gh in game_history]
        actions = [gh['action'] for gh in game_history]
        
        self.train(game_states, actions)
    
    def update_opponent_model(self, game_state: Dict):
        for player, action in game_state['last_actions'].items():
            if player != self.name:
                if player not in self.opponent_models:
                    self.opponent_models[player] = {'actions' : []}
                self.opponent_models[player]['actions'].append(action)

                # update vpip
                if action in ['call', 'raise', 'bet']:
                    self.opponent_models[player]['vpip'] = sum(1 for a in self.opponent_models[player]['actions'] if a in ['call', 'raise', 'bet']) / len(self.opponent_models[player]['actions'])
                
                # update pfr
                if game_state['stage'] == 'preflop' and action == 'raise':
                    self.opponent_models[player]['pfr'] = sum(1 for a in self.opponent_models[player]['actions'] if a == 'raise') / len(self.opponent_models[player]['actions'])

                aggressive_actions = sum(1 for a in self.opponent_models[player]['actions'] if a in ['raise', 'bet'])
                passive_actions = sum(1 for a in self.opponent_models[player]['actions'] if a == 'call')
                self.opponent_models[player]['aggression_factor'] = aggressive_actions / passive_actions if passive_actions > 0 else 2.0

    def adjust_strategy(self, game_state: Dict):
        self.tournament_position = game_state['player_position'][self.name]
        self.num_players = len(game_state['players'])

        if self.num_players <= 3:
            self.aggression_factor = 1.3 # play more aggressive in final table
        elif self.tournament_position / self.num_players < 0.3:
            self.aggression_factor = 1.1 # play slightly more aggressive if in top 30%
        else:
            self.aggression_factor = 1.0
        

        average_stack = sum(player['chips'] for player in game_state['players']) / self.num_players

        if self.chips < average_stack * 0.5:
            self.aggression_factor = 1.2 # play more aggressive with short stack
        elif self.chips > average_stack * 2:
            self.aggression_factor = 0.9 # play more conservativ with big stack

    def save_state(self, filepath):
        state = {
            'name': self.name,
            'chips': self.chips,
            'bounty': self.bounty,
            'model': self.model,
            'opponent_models': dict(self.opponent_models),
            'model_trained': self.model_trained,
            'tournament_position': self.tournament_position,
            'num_players': self.num_players,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        bot = cls(state['name'], state['chips'], state['bounty'])
        bot.model = state['model']
        bot.opponent_models = defaultdict(lambda: {'actions': [], 'vpip': 0, 'pfr': 0, 'aggression_factor': 1.0}, state['opponent_models'])
        bot.model_trained = state['model_trained']
        bot.tournament_position = state['tournament_position']
        bot.num_players = state['num_players']
        return bot