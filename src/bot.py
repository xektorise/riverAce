import numpy as np
import pandas as pd
import pickle
import logging
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from player import Player
from neural_network import NeuralNetwork
from data_processor import DataPreprocessor
from typing import List, Dict, Tuple
from collections import defaultdict

class Bot(Player):
    def __init__(self, name: str, chips: int, bounty: int):
        super().__init__(name, chips, bounty)
        self.logger = logging.getLogger(__name__)
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

        # split the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # scale the features
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # train the model
        self.model.fit(x_train_scaled, y_train)

        # print the model's accuracy
        print(f"Model accuracy: {self.model.score(x_test_scaled, y_test)}")

    def decide(self, game_state: Dict, options: List[str]) -> str:
        try:
            self.update_opponent_model(game_state)
            self.adjust_strategy(game_state)

            features = self.preprocessor.preprocess_game_state(game_state)
            scaled_features = self.preprocessor.transform_data(features)

            hand_strength = features[0][4]
            pot_odds = self.calculate_pot_odds(game_state['pot_size'], game_state['current_bet'])

            opponent_ranges = {}
            for opponent in game_state['players']:
                if opponent['name'] != self.name and not opponent['has_folded']:
                    opponent_range = opponent.get_range(game_state)
                    opponent_description = opponent.get_range_description(game_state)
                    self.logger.info(f"Opponent {opponent['name']} range: {opponent_description}")
                    opponent_ranges[opponent['name']] = self.use_opponent_range(opponent['name'], game_state)

            strongest_opponent_range = min(opponent_ranges.values(), key=lambda x: len(x) if isinstance(x, list) else float('inf'))

            if self.model_trained:
                action_probs = self.model.predict_proba(scaled_features)[0]
                predicted_action = self.action_map[np.argmax(action_probs)]

                if predicted_action in options:
                    # adjust the predicted action based on opponent ranges
                    if len(strongest_opponent_range) <= 3 and predicted_action in ['raise', 'bet']:
                        # If opponent likely has a very strong hand, be more cautious
                        return 'call' if 'call' in options else 'check' if 'check' in options else 'fold'
                    elif len(strongest_opponent_range) > 10 and predicted_action in ['check', 'call']:
                        # If opponent has a wide range, consider being more aggressive
                        return 'raise' if 'raise' in options else 'bet' if 'bet' in options else predicted_action
                    else:
                            return predicted_action

            # fallback to heuristic decision-making
            return self.heuristic_decision(hand_strength, pot_odds, options, strongest_opponent_range)

        except Exception as e:
            self.logger.error(f"Error in decide method: {str(e)}")
            return 'fold'
            

    def heuristic_decision(self, game_state: Dict, hand_strength: float, pot_odds: float, options: List[str], opponent_range: List[str]) -> str:
        if self.should_bluff(game_state):
            if 'raise' in options:
                return 'raise'
            elif 'bet' in options:
                return 'bet'

        # adjust hand strength based on opponent's range
        if len(opponent_range) <= 3:
            hand_strength *= 0.8
        elif len(opponent_range) > 10:
            hand_strength *= 1.2

        stage = game_state['current_stage']
        if stage == 'river':
            # on the river, we might want to be more aggressive with strong hands
            if hand_strength > 0.8:
                if 'raise' in options:
                    return 'raise'
                elif 'bet' in options:
                    return 'bet'

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
    
    def should_bluff(self, game_state: Dict) -> bool:
        # bluffing frequency base rate (can be adjusted)
        base_bluff_frequency = 0.1

        # adjust bluff frequency based on position
        position_factor = self.get_position_factor(game_state)

        # adjust based on the stage of the hand
        stage_factor = self.get_stage_factor(game_state['current_stage'])

        # adjust based on the bot's image
        image_factor = self.get_image_factor()

        # adjust based on hand strength
        hand_strength = self.evaluate_hand_strength(game_state)
        hand_factor = 1 - hand_strength  # Bluff more with weaker hands

        # adjust based on number of players in the hand
        players_in_hand = len([p for p in game_state['players'] if not p['has_folded']])
        player_factor = 1 / players_in_hand  # bluff more in heads-up situations

        # adjust based on stack size
        stack_factor = self.get_stack_factor(game_state)

        # adjust based on opponent tendencies
        opponent_factor = self.get_opponent_factor(game_state)

        # calculate final bluff probability
        bluff_probability = (
            base_bluff_frequency *
            position_factor *
            stage_factor *
            image_factor *
            hand_factor *
            player_factor *
            stack_factor *
            opponent_factor
        )

        # ensure the probability is between 0 and 1
        bluff_probability = max(0, min(1, bluff_probability))

        self.logger.info(f"Bluff probability: {bluff_probability:.2f}")

        return random.random() < bluff_probability

    def get_position_factor(self, game_state: Dict) -> float:
        # bluff more in late position
        if self.position in [Position.BUTTON, Position.CUTOFF]:
            return 1.5
        elif self.position in [Position.MIDDLE_POSITION]:
            return 1.2
        else:
            return 1.0

    def get_stage_factor(self, stage: str) -> float:
        # bluff more on later streets
        if stage == 'river':
            return 1.5
        elif stage == 'turn':
            return 1.3
        elif stage == 'flop':
            return 1.1
        else:  # pre-flop
            return 1.0

    def get_image_factor(self) -> float:
        # if bot has been playing tight, it's a good time to bluff
        if self.calculate_vpip() < 20:
            return 1.5
        elif self.calculate_vpip() > 30:
            return 0.8
        else:
            return 1.0

    def get_stack_factor(self, game_state: Dict) -> float:
        # bluff more with a bigger stack
        avg_stack = sum(p['chips'] for p in game_state['players']) / len(game_state['players'])
        if self.chips > 2 * avg_stack:
            return 1.3
        elif self.chips < 0.5 * avg_stack:
            return 0.7
        else:
            return 1.0

    def get_opponent_factor(self, game_state: Dict) -> float:
        # bluff more against tight opponents
        opponents = [p for p in game_state['players'] if p['name'] != self.name and not p['has_folded']]
        if not opponents:
            return 1.0

        avg_opponent_vpip = sum(self.opponent_models[op['name']]['vpip'] for op in opponents) / len(opponents)
        if avg_opponent_vpip < 20:
            return 1.3
        elif avg_opponent_vpip > 30:
            return 0.7
        else:
            return 1.0
    
    def evaluate_hand_strength(self, game_state: Dict) -> float:
        hole_cards = self.hand.cards
        community_cards = game_state['community_cards']
        stage = game_state['current_stage']
        num_players = len([p for p in game_state['players'] if not p['has_folded']])

        # combine hole cards and community cards
        all_cards = hole_cards + community_cards

        # evaluate current hand
        current_hand_value = self.evaluate_hand(all_cards)

        # monte Carlo simulation for hand strength
        num_simulations = 1000
        wins = 0

        for _ in range(num_simulations):
            # create a deck excluding known cards
            deck = [card for card in self.deck.cards if card not in all_cards]
            np.random.shuffle(deck)

            # complete the board
            simulated_board = community_cards + deck[:5-len(community_cards)]

            # evaluate our hand
            our_hand = self.evaluate_hand(hole_cards + simulated_board)

            # simulate opponent hands
            opponent_hands = [self.evaluate_hand(deck[i:i+2] + simulated_board) for i in range(0, (num_players-1)*2, 2)]

            # check if we win
            if our_hand > max(opponent_hands):
                wins += 1

        hand_strength = wins / num_simulations

        # adjust hand strength based on stage and number of players
        if stage == 'pre-flop':
            hand_strength *= 0.8  # reduce confidence pre-flop
        elif stage == 'flop':
            hand_strength *= 0.9
        elif stage == 'turn':
            hand_strength *= 1.0
        else:  # river
            hand_strength *= 1.1  # increase confidence on the river

        # adjust for number of players
        hand_strength *= (1 - (num_players - 2) * 0.1)  # decrease strength as more players are in the hand

        return hand_strength
    
    
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
                

    def use_opponent_range(self, opponent_name: str, game_state: Dict):
        try:
            opponent_range = game_state['players'][opponent_name].get_range(game_state)

            top_hands = [hand for hand, prob in opponent_range if prob > 0.05]

            if len(top_hands) <= 3:
                self.logger.info(f"Opponent {opponent_name} likely has a strong hand. Proceeding with caution.")
                return 'strong'
            elif len(top_hands) <= 10:
                self.logger.info(f"Opponent {opponent_name} has a medium-strength range. Considering semi-bluffs.")
                return 'medium'
            else:
                self.logger.info(f"Opponent {opponent_name} has a wide range. Considering aggressive plays.")
                return 'wide'
        except KeyError:
            self.logger.warning(f"Opponent {opponent_name} not found in game state. Assuming unknown range.")
            return 'unknown'
        
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
    
    def adjust_tournament_strategy(self, tournament_stage: str):
        if tournament_stage == 'early':
            self.aggression_factor *= 0.9  # Play more conservatively early
        elif tournament_stage == 'middle':
            self.aggression_factor *= 1.0  # Play normally in the middle stages
        elif tournament_stage == 'late':
            self.aggression_factor *= 1.1  # Play more aggressively late
        elif tournament_stage == 'bubble':
            # Adjust strategy near the money bubble
            if self.chips > self.average_stack:
                self.aggression_factor *= 1.2  # Put pressure on shorter stacks
            else:
                self.aggression_factor *= 0.8  # Play more cautiously with a short stack
    
    def analyze_performance(self, game_history: List[Dict]):
        wins = sum(1 for game in game_history if game['winner'] == self.name)
        total_games = len(game_history)
        win_rate = wins / total_games if total_games > 0 else 0

        self.logger.info(f"Performance analysis:")
        self.logger.info(f"Total games: {total_games}")
        self.logger.info(f"Wins: {wins}")
        self.logger.info(f"Win rate: {win_rate:.2%}")

        # Analyze decision accuracy
        correct_decisions = sum(1 for game in game_history if game['bot_decision'] == game['optimal_decision'])
        decision_accuracy = correct_decisions / total_games if total_games > 0 else 0
        self.logger.info(f"Decision accuracy: {decision_accuracy:.2%}")
    
    def update_player_tendencies(self, player_name: str, action: str, situation: str):
        if player_name not in self.player_tendencies:
            self.player_tendencies[player_name] = defaultdict(lambda: defaultdict(int))

        self.player_tendencies[player_name][situation][action] += 1

    def get_player_tendency(self, player_name: str, situation: str) -> str:
        if player_name in self.player_tendencies and situation in self.player_tendencies[player_name]:
            actions = self.player_tendencies[player_name][situation]
            return max(actions, key=actions.get)
        return 'unknown'
    
    def calculate_vpip(self) -> float:
        if not self.stats['hands_played']:
            return 0.0
        return (self.stats['hands_played'] - self.stats['hands_folded']) / self.stats['hands_played'] * 100
    
    def reset_for_new_tournament(self):
        self.chips = self.initial_chips
        self.tournament_position = 0
        self.num_players = 0
        self.opponent_models.clear()
        self.player_tendencies.clear()
        self.aggression_factor = 1.0
        self.reset_stats()  # Assuming this method exists in the Player class
    
    
    def tune_hyperparameters(self, x: np.ndarray, y: np.ndarray):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(x, y)
        self.model = grid_search.best_estimator_
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
    def periodic_model_update(self, game_history: List[Dict], update_frequency: int = 1000):
        if len(game_history) % update_frequency == 0:
            self.update_model(game_history[-update_frequency:])

            # Extract X and y from game history
            game_states = [gh['state'] for gh in game_history[-update_frequency:]]
            actions = [gh['action'] for gh in game_history[-update_frequency:]]
            df = self.preprocessor.create_dataframe(game_states, actions)
            x = df[self.preprocessor.feature_columns]
            y = df['action']

            self.tune_hyperparameters(x, y)
            self.logger.info("Model updated and hyperparameters tuned.")

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