import random
import logging
import json
import time
import os
from typing import List
from bot import Bot
from game import Game


class Tournament:
    def __init__(self, num_players: int, starting_chips: int, starting_bounty: int, small_blind: int, big_blind: int):
        self.bots = [Bot(f"Bot_{i}", starting_chips, starting_bounty) for i in range(num_players)]
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.logger = self._setup_logger()
        self.stats = {
            'longest_game' : 0,
            'biggest_pot' : 0,
            'total_hands_played' : 0,
            'total_bounties_collected' : 0
        }
        self.current_game = 0
    

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('tournament.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def run(self, num_games: int):
        for game_num in range(self.current_game, num_games):
            self.current_game = game_num
            self.logger.info(f"Starting Game {game_num + 1}")
            
            random.shuffle(self.bots)
            
            game = Game(self.bots, self.small_blind, self.big_blind)
            winners = game.play_round()
            
            self.update_stats(game)
            
            eliminated_players = [bot for bot in self.bots if bot.chips <= 0]
            for eliminated_player in eliminated_players:
                self.logger.info(f"{eliminated_player.name} has been eliminated")
                self.stats['total_bounties_collected'] += eliminated_player.bounty
            
            self.bots = [bot for bot in self.bots if bot.chips > 0]
            
            for bot in self.bots:
                bot.update_model(game.get_game_history())
            
            self.logger.info(f"Game {game_num + 1} completed. Remaining players: {len(self.bots)}")
            for bot in self.bots:
                self.logger.info(f"{bot.name}: Chips = {bot.chips}, Bounty = {bot.bounty}")
            
            if len(self.bots) == 1:
                self.logger.info(f"Tournament winner: {self.bots[0].name}")
                break
            
            if game_num % 10 == 9:
                self.small_blind *= 2
                self.big_blind *= 2
                self.logger.info(f"Blinds increased to {self.small_blind}/{self.big_blind}")
            
            if game_num % 20 == 19:
                self.save_state()
            
            time.sleep(0.1)
        
        self.conclude_tournament()
    

    def update_stats(self, game):
        self.stats['longest_game'] = max(self.stats['longest_game'], len(game.get_game_history()))
        self.stats['biggest_pot'] = max(self.stats['biggest_pot'], game.pot)
        self.stats['total_hands_played'] += 1
        self.stats['total_bounties_collected'] += sum(bot.bounty for bot in self.bots if bot.chips == 0)
    
    def conclude_tournament(self):
        self.bots.sort(key=lambda x: x.chips, reverse=True)
        for i, bot in enumerate(self.bots):
            self.logger.info(f"{i+1}. {bot.name}: Chips = {bot.chips}, Bounty = {bot.bounty}")
        
        self.distribute_payouts()
        self.log_tournament_stats()

    
    def distribute_payouts(self):
        total_chips = sum(bot.chips for bot in self.bots)
        payouts = [0.5, 0.3, 0.2] #50 %, 30%, 20%

        for i, percentage in enumerate(payouts):
            if i < len(self.bots):
                payout = int(total_chips * percentage)
                self.bots[i].chips += payout
                self.logger.info(f"{self.bots[i].name} receives payout of {payout}")
    
    def log_tournament_stats(self):
        self.logger.info("Tournament statistics: ")
        for stat, value in self.stats.items():
            self.logger.info(f"{stat}: {value}")
    
    def save_state(self):
        state = {
            'current_game': self.current_game,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'stats': self.stats,
            'bots': [bot.to_dict() for bot in self.bots]
        }
        with open('tournament_state.json', 'w') as f:
            json.dump(state, f)
        self.logger.info(f"Tournament state saved at game {self.current_game}")

    @classmethod
    def load_state(cls):
        if not os.path.exists('tournament_state.json'):
            raise FileNotFoundError("No saved tournament state found")
        
        with open('tournament_state.json', 'r') as f:
            state = json.load(f)
        
        tournament = cls(0, 0, 0, state['small_blind'], state['big_blind'])
        tournament.current_game = state['current_game']
        tournament.stats = state['stats']
        tournament.bots = [Bot.from_dict(bot_dict) for bot_dict in state['bots']]
        
        return tournament

if __name__ == "__main__":
    if os.path.exists('tournament_state.json'):
        tournament = Tournament.load_state()
        print("Resuming tournament from saved state")
    else:
        tournament = Tournament(num_players=8, starting_chips=5000, starting_bounty=100, small_blind=10, big_blind=20)
    
    tournament.run(num_games=100)