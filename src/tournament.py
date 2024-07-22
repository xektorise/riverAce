import logging
from typing import List
from game import Game
from player import Player

class Tournament:
    def __init__(self, players: List[Player], starting_chips: int, initial_small_blind: int, initial_big_blind: int):
        self.players = players
        self.starting_chips = starting_chips
        self.small_blind = initial_small_blind
        self.big_blind = initial_big_blind
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('poker_tournament.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def start_tournament(self):
        self.logger.info("Tournament started")
        for player in self.players:
            player.chips = self.starting_chips

        round_number = 1
        while len(self.players) > 1:
            self.logger.info(f"Starting round {round_number}")
            game = Game(self.players, self.small_blind, self.big_blind)
            game.play_round()
            self.players = [p for p in self.players if p.chips > 0]
            self._increase_blinds(round_number)
            round_number += 1

        self.end_tournament()

    def _increase_blinds(self, round_number):
        if round_number % 10 == 0:
            self.small_blind *= 2
            self.big_blind *= 2
            self.logger.info(f"Blinds increased to {self.small_blind}/{self.big_blind}")

    def end_tournament(self):
        winner = self.players[0]
        self.logger.info(f"Tournament ended. Winner: {winner.name} with {winner.chips} chips")

# Usage
if __name__ == "__main__":
    players = [Player("Player1", 1000, 100), Player("Player2", 1000, 100), Player("Player3", 1000, 100)]
    tournament = Tournament(players, starting_chips=1000, initial_small_blind=5, initial_big_blind=10)
    tournament.start_tournament()