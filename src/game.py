import logging
from typing import Dict, List, Tuple
from deck import Deck
from player import Player
from bot import Bot
from card import Card

class Game:
    
    STAGES = ['pre-flop', 'flop', 'turn', 'river']
    
    
    def __init__(self, players: List[Player], small_blind: int, big_blind: int):
        """Initialize the game with players, blinds, and a deck."""
        self.deck = Deck()
        self.players = players
        self.pot = 0
        self.community_cards: List[Card] = []
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.current_bet = 0
        self.current_player_index = 0
        self.dealer_index = 0
        self.side_pots: List[Tuple[int, Player]] = []
        self.current_stage = 'pre-flop'
        self.logger = self._setup_logger()
        self.game_history = []
        self.last_actions = {}
    
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('poker_game.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s ')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def deal_hands(self):
        """Deal hole cards to each player."""
        for player in self.players:
            player.hand.clear()
            player.hand.add_hole_cards(self.deck.deal(2))
        self.logger.info("Hands dealt to all players")
    
    def deal_community_cards(self, num_cards: int):
        """Deal community cards."""
        new_cards = self.deck.deal(num_cards)
        self.community_cards.extend(new_cards)
        self.logger.info(f"{num_cards} community cards dealt: {new_cards}")
    
    def collect_bets(self):
        """Collect bets from all players and add them to the pot."""
        for player in self.players:
            self.pot += player.current_bet
            player.current_bet = 0
        self.logger.info(f"Bets collected. Pot size {self.pot}")
    
    def post_blinds(self):
        """Post small and big blinds."""
        small_blind_player = self.players[(self.dealer_index + 1) % len(self.players)]
        big_blind_player = self.players[(self.dealer_index + 2) % len(self.players)]
        
        small_blind_player.bet(self.small_blind)
        big_blind_player.bet(self.big_blind)
        
        self.current_bet = self.big_blind
        self.current_player_index = (self.dealer_index + 3) % len(self.players)
        self.logger.info(f"Blinds posted. SB: {small_blind_player}, BB: {big_blind_player}")
    
    def betting_round(self) -> None:
        """Conduct a single round of betting."""
        active_players = [player for player in self.players if not player.has_folded and player.chips > 0]
        players_acted = 0
        last_raiser = None
        first_bet_made = self.current_bet > 0

        while not self._betting_round_complete(active_players, players_acted, last_raiser):
            current_player = active_players[self.current_player_index % len(active_players)]
            if current_player.has_folded or current_player.chips == 0:
                self.current_player_index = (self.current_player_index + 1) % len(active_players)
                continue
            
            action = self._get_player_action(current_player, first_bet_made)
            self._process_player_action(current_player, action)
            self.last_actions[current_player.name] = action
            
            if action in ['bet', 'raise']:
                last_raiser = current_player
                players_acted = 1
                first_bet_made = True
            else:
                players_acted += 1
            
            self.current_player_index = (self.current_player_index + 1) % len(active_players)
        
        self.collect_bets()
    
    def _betting_round_complete(self, active_players: List[Player], players_acted: int, last_raiser: Player) -> bool:
        return (players_acted >= len(active_players) and 
                (self.current_bet == 0 or 
                 all(player.current_bet == self.current_bet or player.has_folded or player.chips == 0 
                     for player in active_players)))
    
    def _get_player_action(self, player: Player, first_bet_made: bool) -> str:
        options = ['call', 'fold', 'check', 'raise', 'all-in'] if first_bet_made else ['bet', 'fold', 'check', 'raise']
        
        if self.current_bet > player.current_bet:
            if 'check' in options:
                options.remove('check')
            elif self.current_bet == player.current_bet:
                if 'call' in options:
                    options.remove('call')

            if self.current_bet >= player.chips + player.current_bet:
                options = ['fold', 'all-in']
            elif 'bet' in options and self.current_bet > 0:
                options.remove('bet')

        return player.decide(self.get_game_state(), options=options)
    
    def _process_player_action(self, player: Player, action: str):
        if action == 'bet':
            bet_amount = player.bet(self.big_blind)
            self.current_bet = bet_amount
            self.logger.info(f"Player {player.name} bet {bet_amount}")
        elif action == 'check':
            player.check()
            self.logger.info(f"Player {player.name} checked")
        elif action == 'raise':
            raise_amount = player.raise_bet(self.pot, self.current_bet)
            self.current_bet = raise_amount
            self.logger.info(f"Player {player.name} raised to {raise_amount}")
        elif action == 'call':
            call_amount = self.current_bet - player.current_bet
            if call_amount > player.chips:
                all_in_amount = player.all_in()
                self.logger.info(f"Player {player.name} called all-in for {all_in_amount}")
            else:
                player.bet(call_amount)
                self.logger.info(f"Player {player.name} called {call_amount}")
        elif action == 'fold':
            player.fold()
            self.logger.info(f"Player {player.name} folded")
        elif action == 'all-in':
            all_in_amount = player.all_in()
            if all_in_amount > self.current_bet:
                self.current_bet = all_in_amount
            self.handle_side_pots(player, all_in_amount)
            self.logger.info(f"Player {player.name} went all-in for {all_in_amount}")
    
    
    
    def handle_side_pots(self, all_in_player: Player, all_in_amount: int) -> None:
        """Handle side pots when a player goes all-in."""
        side_pot = 0
        for player in self.players:
            if player.current_bet > all_in_amount:
                excess = player.current_bet - all_in_amount
                player.current_bet = all_in_amount
                side_pot += excess

        if side_pot > 0:
            self.side_pots.append((side_pot, [p for p in self.players if p != all_in_player and not p.has_folded]))
        
        self.logger.info(f"Side pot of {side_pot} created, excluding {all_in_player.name}")

    def move_dealer(self):
        """Move the dealer button to the next player."""
        self.dealer_index = (self.dealer_index + 1) % len(self.players)
        self.logger.info(f"Dealer moved to player {self.players[self.dealer_index].name}")
    
    def play_round(self):
        try:
            self.logger.info("New round started")
            self.deck.shuffle()
            self.deal_hands()
            self.post_blinds()

            for stage in self.STAGES:
                self.current_stage = stage
                self.logger.info(f"Stage: {stage}")
                if stage == 'flop':
                    self.deal_community_cards(3)
                elif stage in ['turn', 'river']:
                    self.deal_community_cards(1)
                self.betting_round()
                self.logger.info(f"Pot after {stage}: {self.pot}")

            winners = self.determine_winner()
            self.award_pot(winners)
            self.move_dealer()
            self.reset_for_new_round()
            return winners
        except Exception as e:
            self.logger.error(f"Error during round: {str(e)}")
            self.logger.error(f"Current game state: {self.get_game_state()}")
            return []

    def award_pot(self, winners):
        if not winners:
            return
        pot_share = self.pot // len(winners)
        for winner in winners:
            winner.chips += pot_share
            self.logger.info(f"{winner.name} wins {pot_share} chips")
        self.pot = 0
    
    def determine_winner(self):
        """Determine the winner of the round and handle split pots."""
        best_hand = None
        winners = []
        for player in self.players:
            if not player.has_folded:
                player_hand = player.hand.evaluate(self.community_cards)
                
                if not best_hand or player_hand > best_hand:
                    best_hand = player_hand
                    winners = [player]
                elif player_hand == best_hand:
                    winners.append(player)
        
        if winners:
            winning_hand, hand_description = best_hand
            self.split_pot(winners)
            self.logger.info(f"Winning hand: {hand_description} - {winning_hand}")

        # distribute bounties and eliminate players with zero chips
        eliminated_players = [player for player in self.players if player.chips == 0]
        for eliminated_player in eliminated_players:

            eliminator = max((p for p in self.players if not p.has_folded), key=lambda p: p.current_bet)
            eliminator.chips += eliminated_player.bounty
            self.logger.info(f"{eliminator.name} won {eliminated_player.bounty} bounty from {eliminated_player.name}")
            self.remove_player(eliminated_player)

        self.pot = 0
        self.side_pots.clear()
        self.logger.info(f"Round ended. Winners {[winner.name for winner in winners]}")

        return winners

    def split_pot(self, winners: List[Player]):
        """Split the pot among the winners."""
        num_winners = len(winners)
        pot_share = self.pot // num_winners
        remainder = self.pot % num_winners

        for winner in winners:
            winner.chips += pot_share

        # distribute the remainder chips starting from the dealer
        for i in range(remainder):
            winners[(self.dealer_index + i) % num_winners].chips += 1
        
        self.logger.info(f"Pot split among {num_winners} players")
    
    def remove_player(self, player: Player):
        """Remove a player from the game."""
        self.players = [p for p in self.players if p.name != player.name]
        self.logger.info(f"Players {player.name} removed from the game.")
        if len(self.players) == 1:
            self.logger.info(f"Game over. {self.players[0].name} wins")
    

    def reset_for_new_round(self):
        self.pot = 0
        self.community_cards = []
        self.current_bet = 0
        self.side_pots = []
        self.current_stage = 'pre-flop'
        for player in self.players:
            player.reset_for_new_round()

    def get_game_history(self) -> List[Dict]:
        """Get the game history for bot training"""
        return self.game_history
    

    def add_game_to_history(self, state: Dict, action: str):
        """Add current game state to and action to game history"""
        self.game_history.append({
            'state': state,
            'action' : action
        })

    def get_average_stack(self) -> float:
        """Calculate average stack size"""
        return sum(player.chips for player in self.players) / len(self.players)
    


    def get_game_state(self) -> Dict:
        """Get the current game state."""
        return {
            'pot_size': self.pot,
            'community_cards': self.community_cards,
            'current_bet': self.current_bet,
            'player_chips': {player.name: player.chips for player in self.players},
            'player_positions': {player.name: (self.dealer_index + i) % len(self.players) for i, player in enumerate(self.players)},
            'bounties': {player.name: player.bounty for player in self.players},
            'current_stage' : self.current_stage,
            'last_action' : self.last_actions,
            'average_stack' : self.get_average_stack()
        }