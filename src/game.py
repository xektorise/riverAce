import logging
from typing import Dict, List, Tuple
from deck import Deck
from player import Player
from bot import Bot
from card import Card

class Game:
    
    STAGES = ['pre-flop', 'flop', 'turn', 'river']
    
    
    def __init__(self, players: List[Player], small_blind: int, big_blind: int, ante: int = ante):
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
        self.ante = ante
    
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler('poker_game.log')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

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
        
    def post_ante(self):
        for player in self.players:
            self.handle_insufficient_chips(player, self.ante, "ante")
        self.logger.info(f"Ante posted. Pot size: {self.pot}")

    def post_blinds(self):
        small_blind_player = self.players[(self.dealer_index + 1) % len(self.players)]
        big_blind_player = self.players[(self.dealer_index + 2) % len(self.players)]

        self.handle_insufficient_chips(small_blind_player, self.small_blind, "small blind")
        self.handle_insufficient_chips(big_blind_player, self.big_blind, "big blind")

        self.current_bet = self.big_blind
        self.current_player_index = (self.dealer_index + 3) % len(self.players)
        self.logger.info(f"Blinds posted. SB: {small_blind_player}, BB: {big_blind_player}")
        
    def increase_blinds(self, new_small_blind: int, new_big_blind: int, new_ante: int):
        self.small_blind = new_small_blind
        self.big_blind = new_big_blind
        self.ante = new_ante
        self.logger.info(f"Blinds increased to {self.small_blind}/{self.big_blind}, ante {self.ante}")

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
            
            if action == 'check' and 'raise' in self._get_avaiable_actions(current_player):
                self.logger.info(f"Player {current_player.name} checked")
                check_raise_action = self._get_player_action(current_player, first_bet_made, allow_check_raise=True)
                if check_raise_action == 'raise':
                    action = check_raise_action
                
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
        available_actions = self._get_available_actions(player)  
        
        game_state = self.get_game_state()
        return player.decide(game_state, options=available_actions)      

    def _get_available_actions(self, player: Player) -> List[str]:
        available_actions = []

        # Check if the player has enough chips to participate
        if player.chips == 0:
            return ['fold']

        # Determine the cost to call
        call_cost = self.current_bet - player.current_bet

        # Check if the player can call
        if call_cost == 0:
            available_actions.append('check')
        elif call_cost < player.chips:
            available_actions.append('call')

        # Check if the player can raise
        min_raise = self.current_bet * 2 - player.current_bet
        if player.chips > min_raise:
            available_actions.append('raise')

        # Check if the player can bet (only if no bets have been made)
        if self.current_bet == 0 and player.chips >= self.big_blind:
            available_actions.append('bet')

        # Player can always fold
        available_actions.append('fold')

        # Check if the player can go all-in
        if player.chips > 0:
            available_actions.append('all-in')

        # Special case: if the player can't call but has chips, they can only go all-in
        if call_cost >= player.chips and 'all-in' in available_actions:
            return ['fold', 'all-in']

        return available_actions

    
    def _process_player_action(self, player: Player, action: str):
        if action == 'bet':
            bet_amount = min(player.chips, self.big_blind)
            player.bet(bet_amount)
            self.current_bet = bet_amount
            self.logger.info(f"Player {player.name} bet {bet_amount}")

        elif action == 'check':
            player.check()
            self.logger.info(f"Player {player.name} checked")

        elif action == 'raise':
            min_raise = self.current_bet * 2 - player.current_bet
            raise_amount = min(player.chips, max(min_raise, player.raise_bet(self.pot, self.current_bet, min_raise)))
            if raise_amount == player.chips:
                self.handle_all_in(player)
            else:
                player.bet(raise_amount - player.current_bet)
                self.current_bet = raise_amount
                self.logger.info(f"Player {player.name} raised to {raise_amount}")

        elif action == 'call':
            call_amount = min(self.current_bet - player.current_bet, player.chips)
            if call_amount == player.chips:
                self.handle_all_in(player)
            else:
                player.bet(call_amount)
                self.logger.info(f"Player {player.name} called {call_amount}")

        elif action == 'fold':
            player.fold()
            self.logger.info(f"Player {player.name} folded")

        elif action == 'all-in':
            self.handle_all_in(player)

        self.handle_short_stack(player)
    
    
    
    def handle_side_pots(self):
        players_all_in = [p for p in self.players if p.chips == 0 and not p.has_folded]
        if not players_all_in:
            return

        players_all_in.sort(key=lambda p: p.current_bet)
        remaining_players = [p for p in self.players if p.chips > 0 and not p.has_folded]

        for i, all_in_player in enumerate(players_all_in):
            side_pot = 0
            for player in self.players:
                if player.current_bet > all_in_player.current_bet:
                    contribution = all_in_player.current_bet if i == 0 else all_in_player.current_bet - players_all_in[i-1].current_bet
                    side_pot += contribution
                    player.current_bet -= contribution

            if side_pot > 0:
                self.side_pots.append((side_pot, [all_in_player] + remaining_players))

        # Main pot
        main_pot = sum(player.current_bet for player in self.players)
        self.pot = main_pot + sum(pot for pot, _ in self.side_pots) 

    def move_dealer(self):
        """Move the dealer button to the next player."""
        self.dealer_index = (self.dealer_index + 1) % len(self.players)
        self.logger.info(f"Dealer moved to player {self.players[self.dealer_index].name}")
    
    def play_round(self):
        try:
            self.logger.info("New round started")
            self.last_actions.clear()
            self.deck.shuffle()
            self.update_positions()
            self.post_ante()
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

                # Check if the hand is over after each betting round
                active_players = [p for p in self.players if not p.has_folded]
                if len(active_players) == 1:
                    self.award_pot(active_players)
                    self.logger.info(f"Hand ended early. {active_players[0].name} wins.")
                    return active_players

            winners = self.determine_winner()
            self.move_dealer()
            self.reset_for_new_round()
            return winners
        except ValueError as ve:
            self.logger.error(f"ValueError during round: {str(ve)}")
        except IndexError as ie:
            self.logger.error(f"IndexError during round: {str(ie)}")
        except AttributeError as ae:
            self.logger.error(f"AttributeError during round: {str(ae)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during round: {str(e)}")

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

        # Distribute bounties and eliminate players with zero chips
        self.handle_eliminations()

        self.pot = 0
        self.side_pots.clear()
        self.logger.info(f"Round ended. Winners: {[winner.name for winner in winners]}")

        return winners

    def split_pot(self, winners: List[Player]):
        """Split the pot among the winners."""
        if not self.side_pots:
            # If no side pots, split the main pot
            pot_share = self.pot // len(winners)
            remainder = self.pot % len(winners)
            for winner in winners:
                winner.chips += pot_share
                self.logger.info(f"{winner.name} wins {pot_share} chips")
        else:
            # Handle side pots
            for side_pot, eligible_players in self.side_pots:
                pot_winners = [w for w in winners if w in eligible_players]
                if pot_winners:
                    pot_share = side_pot // len(pot_winners)
                    remainder = side_pot % len(pot_winners)
                    for winner in pot_winners:
                        winner.chips += pot_share
                        self.logger.info(f"{winner.name} wins {pot_share} chips from side pot")

                # Distribute remainder chips
                sorted_winners = sorted(pot_winners, key=lambda p: (self.dealer_index - self.players.index(p)) % len(self.players))
                for i in range(remainder):
                    sorted_winners[i].chips += 1

        self.logger.info(f"Pot split among {len(winners)} players")
        
    
    def handle_eliminations(self):
        eliminated_players = [player for player in self.players if player.chips == 0]
        for eliminated_player in eliminated_players:
            eliminator = max((p for p in self.players if not p.has_folded), key=lambda p: p.current_bet)
            bounty_won = eliminated_player.bounty // 2  # half the bounty goes to the eliminator
            eliminator.chips += bounty_won
            eliminator.bounty += bounty_won  # the eliminator's bounty increases
            eliminator.add_bounty_won(bounty_won)
            self.logger.info(f"{eliminator.name} won {bounty_won} bounty from {eliminated_player.name}")
            self.remove_player(eliminated_player)
    
    def update_positions(self):
        num_players = len(self.players)
        self.players[self.dealer_index].position = "BUTTON"
        self.players[(self.dealer_index + 1) % num_players].position = "SMALL_BLIND"
        self.players[(self.dealer_index + 2) % num_players].position = "BIG_BLIND"

        for i in range(3, num_players):
            position_index = (self.dealer_index + i) % num_players
            if i == num_players - 1:
                self.players[position_index].position = "CUTOFF"
            elif i == 3:
                self.players[position_index].position = "UNDER_THE_GUN"
            else:
                self.players[position_index].position = "MIDDLE_POSITION"

        self.logger.info("Player positions updated")
        
    def handle_insufficient_chips(self, player: Player, required_amount: int, action: str):
        if player.chips < required_amount:
            self.logger.info(f"{player.name} doesn't have enough chips for {action}. Going all-in.")
            self.handle_all_in(player)
        else:
            player.bet(required_amount)
            self.pot += required_amount

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
    
    
    def calculate_icm(self):
        total_chips = sum(player.chips for player in self.players)
        icm_values = {}
        for player in self.players:
            icm_values[player.name] = player.chips / total_chips
        return icm_values
    
    
    def handle_all_in(self, player: Player):
        all_in_amount = player.all_in()
        if all_in_amount > self.current_bet:
            self.current_bet = all_in_amount
        self.handle_side_pots()
        self.logger.info(f"Player {player.name} went all-in for {all_in_amount}")
    
    
    def handle_short_stack(self, player: Player):
        if 0 < player.chips < self.big_blind:
            self.logger.info(f"{player.name} has less than the big blind, forcing all-in")
            self.handle_all_in(player)
            
    def end_tournament(self):
        """Handle the end of the tournament."""
        final_standings = sorted(self.players, key=lambda p: p.chips, reverse=True)

        self.logger.info("Tournament Ended. Final Standings:")
        for i, player in enumerate(final_standings, 1):
            total_winnings = player.chips + player.total_bounty_won
            self.logger.info(f"{i}. {player.name}: {player.chips} chips, {player.bounty} bounty, Total Winnings: {total_winnings}")

        winner = final_standings[0]
        total_bounty = sum(player.bounty for player in self.players)
        winner.chips += total_bounty

        self.logger.info(f"Tournament Winner: {winner.name}")
        self.logger.info(f"Final Chip Count: {winner.chips}")
        self.logger.info(f"Total Bounty Collected: {winner.total_bounty_won}")
        self.logger.info(f"Total Prize: {winner.chips} chips")

        return winner

    def is_game_over(self) -> bool:
        return len([p for p in self.players if p.chips > 0]) <= 1


    def get_game_state(self) -> Dict:
        """Get the current game state."""
        return {
            'pot_size': self.pot,
            'community_cards': self.community_cards,
            'current_bet': self.current_bet,
            'player_chips': {player.name: player.chips for player in self.players},
            'player_position': {player.name: (self.dealer_index + i) % len(self.players) for i, player in enumerate(self.players)},
            'bounties': {player.name: player.bounty for player in self.players},
            'current_stage' : self.current_stage,
            'last_actions' : self.last_actions,
            'average_stack' : self.get_average_stack(),
            'min_raise': self.current_bet * 2 - self.players[self.current_player_index].current_bet if self.players else 0,
            'players_in_hand' : [p.name for p in self.players if not p.has_folded]
        }