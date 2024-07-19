from deck import Deck
from player import Player
from bot import Bot

class Game:
    def __init__(self, players, small_blind, big_blind):
        """Initialize the game with players, blinds, and a deck."""
        self.deck = Deck()
        self.players = players
        self.pot = 0
        self.community_cards = []
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.current_bet = 0
        self.current_player_index = 0
        self.dealer_index = 0
        self.side_pots = []

    def deal_hands(self):
        """Deal hole cards to each player."""
        for player in self.players:
            player.hand.clear()
            player.hand.add_hole_cards(self.deck.deal(2))
    
    def deal_community_cards(self, num_cards):
        """Deal community cards."""
        self.community_cards.extend(self.deck.deal(num_cards))
    
    def collect_bets(self):
        """Collect bets from all players and add them to the pot."""
        for player in self.players:
            self.pot += player.current_bet
            player.current_bet = 0
    
    def post_blinds(self):
        """Post small and big blinds."""
        small_blind_player = self.players[(self.dealer_index + 1) % len(self.players)]
        big_blind_player = self.players[(self.dealer_index + 2) % len(self.players)]
        
        small_blind_player.bet(self.small_blind)
        big_blind_player.bet(self.big_blind)
        
        self.current_bet = self.big_blind
        self.current_player_index = (self.dealer_index + 3) % len(self.players)
    
    def betting_round(self):
        """Conduct a single round of betting."""
        active_players = [player for player in self.players if not player.has_folded and player.chips > 0]
        first_bet_made = False

        while True:
            for player in active_players:
                if not player.has_folded and player.chips > 0:
                    if not first_bet_made:
                        action = player.decide(self.get_game_state(), options=['bet', 'fold', 'raise'])
                        if action == 'bet':
                            bet_amount = player.bet(self.big_blind)
                            self.current_bet = bet_amount
                            first_bet_made = True
                        elif action == 'raise':
                            raise_amount = player.raise_bet(self.pot, self.current_bet)
                            self.current_bet = raise_amount
                            first_bet_made = True
                        elif action == 'fold':
                            player.fold()
                    else:
                        action = player.decide(self.get_game_state(), options=['call', 'fold', 'raise', 'all_in'])
                        if action == 'raise':
                            raise_amount = player.raise_bet(self.pot, self.current_bet)
                            self.current_bet = raise_amount
                        elif action == 'call':
                            call_amount = self.current_bet - player.current_bet
                            if call_amount > player.chips:
                                player.all_in()
                            else:
                                player.bet(call_amount)
                        elif action == 'fold':
                            player.fold()
                        elif action == 'all_in':
                            all_in_amount = player.all_in()
                            self.handle_side_pots(player, all_in_amount)
                    
            self.collect_bets()

            active_players = [player for player in self.players if not player.has_folded and player.chips > 0]
            if all(player.current_bet >= self.current_bet for player in active_players):
                break
    
    
    def handle_side_pots(self, all_in_player, all_in_amount):
        """Handle side pots when a player goes all-in."""
        for player in self.players:
            if player != all_in_player and player.current_bet > all_in_amount:
                excess_bet = player.current_bet - all_in_amount
                player.current_bet -= excess_bet
                self.side_pots.append((excess_bet, player))

    def move_dealer(self):
        """Move the dealer button to the next player."""
        self.dealer_index = (self.dealer_index + 1) % len(self.players)
    
    def play_round(self):
        """Play a full round of poker."""
        self.deck.shuffle()
        self.deal_hands()
        self.post_blinds()

        # pre-flop betting
        self.betting_round()

        # flop
        self.deal_community_cards(3)
        self.betting_round()

        # turn
        self.deal_community_cards(1)
        self.betting_round()

        # river
        self.deal_community_cards(1)
        self.betting_round()

        # determine winner
        self.determine_winner()
        self.move_dealer()
    
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
            self.split_pot(winners)

        # distribute bounties and eliminate players with zero chips
        eliminated_players = [player for player in self.players if player.chips == 0]
        for eliminated_player in eliminated_players:
            if winners:
                winners[0].chips += eliminated_player.bounty 
            self.remove_player(eliminated_player)

        self.pot = 0
        self.side_pots.clear()

    def split_pot(self, winners):
        """Split the pot among the winners."""
        num_winners = len(winners)
        pot_share = self.pot // num_winners
        remainder = self.pot % num_winners

        for winner in winners:
            winner.chips += pot_share

        # distribute the remainder chips starting from the dealer
        for i in range(remainder):
            winners[(self.dealer_index + i) % num_winners].chips += 1
    
    def remove_player(self, player):
        """Remove a player from the game."""
        self.players = [p for p in self.players if p.name != player.name]
    
    def get_game_state(self):
        """Get the current game state."""
        return {
            'pot_size': self.pot,
            'community_cards': self.community_cards,
            'current_bet': self.current_bet,
            'player_chips': {player.name: player.chips for player in self.players},
            'player_positions': {player.name: (self.dealer_index + i) % len(self.players) for i, player in enumerate(self.players)},
            'bounties': {player.name: player.bounty for player in self.players}
        }