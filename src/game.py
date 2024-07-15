from deck import Deck
from player import Player

class Game:
    def __init__(self, players, small_blind, big_blind):
        self.deck = Deck()
        self.players = players
        self.pot = 0
        self.community_cards = []
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.current_bet = 0
        self.current_player_index = 0
        self.dealer_index = 0
        
        def deal_hands(self):
            for player in self.players:
                player.hand.clear()
                player.hand = self.deck.deal(2)
        
        def deal_community_cards(self, num_cards):
            self.community_cards.extend(self.deck.deal(num_cards))
        
        def collect_bets(self):
            pass
        
        def post_blinds(self):
            small_blind_player = self.player[(self.dealer_index + 1) % len(self.players)]
            big_blind_player = self.player[(self.dealer_index + 2) % len(self.players)]
            
            small_blind_player.bet(self.small_blind)
            big_blind_player.bet(self.big_blind)
            
            self.pot += self.small_blind + self.big_blind
            self.current_bet = self.big_blind
            self.current_player_index = (self.dealer_index) % len(self.players)
        
        def betting_rounds(self):
            active_players = [player for player in self.players if not player.has_folded]
            for player in active_players:
                if not player.has_folded:
                    action = player.decide(self.get_game_state())

                    if action == 'bet':
                        bet_amount = player.bet(self.current_bet)
                        self.pot += bet_amount
                    elif action == 'check':
                        player.check()
                    elif action == 'fold':
                        player.fold()
                    elif action == 'raise':
                        raise_amount = player.bet(self.current_bet * 2) # beispiel raise
                        self.pot += raise_amount
                        self.current_bet = raise_amount



        def move_dealer(self):
            self.dealer_index  = (self.dealer_index + 1) % len(self.players)
        
        def play_round(self):
            self.deck.shuffle()
            self.deal_hands()
            self.post_blinds()

            # pre-flop betting
            self.deal_community_cards(3)
            self.betting_round()
            self.collect_bets()

            # turn
            self.deal_community_cards(1)
            self.betting_round()
            self.collect_bets()

            # river
            self.deal_community_cards(1)
            self.betting_round()
            self.collect_bets()

            # determine winner
            self.determine_winner()
            self.move_dealer()
        
        
        def determine_winner(self):
            best_hand = None
            winner = None
            for player in self.players:
                if not player.has_folded:
                    player_hand = player.hand.evaluate(self.community_cards)
                    
                    if not best_hand or player_hand > best_hand:
                        best_hand = player_hand
                        winner = player
                    
                    if winner:
                        winner.chips += self.pot
                    self.pot = 0
        
        def remove_player(self, player):
            self.players.remove(player)
        
        def game_state(self):
            pass
        