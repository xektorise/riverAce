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
            for _ in range(len(self.players)):
                player = self.player[self.current_player_index % len(self.players)]
                self.current_player_index += 1
                # more to come
        
        def move_dealer(self):
            pass
        
        def play_round(self):
            self.deck.shuffle()
            self.deal_hands()
            self.post_blinds()
            
            #spielrunden implementieren (preflop, flop, turn, river)
        
        
        def determine_winner(self):
            pass
        
        def remove_player(self):
            pass
        