from collections import Counter
from card import Card

class Hand:
    def __init__(self):
        self.hole_cards = []
        self.community_cards = []

    def add_hole_cards(self, new_cards):
        self.hole_cards.extend(new_cards)

    def add_community_cards(self, new_cards):
        self.community_cards.extend(new_cards)

    def clear(self):
        self.hole_cards = []
        self.community_cards = []

    def evaluate(self):
        all_cards = self.hole_cards + self.community_cards
        
        if len(all_cards) < 5:
            return 'Incomplete', []

        rank_order = {rank: index for index, rank in enumerate(Card.ranks, start=2)}
        ranks = sorted([rank_order[card.rank] for card in all_cards], reverse=True)
        suits = [card.suit for card in all_cards]
        
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush, flush_suit = self._is_flush(suits)
        is_straight, straight_ranks = self._is_straight(ranks)
        
        # Evaluate hand types
        if is_flush and is_straight and self._is_straight_flush(all_cards, flush_suit, straight_ranks, rank_order):
            return 'Straight Flush', straight_ranks
        if 4 in rank_counts.values():
            return 'Four of a Kind', self._get_rank_hand(rank_counts, 4) + self._get_highest_hand(ranks, rank_counts, 1)
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            return 'Full House', self._get_rank_hand(rank_counts, 3) + self._get_rank_hand(rank_counts, 2, 1)
        if is_flush:
            return 'Flush', self._get_highest_hand(ranks, rank_counts, 5, flush=True, flush_suit=flush_suit)
        if is_straight:
            return 'Straight', straight_ranks
        if 3 in rank_counts.values():
            return 'Three of a Kind', self._get_rank_hand(rank_counts, 3) + self._get_highest_hand(ranks, rank_counts, 2)
        if list(rank_counts.values()).count(2) >= 2:
            return 'Two Pair', self._get_rank_hand(rank_counts, 2, pair_count=2) + self._get_highest_hand(ranks, rank_counts, 1)
        if 2 in rank_counts.values():
            return 'One Pair', self._get_rank_hand(rank_counts, 2) + self._get_highest_hand(ranks, rank_counts, 3)
        
        return 'High Card', self._get_highest_hand(ranks, rank_counts, 5)

    def _is_flush(self, suits):
        suit_counts = Counter(suits)
        for suit, count in suit_counts.items():
            if count >= 5:
                return True, suit
        return False, None
    
    def _is_straight(self, ranks):
        unique_ranks = list(set(ranks))
        unique_ranks.sort()
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i + 4] - unique_ranks[i] == 4:
                return True, unique_ranks[i:i + 5]
        # Check for Ace-low straight (A-2-3-4-5)
        if unique_ranks[-1] == 14 and set(unique_ranks[:4]) == {2, 3, 4, 5}:
            return True, [5, 4, 3, 2, 14]
        return False, []

    def _is_straight_flush(self, cards, flush_suit, straight_ranks, rank_order):
        flush_cards = [card for card in cards if card.suit == flush_suit]
        flush_ranks = sorted([rank_order[card.rank] for card in flush_cards], reverse=True)
        return set(straight_ranks).issubset(set(flush_ranks))
    
    def _get_highest_hand(self, ranks, counts, num, flush=False, flush_suit=None):
        if flush:
            return sorted([rank for rank in ranks if counts[rank] >= 1 and flush_suit], reverse=True)[:num]
        return [rank for rank in ranks if counts[rank] == 1][:num]

    def _get_rank_hand(self, rank_counts, count, pair_count=1):
        return sorted([rank for rank, cnt in rank_counts.items() if cnt == count], reverse=True)[:pair_count]
    
    def _get_straight_hand(self, ranks):
        unique_ranks = list(set(ranks))
        unique_ranks.sort(reverse=True)
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i + 4] == 4:
                return unique_ranks[i:i + 5]
        # Check for Ace-low straight (A-2-3-4-5)
        if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
            return [5, 4, 3, 2, 14]
        return []