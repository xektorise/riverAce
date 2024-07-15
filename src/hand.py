from collections import Counter

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

        rank_order = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        ranks = sorted([rank_order[card[0]] for card in all_cards], reverse=True)
        suits = [card[1] for card in all_cards]
        
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush = max(suit_counts.values()) >= 5
        is_straight = self._is_straight(ranks)
        
        # straight flush
        if is_flush and is_straight:
            return 'Straight Flush', ranks

        # four of a kind
        if 4 in rank_counts.values():
            return 'Four of a Kind', [rank for rank, count in rank_counts.items() if count == 4] + ranks

        # full house
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            return 'Full House', [rank for rank, count in rank_counts.items() if count == 3] + [rank for rank, count in rank_counts.items() if count == 2]

        # flush
        if is_flush:
            return 'Flush', ranks

        # straight
        if is_straight:
            return 'Straight', ranks

        # three of a kind
        if 3 in rank_counts.values():
            return 'Three of a Kind', [rank for rank, count in rank_counts.items() if count == 3] + ranks

        # two pair
        if list(rank_counts.values()).count(2) >= 2:
            return 'Two Pair', [rank for rank, count in rank_counts.items() if count == 2] + ranks
        
        # one pair
        if 2 in rank_counts.values():
            return 'One Pair', [rank for rank, count in rank_counts.items() if count == 2] + ranks
        
        # high card
        return 'High Card', ranks

    def _is_straight(self, ranks):
        unique_ranks = list(set(ranks))
        unique_ranks.sort()
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i + 4] - unique_ranks[i] == 4:
                return True
        return False
