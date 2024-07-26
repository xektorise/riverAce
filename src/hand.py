from typing import List, Tuple
from collections import Counter
from card import Card, Rank, Suit
import itertools

class Hand:
    
    HAND_RANKINGS = {
        'High Card': 1,
        'One Pair': 2,
        'Two Pair': 3,
        'Three of a Kind': 4,
        'Straight': 5,
        'Flush': 6,
        'Full House': 7,
        'Four of a Kind': 8,
        'Straight Flush': 9,
        'Royal Flush': 10
    }
    
    def __init__(self):
        self.hole_cards: List[Card] = []
        self.community_cards: List[Card] = []
        
    def __lt__(self, other: 'Hand') -> bool:
        return self.get_hand_strength() < other.get_hand_strength()

    def __eq__(self, other: 'Hand') -> bool:
        return self.get_hand_strength() == other.get_hand_strength()
    
    def __str__(self) -> str:
        return f"Hole cards: {', '.join(str(card) for card in self.hole_cards)}, " \
               f"Community cards: {', '.join(str(card) for card in self.community_cards)}"

    def __repr__(self) -> str:
        return f"Hand(hole_cards={self.hole_cards}, community_cards={self.community_cards})"

    def add_hole_cards(self, new_cards: List[Card]) -> None:
        """Add hole cards to the hand."""
        self.hole_cards.extend(new_cards)

    def add_community_cards(self, new_cards: List[Card]) -> None:
        """Add community cards to the hand."""
        self.community_cards.extend(new_cards)

    def clear(self) -> None:
        """Clear the current hand."""
        self.hole_cards.clear()
        self.community_cards.clear()

    def evaluate(self) -> Tuple[str, List[Rank]]:
        """Evaluate the hand and return its rank and the highest cards involved."""
        all_cards = self.hole_cards + self.community_cards
        
        if len(all_cards) < 5:
            return 'Incomplete', []

        ranks = [card.rank.value for card in all_cards]
        suits = [card.suit.value for card in all_cards]
        
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush = max(suit_counts.values(), default=0) >= 5
        is_straight = self._is_straight(ranks)
        
        if is_flush and is_straight:
            straight_ranks = self._get_straight_ranks(ranks)
            if straight_ranks == [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN]:  # ace, king, queen, jack, ten
                return 'Royal Flush', straight_ranks
            return 'Straight Flush', straight_ranks
        if 4 in rank_counts.values():
            return 'Four of a Kind', self._get_rank_hand(rank_counts, 4, 1)
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            return 'Full House', self._get_rank_hand(rank_counts, 3, 2)
        if is_flush:
            return 'Flush', sorted(ranks, reverse=True)[:5]
        if is_straight:
            return 'Straight', self._get_straight_ranks(ranks)
        if 3 in rank_counts.values():
            return 'Three of a Kind', self._get_rank_hand(rank_counts, 3, 2)
        if list(rank_counts.values()).count(2) >= 2:
            return 'Two Pair', self._get_rank_hand(rank_counts, 2, 3)
        if 2 in rank_counts.values():
            return 'One Pair', self._get_rank_hand(rank_counts, 2, 3)
        
        return 'High Card', sorted(ranks, reverse=True)[:5]

    def _is_straight(self, ranks: List[Rank]) -> bool:
        unique_ranks = sorted(set(ranks), key=lambda r: r.value)
        if len(unique_ranks) < 5:
            return False
        if Rank.ACE in unique_ranks and Rank.TWO in unique_ranks:
            unique_ranks.append(Rank.ACE)  # add a low ace
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i+4].value - unique_ranks[i].value == 4:
                return True
        return False
    
    def _get_straight_ranks(self, ranks: List[Rank]) -> List[Rank]:
        unique_ranks = sorted(set(ranks), reverse=True)
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i].value - unique_ranks[i+4].value == 4:
                return unique_ranks[i:i+5]
        # check for Ace-low straight (A-2-3-4-5)
        if set(unique_ranks[:4]) == {Rank.FIVE, Rank.FOUR, Rank.THREE, Rank.TWO} and Rank.ACE in unique_ranks:
            return [Rank.FIVE, Rank.FOUR, Rank.THREE, Rank.TWO, Rank.ACE]
        return []
    
    def _get_rank_hand(self, rank_counts: Counter, primary_count: int, kicker_count: int) -> List[Rank]:
        primary_ranks = [Rank(rank) for rank, count in rank_counts.items() if count == primary_count]
        kicker_ranks = [Rank(rank) for rank, count in rank_counts.items() if count < primary_count]
        return sorted(primary_ranks, reverse=True) + sorted(kicker_ranks, reverse=True)[:kicker_count]
    
    def get_hand_strength(self) -> float:
        hand_rank, hand_cards = self.evaluate()
        base_score = self.HAND_RANKINGS[hand_rank] * 1000000
        for i, card in enumerate(hand_cards):
            base_score += card.value * (100 ** (4 - i))
        return base_score / 10000000  # normalize to 0-1 range for comparisons
    
    def get_best_hand(self) -> List[Card]:
        all_cards = self.hole_cards + self.community_cards
        best_hand_strength = 0
        best_hand = []
        for combo in itertools.combinations(all_cards, 5):
            hand = Hand()
            hand.add_hole_cards(list(combo))
            strength = hand.get_hand_strength()
            if strength > best_hand_strength:
                best_hand_strength = strength
                best_hand = list(combo)
        return best_hand
    
    def describe_hand(self) -> str:
        hand_type, _ = self.evaluate()
        return f"{hand_type}: {', '.join(str(card) for card in self.get_best_hand())}"