#!/usr/bin/env python3
"""
Test script for poker_tools.py to check if hand evaluation works properly.
"""

import sys
import traceback
import pokerkit
from pokerkit.utilities import Card
from pokerkit.hands import StandardHighHand

from src.tools.poker_tools import test_evaluate_hand, _parse_cards


def main():
    """Main test function."""
    try:
        print("Testing pokerkit version information...")
        print(f"Pokerkit version: {getattr(pokerkit, '__version__', 'unknown')}")
        print(f"Card class: {Card}")
        print(f"StandardHighHand class: {StandardHighHand}")
        print(f"Has Card.parse: {hasattr(Card, 'parse')}")
        print(f"Has StandardHighHand.from_game: {hasattr(StandardHighHand, 'from_game')}")
        
        print("\nTesting card parsing...")
        test_cards = 'AcKc'
        cards = _parse_cards(test_cards)
        print(f"Parsed cards: {cards}")
        
        print("\nTesting hand evaluation...")
        result = test_evaluate_hand('AcKc', 'QcJcTcAd2s')
        print(f"Hand evaluation result: {result}")
        
        print("\nTesting Card.parse directly...")
        direct_cards = tuple(Card.parse('AcKc'))
        print(f"Direct parse result: {direct_cards}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())