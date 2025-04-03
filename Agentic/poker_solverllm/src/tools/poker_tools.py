"""
Poker evaluation tools using PokerKit.
"""

import traceback
from typing import List, Tuple, Set, FrozenSet

# PokerKit imports
import pokerkit
from pokerkit.analysis import calculate_equities, parse_range
from pokerkit.hands import StandardHighHand
from pokerkit.utilities import Card, Deck

# LangChain tool decorator
from langchain_core.tools import tool

def _parse_cards(card_string: str) -> Tuple[Card, ...]:
    """
    Parses a string of cards into a tuple of Card objects.
    
    Args:
        card_string: String representation of cards (e.g., 'AcKc')
        
    Returns:
        Tuple of Card objects
    """
    if not card_string:
        return tuple()
    
    try:
        # Use Card.parse which returns a generator, so convert to tuple
        return tuple(Card.parse(card_string))
    except ValueError as e:
        # Handle potential errors if string isn't valid cards
        raise ValueError(f"Invalid card string: '{card_string}'. Error: {e}")
    except AttributeError:
        # Fallback implementation if Card.parse is not available
        print("WARNING: Card.parse not available, using fallback implementation")
        cards = []
        # Parse cards two characters at a time (rank + suit)
        if len(card_string) % 2 != 0:
            raise ValueError(f"Card string must have even length: '{card_string}'")
        
        for i in range(0, len(card_string), 2):
            rank_char = card_string[i]
            suit_char = card_string[i+1]
            
            # Map characters to Rank and Suit objects
            rank_map = {'2': pokerkit.utilities.Rank.DEUCE, 
                       '3': pokerkit.utilities.Rank.TREY,
                       '4': pokerkit.utilities.Rank.FOUR,
                       '5': pokerkit.utilities.Rank.FIVE,
                       '6': pokerkit.utilities.Rank.SIX,
                       '7': pokerkit.utilities.Rank.SEVEN,
                       '8': pokerkit.utilities.Rank.EIGHT,
                       '9': pokerkit.utilities.Rank.NINE,
                       'T': pokerkit.utilities.Rank.TEN,
                       'J': pokerkit.utilities.Rank.JACK,
                       'Q': pokerkit.utilities.Rank.QUEEN,
                       'K': pokerkit.utilities.Rank.KING,
                       'A': pokerkit.utilities.Rank.ACE}
            
            suit_map = {'c': pokerkit.utilities.Suit.CLUB,
                       'd': pokerkit.utilities.Suit.DIAMOND,
                       'h': pokerkit.utilities.Suit.HEART,
                       's': pokerkit.utilities.Suit.SPADE}
            
            if rank_char not in rank_map:
                raise ValueError(f"Invalid rank character: '{rank_char}'")
            if suit_char not in suit_map:
                raise ValueError(f"Invalid suit character: '{suit_char}'")
                
            rank = rank_map[rank_char]
            suit = suit_map[suit_char]
            
            cards.append(Card(rank, suit))
            
        return tuple(cards)
    except Exception as e:
        raise ValueError(f"Error parsing card string '{card_string}': {e}")


@tool
def evaluate_poker_hand(hole_cards_string: str, board_string: str = "") -> str:
    """
    Evaluates the best possible standard high poker hand given hole cards and board cards.
    
    Args:
        hole_cards_string: The player's hole cards (e.g., 'AcKc'). Must be specific cards.
        board_string: The community cards on the board (e.g., 'QcJcTc'). Can be empty.
        
    Returns:
        A string describing the hand's rank and category (e.g., 'Straight Flush', 'Pair of Aces').
    """
    try:
        hole_cards = _parse_cards(hole_cards_string)
        board_cards = _parse_cards(board_string)
        
        if len(hole_cards) < 2:  # Need at least hole cards
            return "Error: Not enough cards provided for evaluation."
        if len(hole_cards) != 2:  # Assuming typical Hold'em input
            return f"Input specifies {len(hole_cards)} hole cards. Evaluation typically assumes 2 (Hold'em)."

        # StandardHighHand evaluates the best 5-card hand from the input cards
        if (len(hole_cards) + len(board_cards)) < 5:
            hand_type = "Pre-flop Hand" if not board_cards else "Incomplete Hand"
            return f"{hand_type}: {', '.join(map(str, hole_cards))} on board [{', '.join(map(str, board_cards))}]. Not enough cards for full 5-card evaluation."

        # Use from_game method to correctly evaluate based on hole cards and board
        try:
            evaluated_hand = StandardHighHand.from_game(hole_cards_string, board_string)
        except AttributeError:
            # Fallback if from_game is not available
            all_cards = hole_cards + board_cards
            evaluated_hand = StandardHighHand(all_cards)

        # The __str__ representation of StandardHighHand often gives a good description
        hand_description = str(evaluated_hand)

        return f"Best 5-card hand for {hole_cards_string} on board [{board_string}]: {hand_description}"

    except ValueError as ve:
        return f"Input Error: {ve}. Please provide specific cards like 'AsKd' or '2c3d4h5s6c'."
    except Exception as e:
        # Print stack trace for debugging
        traceback.print_exc()
        return f"Error evaluating hand: {e}"


@tool
def parse_hand_range(range_string: str) -> str:
    """
    Parses a poker hand range string into a set of specific hands.
    
    Args:
        range_string: The range notation (e.g., 'AJ+', '77-TT', 'AKo', 'QJs', '8h7h')
                     Multiple ranges separated by spaces or semicolons are allowed.
                     
    Returns:
        A summary string indicating the number of combinations in the parsed range(s).
    """
    try:
        # parse_range can handle multiple space/semicolon separated ranges
        parsed_hands_set: Set[FrozenSet[Card]] = parse_range(range_string)
        count = len(parsed_hands_set)
        return f"Parsed range '{range_string}': Contains {count} hand combinations."
    except Exception as e:
        return f"Error parsing range '{range_string}': {e}. Please use standard poker range notation (e.g., 'AKs', 'QQ+', 'AsKh')."


@tool
def calculate_holdem_equity(player_ranges: List[str], board_string: str = "", num_simulations: int = 10000) -> str:
    """
    Calculates Hold'em win/tie probability (equity) for players given their hand ranges and the board.
    
    Args:
        player_ranges: A list of strings, where each string is a player's hand range
                      (e.g., ['AsKh', 'QQ+', 'AKs,AQs,KQs']). Specific hands or range notation allowed.
        board_string: The community cards currently on the board (e.g., '8h9hTh'). Can be empty for pre-flop.
        num_simulations: The number of random deals to simulate. Default 10000. Max 50000 recommended.
        
    Returns:
        A string summarizing the equity for each player range.
    """
    from src.constants.config import MAX_SIMULATIONS
    
    if num_simulations > MAX_SIMULATIONS:
        num_simulations = MAX_SIMULATIONS
        print(f"Warning: Reduced simulations to {MAX_SIMULATIONS}.")
    if num_simulations <= 0:
        return "Error: Number of simulations must be positive."

    try:
        # Parse board cards
        board_cards = _parse_cards(board_string)
        if len(board_cards) > 5:
            return "Error: Board cannot have more than 5 cards in Hold'em."

        # Parse player ranges
        parsed_player_ranges: List[Set[FrozenSet[Card]]] = []
        for i, r_str in enumerate(player_ranges):
            try:
                parsed_range = parse_range(r_str)
                if not parsed_range:
                    return f"Error: Player {i+1}'s range '{r_str}' parsed to empty set. Check range notation and conflicts with board."
                parsed_player_ranges.append(parsed_range)
            except Exception as e:
                return f"Error parsing range for Player {i+1} ('{r_str}'): {e}"

        # Define parameters for calculate_equities
        hole_card_count = 2
        board_card_count = 5  # Total board cards in Hold'em
        deck_type = Deck.STANDARD
        hand_types = (StandardHighHand,)  # Using standard high hand evaluation

        # Check for obvious card conflicts
        all_known_cards = set(board_cards)
        for i, p_range in enumerate(parsed_player_ranges):
            # Check specific hands for conflicts
            if len(p_range) == 1:  # Specific hand provided
                hand = next(iter(p_range))  # Get the single hand frozenset
                if not hand.isdisjoint(all_known_cards):
                    hand_str = "".join(map(str, hand))
                    return f"Error: Player {i+1}'s hand '{hand_str}' conflicts with board or another known hand."
                all_known_cards.update(hand)

        # Call calculate_equities
        equities = calculate_equities(
            ranges=tuple(parsed_player_ranges),
            board=board_cards,
            hole_card_count=hole_card_count,
            board_card_count=board_card_count,
            deck_type=deck_type,
            hand_types=hand_types,
            sample_count=num_simulations,
        )

        # Format results
        if not equities or len(equities) != len(player_ranges):
            return "Error: Equity calculation failed to return expected results."

        results = []
        for i, equity_val in enumerate(equities):
            equity_pct = equity_val * 100
            results.append(f"Player {i+1} (Range: '{player_ranges[i]}'): Equity ~{equity_pct:.2f}%")

        return (f"Hold'em equity estimation based on {num_simulations} simulations "
                f"(Board: [{board_string}]):\n" + "\n".join(results))

    except ValueError as ve:
        return f"Input Error: {ve}"
    except Exception as e:
        return f"An error occurred during equity calculation: {e}"


# Export all tools for the agent
all_tools = [evaluate_poker_hand, parse_hand_range, calculate_holdem_equity]

# Test function without langchain tool wrapping
def test_evaluate_hand(hole_cards_string: str, board_string: str = "") -> str:
    """
    Direct test function for evaluate_poker_hand without the LangChain tool wrapper.
    
    Args:
        hole_cards_string: The player's hole cards (e.g., 'AcKc')
        board_string: The community cards on the board (e.g., 'QcJcTc')
        
    Returns:
        A string describing the evaluation result
    """
    try:
        # Don't use the tool decorator function directly
        # Instead duplicate the logic to avoid tool calling overhead
        hole_cards = _parse_cards(hole_cards_string)
        board_cards = _parse_cards(board_string)
        
        if len(hole_cards) < 2:  # Need at least hole cards
            return "Error: Not enough cards provided for evaluation."
        if len(hole_cards) != 2:  # Assuming typical Hold'em input
            return f"Input specifies {len(hole_cards)} hole cards. Evaluation typically assumes 2 (Hold'em)."

        # StandardHighHand evaluates the best 5-card hand from the input cards
        if (len(hole_cards) + len(board_cards)) < 5:
            hand_type = "Pre-flop Hand" if not board_cards else "Incomplete Hand"
            return f"{hand_type}: {', '.join(map(str, hole_cards))} on board [{', '.join(map(str, board_cards))}]. Not enough cards for full 5-card evaluation."

        # Use from_game method to correctly evaluate based on hole cards and board
        try:
            evaluated_hand = StandardHighHand.from_game(hole_cards_string, board_string)
        except AttributeError:
            # Fallback if from_game is not available
            all_cards = hole_cards + board_cards
            evaluated_hand = StandardHighHand(all_cards)

        # The __str__ representation of StandardHighHand often gives a good description
        hand_description = str(evaluated_hand)

        return f"Best 5-card hand for {hole_cards_string} on board [{board_string}]: {hand_description}"
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"