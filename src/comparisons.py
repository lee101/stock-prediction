"""Utility functions for comparing trading-related values."""

def is_same_side(side1: str, side2: str) -> bool:
    """
    Compare position sides accounting for different nomenclature.
    Handles 'buy'/'long' and 'sell'/'short' equivalence.
    
    Args:
        side1: First position side
        side2: Second position side
    Returns:
        bool: True if sides are equivalent
    """
    buy_variants = {'buy', 'long'}
    sell_variants = {'sell', 'short'}
    
    side1 = side1.lower()
    side2 = side2.lower()
    
    if side1 in buy_variants and side2 in buy_variants:
        return True
    if side1 in sell_variants and side2 in sell_variants:
        return True
    return False 