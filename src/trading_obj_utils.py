from typing import Iterable, List, Any

from src.fixtures import crypto_symbols


PositionLike = Any


def filter_to_realistic_positions(all_positions: Iterable[PositionLike]) -> List[PositionLike]:
    positions: List[PositionLike] = []
    for position in all_positions:
        if position.symbol in ['LTCUSD'] and float(position.qty) >= .1:
            positions.append(position)
        elif position.symbol in ['ETHUSD'] and float(position.qty) >= .01:
            positions.append(position)
        elif position.symbol in ['BTCUSD'] and float(position.qty) >= .001:
            positions.append(position)
        elif position.symbol in ["UNIUSD"] and float(position.qty) >= 5:
            positions.append(position)
        elif position.symbol not in crypto_symbols:
            positions.append(position)
    return positions
