from src.fixtures import crypto_symbols


def filter_to_realistic_positions(all_positions):
    positions = []
    for position in all_positions:
        if position.symbol in ['LTCUSD'] and float(position.qty) >= .1:
            positions.append(position)
        elif position.symbol in ['ETHUSD'] and float(position.qty) >= .01:
            positions.append(position)
        elif position.symbol in ['BTCUSD'] and float(position.qty) >= .001:
            positions.append(position)
        elif position.symbol in ["UNIUSD"] and float(position.qty) >= 5:
            positions.append(position)
        elif position.symbol in ['PAXGUSD']:
            positions.append(position)  # todo workout reslution for these
        elif position.symbol not in crypto_symbols:
            positions.append(position)
    return positions
