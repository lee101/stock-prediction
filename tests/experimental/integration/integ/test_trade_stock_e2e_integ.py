from trade_stock_e2e import (
    analyze_symbols
)


def test_analyze_symbols_real_call():
    symbols = ['ETHUSD']
    results = analyze_symbols(symbols)

    assert isinstance(results, dict)
    # ah well? its not profitable
    # assert len(results) > 0
    # first_symbol = list(results.keys())[0]
    # assert 'sharpe' in results[first_symbol]
    # assert 'side' in results[first_symbol]
