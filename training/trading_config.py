"""
Realistic Trading Cost Configurations
Based on actual broker fees and market conditions
"""

class TradingCosts:
    """Base class for trading costs"""
    def __init__(self):
        self.commission = 0.0
        self.min_commission = 0.0
        self.spread_pct = 0.0
        self.slippage_pct = 0.0


class CryptoTradingCosts(TradingCosts):
    """
    Realistic crypto trading costs based on major exchanges
    """
    def __init__(self, exchange='default'):
        super().__init__()
        
        if exchange == 'binance':
            # Binance spot trading fees
            self.commission = 0.001  # 0.1% (can be 0.075% with BNB)
            self.min_commission = 0.0  # No minimum
            self.spread_pct = 0.0001  # 0.01% typical for major pairs
            self.slippage_pct = 0.00005  # 0.005% for liquid pairs
            
        elif exchange == 'coinbase':
            # Coinbase Advanced Trade
            self.commission = 0.005  # 0.5% for smaller volumes
            self.min_commission = 0.0
            self.spread_pct = 0.0005  # 0.05% typical
            self.slippage_pct = 0.0001  # 0.01%
            
        else:  # Default realistic crypto
            self.commission = 0.0015  # 0.15% as you mentioned
            self.min_commission = 0.0
            self.spread_pct = 0.0002  # 0.02% for liquid pairs
            self.slippage_pct = 0.0001  # 0.01% minimal for liquid markets


class StockTradingCosts(TradingCosts):
    """
    Realistic stock trading costs based on modern brokers
    """
    def __init__(self, broker='default'):
        super().__init__()
        
        if broker == 'robinhood' or broker == 'alpaca':
            # Zero commission brokers (Robinhood, Alpaca, etc.)
            self.commission = 0.0  # $0 commission
            self.min_commission = 0.0
            # They make money from payment for order flow
            self.spread_pct = 0.00005  # 0.005% - very tight for liquid stocks
            self.slippage_pct = 0.00002  # 0.002% - minimal for liquid stocks
            
        elif broker == 'interactive_brokers':
            # Interactive Brokers (pro pricing)
            self.commission = 0.00005  # $0.005 per share, ~0.005% for $100 stock
            self.min_commission = 1.0  # $1 minimum
            self.spread_pct = 0.00001  # 0.001% - best execution
            self.slippage_pct = 0.00001  # 0.001% - minimal
            
        elif broker == 'td_ameritrade':
            # TD Ameritrade / Schwab
            self.commission = 0.0  # $0 for stocks
            self.min_commission = 0.0
            self.spread_pct = 0.00005  # 0.005%
            self.slippage_pct = 0.00002  # 0.002%
            
        else:  # Default modern stock broker
            self.commission = 0.0  # Most brokers are $0 commission now
            self.min_commission = 0.0
            self.spread_pct = 0.00003  # 0.003% - very tight spreads
            self.slippage_pct = 0.00002  # 0.002% - minimal slippage


class ForexTradingCosts(TradingCosts):
    """
    Realistic forex trading costs
    """
    def __init__(self):
        super().__init__()
        self.commission = 0.0  # Usually built into spread
        self.min_commission = 0.0
        self.spread_pct = 0.0001  # 1 pip for major pairs (0.01%)
        self.slippage_pct = 0.00005  # Very liquid market


class OptionsDataCosts(TradingCosts):
    """
    Options trading costs (per contract)
    """
    def __init__(self):
        super().__init__()
        self.commission = 0.65  # $0.65 per contract typical
        self.min_commission = 0.0
        self.spread_pct = 0.05  # 5% - much wider spreads
        self.slippage_pct = 0.02  # 2% - less liquid


def get_trading_costs(asset_type='stock', broker='default'):
    """
    Factory function to get appropriate trading costs
    
    Args:
        asset_type: 'stock', 'crypto', 'forex', 'options'
        broker: specific broker/exchange name
    
    Returns:
        TradingCosts object with realistic fee structure
    """
    if asset_type.lower() == 'crypto':
        return CryptoTradingCosts(broker)
    elif asset_type.lower() == 'stock':
        return StockTradingCosts(broker)
    elif asset_type.lower() == 'forex':
        return ForexTradingCosts()
    elif asset_type.lower() == 'options':
        return OptionsDataCosts()
    else:
        return StockTradingCosts()  # Default to stock


def print_cost_comparison():
    """Print a comparison of trading costs across different platforms"""
    
    print("\n" + "="*80)
    print("REALISTIC TRADING COST COMPARISON")
    print("="*80)
    
    # Stocks
    print("\n📈 STOCK TRADING COSTS:")
    print("-"*40)
    for broker in ['robinhood', 'interactive_brokers', 'td_ameritrade']:
        costs = StockTradingCosts(broker)
        print(f"\n{broker.replace('_', ' ').title()}:")
        print(f"  Commission: {costs.commission:.4%} (min ${costs.min_commission})")
        print(f"  Spread: {costs.spread_pct:.4%}")
        print(f"  Slippage: {costs.slippage_pct:.4%}")
        print(f"  Total cost per trade: ~{(costs.commission + costs.spread_pct + costs.slippage_pct):.4%}")
    
    # Crypto
    print("\n💰 CRYPTO TRADING COSTS:")
    print("-"*40)
    for exchange in ['binance', 'coinbase', 'default']:
        costs = CryptoTradingCosts(exchange)
        print(f"\n{exchange.title()}:")
        print(f"  Commission: {costs.commission:.4%}")
        print(f"  Spread: {costs.spread_pct:.4%}")
        print(f"  Slippage: {costs.slippage_pct:.4%}")
        print(f"  Total cost per trade: ~{(costs.commission + costs.spread_pct + costs.slippage_pct):.4%}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("-"*40)
    print("• Stock trading is essentially FREE on most modern brokers")
    print("• Crypto fees are 10-100x higher than stocks")
    print("• Slippage is minimal on liquid assets")
    print("• Spread is the main hidden cost for zero-commission brokers")
    print("="*80)


if __name__ == '__main__':
    print_cost_comparison()