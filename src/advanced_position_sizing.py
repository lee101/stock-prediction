"""
Advanced position sizing strategies for comprehensive backtesting.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

Returns = Union[pd.Series, pd.DataFrame]


def kelly_criterion_sizing(predicted_returns: Returns, win_rate: float = 0.55, avg_win: float = 0.02, avg_loss: float = 0.01) -> Returns:
    """
    Kelly Criterion position sizing based on win rate and average win/loss.
    
    Kelly % = (bp - q) / b
    where:
    - b = odds (avg_win / avg_loss)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """
    b = avg_win / avg_loss if avg_loss > 0 else 1
    p = win_rate
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    kelly_fraction = max(0, min(kelly_fraction, 1))  # Clamp between 0 and 1
    
    # Apply Kelly fraction to predicted returns (direction matters)
    if isinstance(predicted_returns, pd.DataFrame):
        sizes = predicted_returns.copy()
        sizes[sizes > 0] = kelly_fraction
        sizes[sizes < 0] = -kelly_fraction
        return sizes
    else:
        sizes = predicted_returns.copy()
        sizes[sizes > 0] = kelly_fraction
        sizes[sizes < 0] = -kelly_fraction
        return sizes


def momentum_sizing(predicted_returns: Returns, window: int = 20, momentum_factor: float = 2.0) -> Returns:
    """
    Size positions based on momentum - increase size when predictions are trending in same direction.
    """
    if isinstance(predicted_returns, pd.DataFrame):
        momentum_scores = predicted_returns.rolling(window=window).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        # Scale momentum: 0.5 = neutral, 1.0 = all positive, 0.0 = all negative
        momentum_multiplier = ((momentum_scores - 0.5) * momentum_factor + 1).clip(0.1, 3.0)
        return predicted_returns * momentum_multiplier
    else:
        momentum_score = predicted_returns.rolling(window=window).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        momentum_multiplier = ((momentum_score - 0.5) * momentum_factor + 1).clip(0.1, 3.0)
        return predicted_returns * momentum_multiplier


def regime_aware_sizing(predicted_returns: Returns, volatility_window: int = 30, vol_threshold: float = 0.02) -> Returns:
    """
    Adjust position sizes based on market regime (high vs low volatility).
    """
    if isinstance(predicted_returns, pd.DataFrame):
        # Calculate rolling volatility for each asset
        volatility = predicted_returns.rolling(window=volatility_window).std()
        
        # Create regime multiplier (reduce size in high vol regime)
        regime_multiplier = (vol_threshold / volatility).clip(0.2, 2.0)
        return predicted_returns * regime_multiplier
    else:
        volatility = predicted_returns.rolling(window=volatility_window).std()
        regime_multiplier = (vol_threshold / volatility).clip(0.2, 2.0)
        return predicted_returns * regime_multiplier


def correlation_adjusted_sizing(predicted_returns: pd.DataFrame, lookback: int = 60, max_correlation: float = 0.7) -> pd.DataFrame:
    """
    Adjust position sizes based on correlation between assets to avoid over-concentration.
    """
    if not isinstance(predicted_returns, pd.DataFrame):
        raise ValueError("correlation_adjusted_sizing requires DataFrame input")
    
    sizes = predicted_returns.copy()
    
    for i in range(lookback, len(predicted_returns)):
        # Calculate correlation matrix for the lookback period
        returns_window = predicted_returns.iloc[i-lookback:i]
        corr_matrix = returns_window.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2 and corr_matrix.loc[col1, col2] > max_correlation:
                    high_corr_pairs.append((col1, col2))
        
        # Reduce position sizes for highly correlated assets
        row_sizes = sizes.iloc[i].copy()
        for col1, col2 in high_corr_pairs:
            # Reduce the size of the smaller position
            if abs(row_sizes[col1]) < abs(row_sizes[col2]):
                row_sizes[col1] *= 0.5
            else:
                row_sizes[col2] *= 0.5
        
        sizes.iloc[i] = row_sizes
    
    return sizes


def adaptive_k_sizing(predicted_returns: Returns, base_k: float = 3.0, adaptation_window: int = 30) -> Returns:
    """
    Adaptive K-divisor that adjusts based on recent performance.
    """
    if isinstance(predicted_returns, pd.DataFrame):
        # Calculate recent volatility to adjust K
        recent_vol = predicted_returns.rolling(window=adaptation_window).std()
        avg_vol = recent_vol.mean()
        
        # Adjust K based on volatility (higher vol -> higher K -> smaller positions)
        k_adjustment = recent_vol / avg_vol
        adaptive_k = base_k * k_adjustment
        
        return predicted_returns / adaptive_k
    else:
        recent_vol = predicted_returns.rolling(window=adaptation_window).std()
        avg_vol = recent_vol.mean()
        
        k_adjustment = recent_vol / avg_vol
        adaptive_k = base_k * k_adjustment
        
        return predicted_returns / adaptive_k


def confidence_weighted_sizing(predicted_returns: Returns, confidence_scores: Optional[Returns] = None) -> Returns:
    """
    Weight position sizes by prediction confidence.
    If no confidence scores provided, use absolute magnitude of predictions as proxy.
    """
    if confidence_scores is None:
        # Use absolute magnitude as confidence proxy
        confidence_scores = abs(predicted_returns)
    
    # Normalize confidence scores
    if isinstance(confidence_scores, pd.DataFrame):
        confidence_normalized = confidence_scores.div(confidence_scores.max(axis=1), axis=0).fillna(0)
    else:
        confidence_normalized = confidence_scores / confidence_scores.max()
    
    return predicted_returns * confidence_normalized


def sector_balanced_sizing(predicted_returns: pd.DataFrame, sector_mapping: Dict[str, str], max_sector_weight: float = 0.4) -> pd.DataFrame:
    """
    Balance position sizes across sectors to avoid concentration risk.
    """
    if not isinstance(predicted_returns, pd.DataFrame):
        raise ValueError("sector_balanced_sizing requires DataFrame input")
    
    sizes = predicted_returns.copy()
    
    for i in range(len(sizes)):
        row_sizes = sizes.iloc[i].copy()
        
        # Group by sector and calculate total exposure
        sector_exposure = {}
        for asset, sector in sector_mapping.items():
            if asset in row_sizes.index:
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += abs(row_sizes[asset])
        
        # Calculate total exposure
        total_exposure = sum(sector_exposure.values())
        
        # Adjust sizes if any sector is over-weighted
        for sector, exposure in sector_exposure.items():
            if exposure > max_sector_weight * total_exposure:
                # Scale down all assets in this sector
                sector_assets = [asset for asset, s in sector_mapping.items() if s == sector and asset in row_sizes.index]
                scale_factor = (max_sector_weight * total_exposure) / exposure
                for asset in sector_assets:
                    row_sizes[asset] *= scale_factor
        
        sizes.iloc[i] = row_sizes
    
    return sizes


def risk_parity_sizing(predicted_returns: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Risk parity position sizing - equal risk contribution from each asset.
    """
    if not isinstance(predicted_returns, pd.DataFrame):
        raise ValueError("risk_parity_sizing requires DataFrame input")
    
    sizes = predicted_returns.copy()
    
    for i in range(lookback, len(predicted_returns)):
        # Calculate covariance matrix for the lookback period
        returns_window = predicted_returns.iloc[i-lookback:i]
        cov_matrix = returns_window.cov()
        
        # Calculate inverse volatility weights
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol_weights = 1 / volatilities
        inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
        
        # Apply weights to predicted returns (maintaining direction)
        row_predictions = predicted_returns.iloc[i]
        row_sizes = row_predictions.copy()
        
        for j, asset in enumerate(row_sizes.index):
            if row_predictions[asset] != 0:
                row_sizes[asset] = np.sign(row_predictions[asset]) * inv_vol_weights[j]
        
        sizes.iloc[i] = row_sizes
    
    return sizes


def machine_learning_sizing(predicted_returns: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    Use simple ML approach to determine optimal position sizes based on historical performance.
    """
    if not isinstance(predicted_returns, pd.DataFrame):
        raise ValueError("machine_learning_sizing requires DataFrame input")
    
    sizes = predicted_returns.copy()
    
    # Simple approach: use correlation between prediction magnitude and next period return
    for i in range(lookback, len(predicted_returns)):
        # Historical data
        hist_predictions = predicted_returns.iloc[i-lookback:i]
        hist_returns = predicted_returns.iloc[i-lookback+1:i+1]  # Next period returns
        
        # Calculate correlation between prediction magnitude and actual returns
        correlation_scores = {}
        for asset in hist_predictions.columns:
            if asset in hist_returns.columns:
                corr = np.corrcoef(abs(hist_predictions[asset]), abs(hist_returns[asset]))[0, 1]
                correlation_scores[asset] = corr if not np.isnan(corr) else 0
        
        # Use correlation as confidence multiplier
        row_predictions = predicted_returns.iloc[i]
        row_sizes = row_predictions.copy()
        
        for asset in row_sizes.index:
            if asset in correlation_scores:
                confidence = max(0, correlation_scores[asset])  # Only positive correlations
                row_sizes[asset] *= confidence
        
        sizes.iloc[i] = row_sizes
    
    return sizes


def multi_timeframe_sizing(predicted_returns: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """
    Combine short-term and long-term predictions for position sizing.
    """
    if not isinstance(predicted_returns, pd.DataFrame):
        raise ValueError("multi_timeframe_sizing requires DataFrame input")
    
    # Calculate short-term and long-term moving averages of predictions
    short_ma = predicted_returns.rolling(window=short_window).mean()
    long_ma = predicted_returns.rolling(window=long_window).mean()
    
    # Combine signals: stronger when both timeframes agree
    combined_signal = predicted_returns.copy()
    
    # Boost signal when short and long term agree
    agreement_boost = np.sign(short_ma) * np.sign(long_ma)  # 1 when same direction, -1 when opposite
    combined_signal = combined_signal * (1 + 0.5 * agreement_boost)
    
    return combined_signal


def get_all_advanced_strategies() -> Dict[str, Callable[[Returns], Returns]]:
    """
    Get dictionary of all advanced position sizing strategies.
    """
    return {
        'kelly_criterion': lambda p: kelly_criterion_sizing(p),
        'momentum_20d': lambda p: momentum_sizing(p, window=20, momentum_factor=2.0),
        'momentum_10d': lambda p: momentum_sizing(p, window=10, momentum_factor=1.5),
        'regime_aware': lambda p: regime_aware_sizing(p),
        'adaptive_k3': lambda p: adaptive_k_sizing(p, base_k=3.0),
        'adaptive_k5': lambda p: adaptive_k_sizing(p, base_k=5.0),
        'confidence_weighted': lambda p: confidence_weighted_sizing(p),
        'multi_timeframe': lambda p: multi_timeframe_sizing(p) if isinstance(p, pd.DataFrame) else p,
    }


def get_dataframe_only_strategies() -> Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    """
    Get strategies that only work with DataFrame inputs (multi-asset).
    """
    return {
        'risk_parity': lambda p: risk_parity_sizing(p),
        'ml_sizing': lambda p: machine_learning_sizing(p),
        'correlation_adjusted': lambda p: correlation_adjusted_sizing(p),
    }


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    n_assets = 5
    
    # Generate correlated returns
    returns = np.random.randn(100, n_assets) * 0.02
    asset_columns = pd.Index([f'Asset_{i}' for i in range(n_assets)])
    returns = pd.DataFrame(returns, index=dates, columns=asset_columns)
    
    # Generate predictions (slightly correlated with future returns)
    predictions = returns.shift(1).fillna(0) + np.random.randn(100, n_assets) * 0.01
    
    # Test different strategies
    strategies = get_all_advanced_strategies()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, strategy_func) in enumerate(list(strategies.items())[:4]):
        try:
            sizes = strategy_func(predictions)
            cumulative_pnl = (sizes * returns).sum(axis=1).cumsum()
            axes[i].plot(cumulative_pnl)
            axes[i].set_title(f'{name} Strategy')
            axes[i].grid(True)
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    plt.tight_layout()
    plt.savefig('advanced_strategies_demo.png')
    plt.show()
    
    print("Advanced position sizing strategies demo completed!")
