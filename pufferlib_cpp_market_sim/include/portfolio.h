#pragma once

#include <torch/torch.h>
#include "market_config.h"
#include "market_state.h"

namespace market_sim {

struct PortfolioMetrics {
    torch::Tensor total_pnl;           // [batch_size] - cumulative PnL
    torch::Tensor realized_pnl;        // [batch_size] - realized gains/losses
    torch::Tensor unrealized_pnl;      // [batch_size] - current position value
    torch::Tensor trading_costs;       // [batch_size] - cumulative fees paid
    torch::Tensor leverage_costs;      // [batch_size] - cumulative leverage interest
    torch::Tensor num_trades;          // [batch_size] - total number of trades
    torch::Tensor sharpe_ratio;        // [batch_size] - rolling sharpe ratio
};

struct PortfolioStepResult {
    torch::Tensor rewards;             // [batch_size]
    torch::Tensor fees_paid;           // [batch_size]
    torch::Tensor leverage_costs;      // [batch_size]
    torch::Tensor realized_pnl;        // [batch_size]
    torch::Tensor unrealized_pnl;      // [batch_size]
    torch::Tensor equity;              // [batch_size]
    torch::Tensor days_held;           // [batch_size]
};

class Portfolio {
public:
    Portfolio(const MarketConfig& config, torch::Device device, int batch_size);

    // Execute trading action
    // action: [batch_size] - continuous action in range [-max_leverage, max_leverage]
    //         Positive = long position, Negative = short position
    //         Magnitude = leverage multiplier
    // Returns: per-step metrics
    PortfolioStepResult step(
        const torch::Tensor& action,
        const torch::Tensor& prices,  // [batch_size, 4] - OHLC
        const MarketState& market_state,
        bool use_high_low_strategy = false
    );

    // Reset portfolio to initial state
    void reset(const torch::Tensor& mask);  // mask: [batch_size] - which envs to reset

    // Get current portfolio state for observation
    // Returns: [batch_size, num_portfolio_features]
    torch::Tensor get_state();

    // Get metrics for logging
    PortfolioMetrics get_metrics() const;

    // Calculate leverage cost for positions > 1.0x
    torch::Tensor calculate_leverage_cost(const torch::Tensor& positions);

private:
    MarketConfig config_;
    torch::Device device_;
    int batch_size_;

    // Portfolio state (all tensors are [batch_size])
    torch::Tensor cash_;               // Available cash
    torch::Tensor positions_;          // Current position size (in dollars)
    torch::Tensor entry_prices_;       // Entry price for current position
    torch::Tensor total_pnl_;          // Cumulative PnL
    torch::Tensor realized_pnl_;       // Realized PnL from closed trades
    torch::Tensor trading_costs_;      // Cumulative trading fees
    torch::Tensor leverage_costs_;     // Cumulative leverage costs
    torch::Tensor num_trades_;         // Number of trades executed
    torch::Tensor daily_returns_;      // [batch_size, 252] - rolling returns for Sharpe
    torch::Tensor days_held_;          // Days current position has been held

    // Calculate trading fees
    torch::Tensor calculate_trading_fee(
        const torch::Tensor& trade_amount,
        float fee_rate
    );

    // Calculate reward based on PnL, fees, and risk metrics
    torch::Tensor calculate_reward(
        const torch::Tensor& pnl_change,
        const torch::Tensor& fees_paid,
        const torch::Tensor& leverage_cost
    );

    // Execute high/low strategy (buy at low, sell at high)
    torch::Tensor execute_high_low_strategy(
        const torch::Tensor& action,
        const torch::Tensor& prices
    );
};

} // namespace market_sim
