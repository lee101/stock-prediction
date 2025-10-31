#include "portfolio.h"
#include <cmath>

namespace market_sim {

Portfolio::Portfolio(const MarketConfig& config, torch::Device device, int batch_size)
    : config_(config), device_(device), batch_size_(batch_size) {

    // Initialize portfolio state tensors on GPU
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    cash_ = torch::full({batch_size}, config_.INITIAL_CAPITAL, opts);
    positions_ = torch::zeros({batch_size}, opts);
    entry_prices_ = torch::zeros({batch_size}, opts);
    total_pnl_ = torch::zeros({batch_size}, opts);
    realized_pnl_ = torch::zeros({batch_size}, opts);
    trading_costs_ = torch::zeros({batch_size}, opts);
    leverage_costs_ = torch::zeros({batch_size}, opts);
    num_trades_ = torch::zeros({batch_size}, opts);
    daily_returns_ = torch::zeros({batch_size, 252}, opts);
    days_held_ = torch::zeros({batch_size}, opts);
}

torch::Tensor Portfolio::step(
    const torch::Tensor& action,
    const torch::Tensor& prices,
    const MarketState& market_state,
    bool use_high_low_strategy
) {
    // action: [batch_size] - continuous action in [-max_leverage, max_leverage]
    // prices: [batch_size, 4] - OHLC

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    // Get current close price
    auto current_price = prices.index({torch::indexing::Slice(), 3});  // close

    // If using high/low strategy, execute based on high/low prices
    torch::Tensor execution_price;
    if (use_high_low_strategy) {
        execution_price = execute_high_low_strategy(action, prices);
    } else {
        execution_price = current_price;
    }

    // Get trading fee rate
    float fee_rate = market_state.get_trading_fee();
    bool is_crypto = market_state.is_crypto();

    // Determine target position size based on action
    // action represents leverage multiplier (-max_leverage to +max_leverage)
    auto clamped_action = torch::clamp(action, -config_.MAX_LEVERAGE, config_.MAX_LEVERAGE);

    // For crypto, no short selling (negative positions)
    if (is_crypto) {
        clamped_action = torch::clamp(clamped_action, 0.0f, config_.MAX_LEVERAGE);
    }

    // Target position in dollars
    auto equity = cash_ + positions_;  // Current portfolio value
    auto target_position = equity * clamped_action;

    // Calculate trade amount (positive = buy, negative = sell)
    auto trade_amount = target_position - positions_;

    // Calculate trading fees
    auto fees = calculate_trading_fee(torch::abs(trade_amount), fee_rate);

    // Update positions
    auto position_change_mask = torch::abs(trade_amount) > 1e-6;  // Only if meaningful change

    // Update entry prices for new/increased positions
    auto increasing_position = (trade_amount * positions_ >= 0) & (torch::abs(trade_amount) > 1e-6);
    entry_prices_ = torch::where(
        increasing_position,
        (entry_prices_ * torch::abs(positions_) + execution_price * torch::abs(trade_amount)) /
        (torch::abs(positions_) + torch::abs(trade_amount) + 1e-8),
        entry_prices_
    );

    // For position reversals or new positions, use current price as entry
    auto new_position = (positions_ == 0) | (trade_amount * positions_ < 0);
    entry_prices_ = torch::where(new_position & position_change_mask, execution_price, entry_prices_);

    // Execute trade
    positions_ = positions_ + trade_amount;
    cash_ = cash_ - trade_amount - fees;
    trading_costs_ = trading_costs_ + fees;
    num_trades_ = torch::where(position_change_mask, num_trades_ + 1, num_trades_);

    // Calculate unrealized PnL
    auto unrealized_pnl = positions_ * (current_price - entry_prices_) / (entry_prices_ + 1e-8);

    // Calculate leverage cost for positions > 1.0x
    auto lev_cost = calculate_leverage_cost(positions_);
    leverage_costs_ = leverage_costs_ + lev_cost;
    cash_ = cash_ - lev_cost;

    // Calculate total equity
    auto total_equity = cash_ + positions_;

    // Calculate daily return
    auto daily_return = (total_equity - config_.INITIAL_CAPITAL) / config_.INITIAL_CAPITAL;

    // Update rolling returns (shift and append)
    daily_returns_ = torch::cat({
        daily_returns_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}),
        daily_return.unsqueeze(1)
    }, /*dim=*/1);

    // Calculate reward
    auto pnl_change = (total_equity - config_.INITIAL_CAPITAL) - total_pnl_;
    total_pnl_ = total_equity - config_.INITIAL_CAPITAL;

    auto reward = calculate_reward(pnl_change, fees, lev_cost);

    days_held_ = days_held_ + 1;

    return reward;
}

void Portfolio::reset(const torch::Tensor& mask) {
    // mask: [batch_size] - which envs to reset (1 = reset, 0 = keep)
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    cash_ = torch::where(mask.to(torch::kBool),
                        torch::full({batch_size_}, config_.INITIAL_CAPITAL, opts),
                        cash_);
    positions_ = torch::where(mask.to(torch::kBool), torch::zeros({batch_size_}, opts), positions_);
    entry_prices_ = torch::where(mask.to(torch::kBool), torch::zeros({batch_size_}, opts), entry_prices_);
    total_pnl_ = torch::where(mask.to(torch::kBool), torch::zeros({batch_size_}, opts), total_pnl_);
    realized_pnl_ = torch::where(mask.to(torch::kBool), torch::zeros({batch_size_}, opts), realized_pnl_);
    trading_costs_ = torch::where(mask.to(torch::kBool), torch::zeros({batch_size_}, opts), trading_costs_);
    leverage_costs_ = torch::where(mask.to(torch::kBool), torch::zeros({batch_size_}, opts), leverage_costs_);
    num_trades_ = torch::where(mask.to(torch::kBool), torch::zeros({batch_size_}, opts), num_trades_);
    days_held_ = torch::where(mask.to(torch::kBool), torch::zeros({batch_size_}, opts), days_held_);
}

torch::Tensor Portfolio::get_state() {
    // Returns: [batch_size, num_portfolio_features]
    // Features: cash, position, entry_price, total_pnl, num_trades, days_held

    auto equity = cash_ + positions_;
    auto position_pct = positions_ / (equity + 1e-8);
    auto pnl_pct = total_pnl_ / config_.INITIAL_CAPITAL;

    return torch::stack({
        cash_ / config_.INITIAL_CAPITAL,      // Normalized cash
        position_pct,                          // Position as % of equity
        pnl_pct,                              // Total return %
        num_trades_ / 100.0f,                 // Normalized trade count
        days_held_ / 30.0f                    // Days held normalized
    }, /*dim=*/1);  // [batch_size, 5]
}

PortfolioMetrics Portfolio::get_metrics() const {
    PortfolioMetrics metrics;

    auto equity = cash_ + positions_;
    auto unrealized_pnl = positions_;  // Simplified for now

    metrics.total_pnl = total_pnl_;
    metrics.realized_pnl = realized_pnl_;
    metrics.unrealized_pnl = unrealized_pnl;
    metrics.trading_costs = trading_costs_;
    metrics.leverage_costs = leverage_costs_;
    metrics.num_trades = num_trades_;

    // Calculate Sharpe ratio (simplified)
    auto returns_std = daily_returns_.std(/*dim=*/1);
    auto returns_mean = daily_returns_.mean(/*dim=*/1);
    metrics.sharpe_ratio = returns_mean / (returns_std + 1e-8) * std::sqrt(252.0);  // Annualized

    return metrics;
}

torch::Tensor Portfolio::calculate_leverage_cost(const torch::Tensor& positions) {
    // Calculate interest cost for positions exceeding 1.0x leverage

    auto equity = cash_ + positions;
    auto leverage_ratio = torch::abs(positions) / (equity + 1e-8);

    // Only charge interest on leverage above 1.0x
    auto excess_leverage = torch::clamp(leverage_ratio - 1.0f, 0.0f, 10.0f);

    // Daily leverage cost = borrowed_amount * daily_rate
    auto borrowed_amount = excess_leverage * equity;
    auto daily_cost = borrowed_amount * config_.DAILY_LEVERAGE_COST;

    return daily_cost;
}

torch::Tensor Portfolio::calculate_trading_fee(
    const torch::Tensor& trade_amount,
    float fee_rate
) {
    return trade_amount * fee_rate;
}

torch::Tensor Portfolio::calculate_reward(
    const torch::Tensor& pnl_change,
    const torch::Tensor& fees_paid,
    const torch::Tensor& leverage_cost
) {
    // Reward = PnL change - costs
    // Normalized by initial capital

    auto net_pnl = pnl_change - fees_paid - leverage_cost;
    auto reward = net_pnl / config_.INITIAL_CAPITAL;

    // Add small penalty for excessive trading
    auto trade_penalty = fees_paid / config_.INITIAL_CAPITAL * 0.5f;

    return reward - trade_penalty;
}

torch::Tensor Portfolio::execute_high_low_strategy(
    const torch::Tensor& action,
    const torch::Tensor& prices
) {
    // Buy at low, sell at high (maxdiff-style strategy)
    // prices: [batch_size, 4] - OHLC

    auto low = prices.index({torch::indexing::Slice(), 2});
    auto high = prices.index({torch::indexing::Slice(), 1});
    auto close = prices.index({torch::indexing::Slice(), 3});

    // If action > 0 (buy), use low price
    // If action < 0 (sell), use high price
    // If action == 0, use close price

    auto buy_mask = action > 0;
    auto sell_mask = action < 0;

    auto execution_price = torch::where(buy_mask, low,
                          torch::where(sell_mask, high, close));

    return execution_price;
}

} // namespace market_sim
