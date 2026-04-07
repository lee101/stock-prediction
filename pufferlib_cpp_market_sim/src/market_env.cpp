#include "market_env.h"
#include <random>
#include <algorithm>

namespace market_sim {

MarketEnvironment::MarketEnvironment(const MarketConfig& config)
    : config_(config),
      device_(torch::Device(config_.device)),
      is_training_(true),
      max_episode_steps_(500),
      batch_size_(config_.NUM_PARALLEL_ENVS) {

    // Initialize logger
    logger_ = std::make_shared<PnLLogger>(config_.log_dir);

    // Initialize portfolio
    portfolio_ = std::make_shared<Portfolio>(config_, device_, batch_size_);

    // Initialize state tensors
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    current_indices_ = torch::zeros({batch_size_}, opts);
    current_symbols_ = torch::zeros({batch_size_}, opts);
    episode_steps_ = torch::zeros({batch_size_}, opts);
}

void MarketEnvironment::load_symbols(const std::vector<std::string>& symbols) {
    market_states_.clear();

    for (const auto& symbol : symbols) {
        auto market_state = std::make_shared<MarketState>(config_, device_);
        market_state->load_data(symbol);
        market_states_.push_back(market_state);
    }

    // Initialize random symbol assignment for each environment
    randomize_starting_positions(torch::ones({batch_size_}, device_));
}

EnvOutput MarketEnvironment::reset(const torch::Tensor& env_indices) {
    // env_indices: [num_envs_to_reset] - which environments to reset

    auto mask = torch::zeros({batch_size_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    for (int i = 0; i < env_indices.size(0); i++) {
        int idx = env_indices[i].item<int>();
        mask[idx] = 1.0f;
    }

    // Reset portfolio for these environments
    portfolio_->reset(mask);

    // Randomize starting positions
    randomize_starting_positions(mask);

    // Reset episode steps
    episode_steps_ = torch::where(
        mask.to(torch::kBool),
        torch::zeros({batch_size_}, episode_steps_.options()),
        episode_steps_
    );

    // Get initial observations
    EnvOutput output;
    std::vector<torch::Tensor> obs_list;

    for (int i = 0; i < batch_size_; i++) {
        int symbol_idx = current_symbols_[i].item<int>();
        auto market_obs = market_states_[symbol_idx]->get_observation(
            current_indices_.index({i}).unsqueeze(0)
        ).squeeze(0);  // [lookback, features]

        // Flatten market observation
        auto flat_market_obs = market_obs.flatten();  // [lookback * features]

        // Get portfolio state
        auto portfolio_state = portfolio_->get_state().index({i});  // [5]

        // Concatenate
        auto full_obs = torch::cat({flat_market_obs, portfolio_state}, /*dim=*/0);
        obs_list.push_back(full_obs);
    }

    output.observations = torch::stack(obs_list, /*dim=*/0);  // [batch_size, obs_dim]
    output.rewards = torch::zeros({batch_size_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    output.dones = torch::zeros({batch_size_}, torch::TensorOptions().dtype(torch::kBool).device(device_));
    output.truncated = torch::zeros({batch_size_}, torch::TensorOptions().dtype(torch::kBool).device(device_));
    output.prices = peek_next_prices();
    auto zero_float = torch::zeros({batch_size_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    output.fees_paid = zero_float.clone();
    output.leverage_costs = zero_float.clone();
    output.realized_pnl = zero_float.clone();
    output.days_held = zero_float.clone();

    return output;
}

EnvOutput MarketEnvironment::step(const torch::Tensor& actions) {
    // SCALAR mode: actions [batch_size]            (legacy, bit-identical path)
    // DPS    mode: actions [batch_size, 3] = (dir, size, limit_offset_bps_norm)
    //              dir   in [-1, 1]   (tanh)
    //              size  in [ 0, 1]   (sigmoid)
    //              off   in [-1, 1]   (tanh) -> scaled by max_limit_offset_bps
    //
    // In DPS mode we collapse the 3-d action down to a single scalar leverage
    // value (target = dir*size*max_leverage_dps) gated by a binary fill check
    // against the bar's [low, high] range:
    //   buy  fills iff bar low  <= limit price
    //   sell fills iff bar high >= limit price
    // Unfilled steps pass leverage = previous_leverage so the position is
    // unchanged (no trade -> no fee). Filled steps pass the new target.

    // Increment indices and episode steps
    current_indices_ = current_indices_ + 1;
    episode_steps_ = episode_steps_ + 1;

    // Get current prices for all environments
    std::vector<torch::Tensor> prices_list;
    for (int i = 0; i < batch_size_; i++) {
        int symbol_idx = current_symbols_[i].item<int>();
        auto prices = market_states_[symbol_idx]->get_current_prices(
            current_indices_.index({i}).unsqueeze(0)
        ).squeeze(0);  // [4]
        prices_list.push_back(prices);
    }
    auto all_prices = torch::stack(prices_list, /*dim=*/0);  // [batch_size, 4]

    // Translate DPS action -> scalar leverage with binary fill gating.
    torch::Tensor scalar_actions;
    if (config_.action_mode == ActionMode::DPS) {
        TORCH_CHECK(actions.dim() == 2 && actions.size(1) == 3,
                    "DPS action_mode expects [batch, 3] actions");
        // OHLC layout matches market_state convention used in execute_high_low_strategy:
        // index 0 = open, 1 = high, 2 = low, 3 = close.
        auto high  = all_prices.index({torch::indexing::Slice(), 1});
        auto low   = all_prices.index({torch::indexing::Slice(), 2});
        auto close = all_prices.index({torch::indexing::Slice(), 3});

        auto dir    = actions.index({torch::indexing::Slice(), 0});
        auto size   = actions.index({torch::indexing::Slice(), 1});
        auto offset = actions.index({torch::indexing::Slice(), 2});

        // Sanitize ranges (policy is responsible for tanh/sigmoid; clamp defensively).
        dir    = torch::clamp(dir,    -1.0f, 1.0f);
        size   = torch::clamp(size,    0.0f, 1.0f);
        offset = torch::clamp(offset, -1.0f, 1.0f);

        auto target_lev = dir * size * config_.max_leverage_dps;

        // Limit price = mid * (1 + sign(dir) * (fill_buffer_bps + offset*max_limit_offset_bps) * 1e-4)
        auto sign_dir = torch::sign(dir);
        auto bps = (config_.fill_buffer_bps + offset * config_.max_limit_offset_bps) * 1e-4f;
        auto limit_price = close * (1.0f + sign_dir * bps);

        auto buy_mask  = dir > 0;
        auto sell_mask = dir < 0;
        auto buy_fill  = buy_mask  & (low  <= limit_price);
        auto sell_fill = sell_mask & (high >= limit_price);
        auto fill_mask = buy_fill | sell_fill;  // dir==0 => no trade desired anyway

        // Unfilled -> hold previous leverage. Compute previous leverage from
        // current portfolio state (positions / equity), so a no-fill is a no-op.
        auto pf_state = portfolio_->get_state();  // [batch, 5], col 1 = position pct of equity
        auto prev_lev = pf_state.index({torch::indexing::Slice(), 1});

        scalar_actions = torch::where(fill_mask, target_lev, prev_lev);

        // Activate the wider DPS leverage cap on the portfolio for this call.
        portfolio_->set_leverage_cap_override(config_.max_leverage_dps);

        // Mark trades & PnL at the limit fill price on filled bars; on unfilled
        // bars there is no trade so the mark price falls back to close (no
        // observable fill exists). This is what gives DPS its trade-edge alpha
        // accounting in the C++ ground truth.
        auto exec_px = torch::where(fill_mask, limit_price, close);
        portfolio_->set_execution_price_override(exec_px);
    } else {
        scalar_actions = actions;
    }

    // Execute portfolio step (for now, use first market state's properties)
    // In a real implementation, you'd want to handle different assets separately
    auto step_result = portfolio_->step(scalar_actions, all_prices, *market_states_[0], false);

    // Restore SCALAR clamp & clear DPS execution-price override so any subsequent
    // caller of portfolio sees legacy behavior.
    if (config_.action_mode == ActionMode::DPS) {
        portfolio_->set_leverage_cap_override(0.0f);
        portfolio_->clear_execution_price_override();
    }
    auto rewards = step_result.rewards;

    // Check termination conditions
    auto dones = check_termination();

    // Get next observations
    std::vector<torch::Tensor> obs_list;
    for (int i = 0; i < batch_size_; i++) {
        int symbol_idx = current_symbols_[i].item<int>();
        auto market_obs = market_states_[symbol_idx]->get_observation(
            current_indices_.index({i}).unsqueeze(0)
        ).squeeze(0);

        auto flat_market_obs = market_obs.flatten();
        auto portfolio_state = portfolio_->get_state().index({i});
        auto full_obs = torch::cat({flat_market_obs, portfolio_state}, /*dim=*/0);
        obs_list.push_back(full_obs);
    }
    auto observations = torch::stack(obs_list, /*dim=*/0);

    // Log PnL periodically (every 10 steps)
    if (episode_steps_[0].item<int>() % 10 == 0) {
        auto metrics = portfolio_->get_metrics();
        std::vector<std::string> symbols;
        for (int i = 0; i < batch_size_; i++) {
            int symbol_idx = current_symbols_[i].item<int>();
            symbols.push_back(market_states_[symbol_idx]->get_observation(
                torch::zeros({1}, torch::kInt64).to(device_)
            ).size(0) > 0 ? "SYMBOL_" + std::to_string(symbol_idx) : "UNKNOWN");
        }

        auto env_ids = torch::arange(batch_size_, torch::kInt64).to(device_);
        auto positions = portfolio_->get_state().index({torch::indexing::Slice(), 1});

        logger_->log_step(
            symbols,
            env_ids,
            positions,
            metrics.total_pnl,
            metrics.realized_pnl,
            metrics.unrealized_pnl,
            metrics.trading_costs,
            metrics.leverage_costs,
            metrics.num_trades,
            is_training_
        );
    }

    // Handle episode termination
    auto done_mask = dones.to(torch::kFloat32);
    if (done_mask.sum().item<float>() > 0) {
        // Log episode completions
        auto metrics = portfolio_->get_metrics();
        auto done_indices = torch::nonzero(dones).squeeze(1);

        for (int i = 0; i < done_indices.size(0); i++) {
            int env_id = done_indices[i].item<int>();
            int symbol_idx = current_symbols_[env_id].item<int>();

            logger_->log_episode_end(
                env_id,
                "SYMBOL_" + std::to_string(symbol_idx),
                metrics.total_pnl[env_id].item<float>(),
                (metrics.total_pnl[env_id] / config_.INITIAL_CAPITAL * 100.0f).item<float>(),
                metrics.num_trades[env_id].item<int>(),
                metrics.sharpe_ratio[env_id].item<float>(),
                is_training_
            );
        }

        // Reset completed environments
        reset(done_indices);
    }

    EnvOutput output;
    output.observations = observations;
    output.rewards = rewards;
    output.dones = dones;
    output.truncated = torch::zeros_like(dones);
    output.prices = all_prices;
    output.fees_paid = step_result.fees_paid;
    output.leverage_costs = step_result.leverage_costs;
    output.realized_pnl = step_result.realized_pnl;
    output.days_held = step_result.days_held;

    return output;
}

torch::Tensor MarketEnvironment::peek_next_prices() const {
    auto price_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    auto prices = torch::zeros({batch_size_, 4}, price_opts);

    for (int i = 0; i < batch_size_; i++) {
        int symbol_idx = current_symbols_[i].item<int>();
        int next_idx = current_indices_[i].item<int>() + 1;
        int data_length = market_states_[symbol_idx]->get_data_length();
        if (next_idx >= data_length) {
            next_idx = data_length - 1;
        }
        auto idx_tensor = torch::tensor({next_idx}, idx_opts);
        prices[i] = market_states_[symbol_idx]->get_current_prices(idx_tensor).squeeze(0);
    }

    return prices;
}

int MarketEnvironment::get_observation_dim() const {
    // Observation = market_features + portfolio_features
    // market_features = lookback_window * num_features (OHLCV = 5)
    // portfolio_features = 5 (cash, position, pnl, trades, days_held)
    return config_.LOOKBACK_WINDOW * 5 + 5;
}

void MarketEnvironment::flush_logs() {
    logger_->flush();
    logger_->write_summary_stats(true);   // Training
    logger_->write_summary_stats(false);  // Testing
}

void MarketEnvironment::randomize_starting_positions(const torch::Tensor& env_indices) {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < batch_size_; i++) {
        if (env_indices[i].item<float>() > 0.5f) {
            // Random symbol
            std::uniform_int_distribution<> symbol_dist(0, market_states_.size() - 1);
            current_symbols_[i] = symbol_dist(gen);

            // Random starting position within first 80% of data
            int symbol_idx = current_symbols_[i].item<int>();
            int data_length = market_states_[symbol_idx]->get_data_length();
            int max_start = static_cast<int>(data_length * 0.8);
            std::uniform_int_distribution<> pos_dist(config_.LOOKBACK_WINDOW, max_start);
            current_indices_[i] = pos_dist(gen);
        }
    }
}

torch::Tensor MarketEnvironment::check_termination() {
    // Environments terminate when:
    // 1. Reached end of data
    // 2. Max episode steps reached
    // 3. Portfolio value < 10% of initial capital (bankrupt)

    auto dones = torch::zeros({batch_size_}, torch::TensorOptions().dtype(torch::kBool).device(device_));

    for (int i = 0; i < batch_size_; i++) {
        int symbol_idx = current_symbols_[i].item<int>();
        int current_idx = current_indices_[i].item<int>();
        int data_length = market_states_[symbol_idx]->get_data_length();

        // Check if reached end of data
        if (current_idx >= data_length - 1) {
            dones[i] = true;
        }

        // Check if max steps reached
        if (episode_steps_[i].item<int>() >= max_episode_steps_) {
            dones[i] = true;
        }
    }

    // Check portfolio value
    auto portfolio_state = portfolio_->get_state();
    auto equity_ratio = portfolio_state.index({torch::indexing::Slice(), 2}) + 1.0f;  // pnl% + 1
    auto bankrupt = equity_ratio < 0.1f;

    dones = dones | bankrupt;

    return dones;
}

} // namespace market_sim
