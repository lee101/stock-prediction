#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int8_t SIDE_BUY = 0;
constexpr int8_t SIDE_SELL = 1;
constexpr int8_t SIDE_SHORT_SELL = 2;
constexpr int8_t SIDE_BUY_COVER = 3;

constexpr int8_t REASON_ENTRY = 0;
constexpr int8_t REASON_TARGET = 1;
constexpr int8_t REASON_TIMEOUT = 2;
constexpr int8_t REASON_EOD = 3;

struct PositionState {
    bool active = false;
    double qty = 0.0;
    double entry_price = 0.0;
    double sell_target = 0.0;
    int market_hours_held = 0;
    int8_t direction = 1;  // +1 long, -1 short
};

struct Candidate {
    int32_t sym = -1;
    int8_t direction = 1;
    double edge = 0.0;
    double entry_price = 0.0;
    double exit_price = 0.0;
    double signal_amount = 0.0;
    double intensity_fraction = 0.0;
    double required_move_frac = 0.0;
};

struct TradeEvent {
    int32_t t_idx = 0;
    int32_t sym_idx = 0;
    int8_t side = SIDE_BUY;
    double price = 0.0;
    double qty = 0.0;
    double cash_after = 0.0;
    double inventory_after = 0.0;
    int8_t reason = REASON_ENTRY;
};

inline bool finite_positive(double v) {
    return std::isfinite(v) && v > 0.0;
}

double directional_entry_amount(
    bool is_short,
    double buy_amount,
    double sell_amount,
    double trade_amount) {
    const double primary = is_short ? sell_amount : buy_amount;
    const double secondary = trade_amount;
    const double tertiary = is_short ? buy_amount : sell_amount;

    auto pick = [](double v) -> bool {
        return std::isfinite(v) && v != 0.0;
    };

    double value = 0.0;
    if (pick(primary)) {
        value = primary;
    } else if (pick(secondary)) {
        value = secondary;
    } else if (pick(tertiary)) {
        value = tertiary;
    }
    if (!std::isfinite(value) || value < 0.0) {
        return 0.0;
    }
    return value;
}

double calibrated_intensity(
    double signal_amount,
    double trade_amount_scale,
    double intensity_power,
    double min_intensity_fraction,
    double side_multiplier) {
    if (trade_amount_scale <= 0.0) {
        return 0.0;
    }
    double raw = signal_amount / trade_amount_scale;
    if (!std::isfinite(raw)) {
        raw = 0.0;
    }
    raw = std::min(std::max(raw, 0.0), 1.0);

    double adjusted = raw;
    if (intensity_power > 0.0 && intensity_power != 1.0) {
        adjusted = std::pow(raw, intensity_power);
    }
    adjusted *= std::max(side_multiplier, 0.0);
    if (raw > 0.0) {
        adjusted = std::max(adjusted, std::max(min_intensity_fraction, 0.0));
    }
    adjusted = std::min(std::max(adjusted, 0.0), 1.0);
    return adjusted;
}

double edge_score(
    double pred_high,
    double pred_low,
    double pred_close,
    double entry_price,
    bool is_long,
    bool use_close_edge,
    double fee_rate) {
    if (!finite_positive(entry_price)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double edge = 0.0;
    if (is_long) {
        const double target = use_close_edge ? pred_close : pred_high;
        edge = (target - entry_price) / entry_price - fee_rate;
    } else {
        const double target = use_close_edge ? pred_close : pred_low;
        edge = (entry_price - target) / entry_price - fee_rate;
    }
    return std::isfinite(edge) ? edge : std::numeric_limits<double>::quiet_NaN();
}

double required_move_to_fill(double open_px, double entry_price, bool is_long, double bar_margin) {
    if (!finite_positive(open_px) || !finite_positive(entry_price)) {
        return std::numeric_limits<double>::infinity();
    }
    if (is_long) {
        const double trigger = entry_price * (1.0 - bar_margin);
        return std::max(0.0, (open_px - trigger) / open_px);
    }
    const double trigger = entry_price * (1.0 + bar_margin);
    return std::max(0.0, (trigger - open_px) / open_px);
}

}  // namespace

py::dict simulate_portfolio_dense(
    py::array_t<double, py::array::c_style | py::array::forcecast> open_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> high_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> low_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> close_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> buy_price_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> sell_price_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> buy_amount_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> sell_amount_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> trade_amount_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> pred_high_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> pred_low_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> pred_close_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> present_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> market_open_now_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> market_open_next_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> is_crypto_arr,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> direction_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> fee_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> timestamps_ns_arr,
    double initial_cash,
    int max_positions,
    double min_edge,
    const std::string& edge_mode,
    int max_hold_hours,
    bool enforce_market_hours,
    bool close_at_eod,
    double trade_amount_scale,
    double entry_intensity_power,
    double entry_min_intensity_fraction,
    double long_intensity_multiplier,
    double short_intensity_multiplier,
    double min_buy_amount,
    double max_leverage,
    bool market_order_entry,
    double bar_margin,
    const std::string& entry_selection_mode,
    double force_close_slippage,
    bool int_qty,
    double margin_annual_rate) {
    if (open_arr.ndim() != 2) {
        throw std::invalid_argument("open array must be rank-2");
    }
    const ssize_t t_count = open_arr.shape(0);
    const ssize_t s_count = open_arr.shape(1);
    auto expect_shape_2d = [t_count, s_count](const py::buffer_info& info, const char* name) {
        if (info.ndim != 2 || info.shape[0] != t_count || info.shape[1] != s_count) {
            throw std::invalid_argument(std::string(name) + " shape mismatch");
        }
    };

    expect_shape_2d(high_arr.request(), "high");
    expect_shape_2d(low_arr.request(), "low");
    expect_shape_2d(close_arr.request(), "close");
    expect_shape_2d(buy_price_arr.request(), "buy_price");
    expect_shape_2d(sell_price_arr.request(), "sell_price");
    expect_shape_2d(buy_amount_arr.request(), "buy_amount");
    expect_shape_2d(sell_amount_arr.request(), "sell_amount");
    expect_shape_2d(trade_amount_arr.request(), "trade_amount");
    expect_shape_2d(pred_high_arr.request(), "pred_high");
    expect_shape_2d(pred_low_arr.request(), "pred_low");
    expect_shape_2d(pred_close_arr.request(), "pred_close");
    expect_shape_2d(present_arr.request(), "present");
    expect_shape_2d(market_open_now_arr.request(), "market_open_now");
    expect_shape_2d(market_open_next_arr.request(), "market_open_next");
    if (is_crypto_arr.ndim() != 1 || is_crypto_arr.shape(0) != s_count) {
        throw std::invalid_argument("is_crypto shape mismatch");
    }
    if (direction_arr.ndim() != 1 || direction_arr.shape(0) != s_count) {
        throw std::invalid_argument("direction shape mismatch");
    }
    if (fee_arr.ndim() != 1 || fee_arr.shape(0) != s_count) {
        throw std::invalid_argument("fee shape mismatch");
    }
    if (timestamps_ns_arr.ndim() != 1 || timestamps_ns_arr.shape(0) != t_count) {
        throw std::invalid_argument("timestamps_ns shape mismatch");
    }

    auto open = open_arr.unchecked<2>();
    auto high = high_arr.unchecked<2>();
    auto low = low_arr.unchecked<2>();
    auto close = close_arr.unchecked<2>();
    auto buy_price = buy_price_arr.unchecked<2>();
    auto sell_price = sell_price_arr.unchecked<2>();
    auto buy_amount = buy_amount_arr.unchecked<2>();
    auto sell_amount = sell_amount_arr.unchecked<2>();
    auto trade_amount = trade_amount_arr.unchecked<2>();
    auto pred_high = pred_high_arr.unchecked<2>();
    auto pred_low = pred_low_arr.unchecked<2>();
    auto pred_close = pred_close_arr.unchecked<2>();
    auto present = present_arr.unchecked<2>();
    auto market_open_now = market_open_now_arr.unchecked<2>();
    auto market_open_next = market_open_next_arr.unchecked<2>();
    auto is_crypto = is_crypto_arr.unchecked<1>();
    auto direction = direction_arr.unchecked<1>();
    auto fee = fee_arr.unchecked<1>();
    auto timestamps_ns = timestamps_ns_arr.unchecked<1>();

    std::vector<PositionState> positions(static_cast<size_t>(s_count));
    std::vector<int32_t> position_order;
    std::vector<double> last_close(static_cast<size_t>(s_count), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> equity_values(static_cast<size_t>(t_count), initial_cash);
    std::vector<TradeEvent> trades;
    trades.reserve(static_cast<size_t>(t_count) * 2);

    auto mark_to_market = [&]() -> double {
        double total = 0.0;
        for (ssize_t s = 0; s < s_count; ++s) {
            const PositionState& pos = positions[static_cast<size_t>(s)];
            if (!pos.active || pos.qty <= 0.0) {
                continue;
            }
            double px = last_close[static_cast<size_t>(s)];
            if (!std::isfinite(px)) {
                px = pos.entry_price;
            }
            if (pos.direction < 0) {
                total += pos.qty * (2.0 * pos.entry_price - px);
            } else {
                total += pos.qty * px;
            }
        }
        return total;
    };

    auto push_trade = [&](int32_t t_idx,
                          int32_t sym_idx,
                          int8_t side,
                          double px,
                          double qty,
                          double cash_after,
                          double inventory_after,
                          int8_t reason) {
        TradeEvent ev;
        ev.t_idx = t_idx;
        ev.sym_idx = sym_idx;
        ev.side = side;
        ev.price = px;
        ev.qty = qty;
        ev.cash_after = cash_after;
        ev.inventory_after = inventory_after;
        ev.reason = reason;
        trades.push_back(ev);
    };

    const bool use_close_edge = !(edge_mode == "high_low" || edge_mode == "high");
    const bool first_trigger_mode = (entry_selection_mode == "first_trigger");
    double cash = initial_cash;

    auto remove_closed_positions = [&](const std::vector<int32_t>& closed_syms) {
        if (closed_syms.empty()) {
            return;
        }
        std::vector<uint8_t> closed_mask(static_cast<size_t>(s_count), 0);
        for (int32_t s : closed_syms) {
            if (s >= 0 && s < s_count) {
                closed_mask[static_cast<size_t>(s)] = 1;
                positions[static_cast<size_t>(s)] = PositionState{};
            }
        }
        std::vector<int32_t> next_order;
        next_order.reserve(position_order.size());
        for (int32_t s : position_order) {
            if (s >= 0 && s < s_count && !closed_mask[static_cast<size_t>(s)]) {
                next_order.push_back(s);
            }
        }
        position_order.swap(next_order);
    };

    for (ssize_t t = 0; t < t_count; ++t) {
        for (ssize_t s = 0; s < s_count; ++s) {
            if (present(t, s)) {
                const double px = close(t, s);
                if (std::isfinite(px)) {
                    last_close[static_cast<size_t>(s)] = px;
                }
            }
        }

        const double mtm_before = mark_to_market();
        const double equity_before = cash + mtm_before;

        if (margin_annual_rate > 0.0) {
            const double position_value = std::abs(mtm_before);
            const double margin_used = std::max(0.0, position_value - std::max(0.0, equity_before));
            if (margin_used > 0.0) {
                cash -= margin_used * margin_annual_rate / 8760.0;
            }
        }

        std::vector<int32_t> closed;
        for (int32_t s : position_order) {
            PositionState& pos = positions[static_cast<size_t>(s)];
            if (!pos.active || !present(t, s)) {
                continue;
            }
            if (!(pos.sell_target > 0.0)) {
                continue;
            }

            const double fee_rate = fee(s);
            bool target_hit = false;
            if (pos.direction < 0) {
                target_hit = low(t, s) <= pos.sell_target * (1.0 - bar_margin);
            } else {
                target_hit = high(t, s) >= pos.sell_target * (1.0 + bar_margin);
            }
            if (!target_hit) {
                continue;
            }
            if (enforce_market_hours && !is_crypto(s) && !market_open_now(t, s)) {
                continue;
            }

            if (pos.direction < 0) {
                cash += pos.qty * (2.0 * pos.entry_price - pos.sell_target * (1.0 + fee_rate));
                push_trade(
                    static_cast<int32_t>(t),
                    static_cast<int32_t>(s),
                    SIDE_BUY_COVER,
                    pos.sell_target,
                    pos.qty,
                    cash,
                    0.0,
                    REASON_TARGET);
            } else {
                cash += pos.qty * pos.sell_target * (1.0 - fee_rate);
                push_trade(
                    static_cast<int32_t>(t),
                    static_cast<int32_t>(s),
                    SIDE_SELL,
                    pos.sell_target,
                    pos.qty,
                    cash,
                    0.0,
                    REASON_TARGET);
            }
            closed.push_back(s);
        }
        remove_closed_positions(closed);

        std::vector<int32_t> closed_timeout;
        for (int32_t s : position_order) {
            PositionState& pos = positions[static_cast<size_t>(s)];
            if (!pos.active) {
                continue;
            }

            if (market_open_now(t, s) || is_crypto(s)) {
                pos.market_hours_held += 1;
            }
            if (max_hold_hours <= 0 || pos.market_hours_held < max_hold_hours) {
                continue;
            }
            if (enforce_market_hours && !is_crypto(s) && !market_open_now(t, s)) {
                continue;
            }

            const double fee_rate = fee(s);
            double px = last_close[static_cast<size_t>(s)];
            if (!std::isfinite(px)) {
                px = pos.entry_price;
            }
            if (pos.direction < 0) {
                const double exit_px = px * (1.0 + force_close_slippage);
                cash += pos.qty * (2.0 * pos.entry_price - exit_px * (1.0 + fee_rate));
                push_trade(
                    static_cast<int32_t>(t),
                    static_cast<int32_t>(s),
                    SIDE_BUY_COVER,
                    exit_px,
                    pos.qty,
                    cash,
                    0.0,
                    REASON_TIMEOUT);
            } else {
                const double exit_px = px * (1.0 - force_close_slippage);
                cash += pos.qty * exit_px * (1.0 - fee_rate);
                push_trade(
                    static_cast<int32_t>(t),
                    static_cast<int32_t>(s),
                    SIDE_SELL,
                    exit_px,
                    pos.qty,
                    cash,
                    0.0,
                    REASON_TIMEOUT);
            }
            closed_timeout.push_back(s);
        }
        remove_closed_positions(closed_timeout);

        if (close_at_eod) {
            std::vector<int32_t> closed_eod;
            for (int32_t s : position_order) {
                PositionState& pos = positions[static_cast<size_t>(s)];
                if (!pos.active || is_crypto(s)) {
                    continue;
                }
                if (!(market_open_now(t, s) && !market_open_next(t, s))) {
                    continue;
                }

                const double fee_rate = fee(s);
                double px = last_close[static_cast<size_t>(s)];
                if (!std::isfinite(px)) {
                    px = pos.entry_price;
                }
                if (pos.direction < 0) {
                    const double exit_px = px * (1.0 + force_close_slippage);
                    cash += pos.qty * (2.0 * pos.entry_price - exit_px * (1.0 + fee_rate));
                    push_trade(
                        static_cast<int32_t>(t),
                        static_cast<int32_t>(s),
                        SIDE_BUY_COVER,
                        exit_px,
                        pos.qty,
                        cash,
                        0.0,
                        REASON_EOD);
                } else {
                    const double exit_px = px * (1.0 - force_close_slippage);
                    cash += pos.qty * exit_px * (1.0 - fee_rate);
                    push_trade(
                        static_cast<int32_t>(t),
                        static_cast<int32_t>(s),
                        SIDE_SELL,
                        exit_px,
                        pos.qty,
                        cash,
                        0.0,
                        REASON_EOD);
                }
                closed_eod.push_back(s);
            }
            remove_closed_positions(closed_eod);
        }

        const int32_t open_count = static_cast<int32_t>(position_order.size());
        const int32_t open_slots = static_cast<int32_t>(max_positions) - open_count;
        if (open_slots <= 0) {
            equity_values[static_cast<size_t>(t)] = cash + mark_to_market();
            continue;
        }

        const double equity_for_entries = cash + mark_to_market();
        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(s_count));

        for (ssize_t s = 0; s < s_count; ++s) {
            if (!present(t, s) || positions[static_cast<size_t>(s)].active) {
                continue;
            }
            if (enforce_market_hours && !is_crypto(s) && !market_open_now(t, s)) {
                continue;
            }

            const bool is_long = direction(s) >= 0;
            const bool is_short = !is_long;
            const double buy_px = buy_price(t, s);
            const double sell_px = sell_price(t, s);

            const double signal_amount = directional_entry_amount(
                is_short, buy_amount(t, s), sell_amount(t, s), trade_amount(t, s));
            const double side_multiplier = is_short ? short_intensity_multiplier : long_intensity_multiplier;
            const double intensity_fraction = calibrated_intensity(
                signal_amount,
                trade_amount_scale,
                entry_intensity_power,
                entry_min_intensity_fraction,
                side_multiplier);

            if (!finite_positive(buy_px) || signal_amount <= 0.0) {
                continue;
            }
            if (min_buy_amount > 0.0 && signal_amount < min_buy_amount) {
                continue;
            }

            const double fee_rate = fee(s);
            const bool pred_high_ok = finite_positive(pred_high(t, s));
            const bool pred_low_ok = finite_positive(pred_low(t, s));
            const bool pred_close_ok = finite_positive(pred_close(t, s));

            double edge = std::numeric_limits<double>::quiet_NaN();
            double entry_px = std::numeric_limits<double>::quiet_NaN();
            double exit_px = std::numeric_limits<double>::quiet_NaN();

            if (is_long) {
                if (pred_high_ok && pred_low_ok && pred_close_ok) {
                    edge = edge_score(
                        pred_high(t, s),
                        pred_low(t, s),
                        pred_close(t, s),
                        buy_px,
                        true,
                        use_close_edge,
                        fee_rate);
                } else if (finite_positive(sell_px)) {
                    edge = (sell_px - buy_px) / buy_px - fee_rate;
                } else {
                    continue;
                }
                entry_px = buy_px;
                exit_px = sell_px;
            } else {
                if (!finite_positive(sell_px)) {
                    continue;
                }
                if (pred_low_ok) {
                    edge = (sell_px - pred_low(t, s)) / sell_px - fee_rate;
                } else {
                    edge = (sell_px - buy_px) / sell_px - fee_rate;
                }
                entry_px = sell_px;
                exit_px = buy_px;
            }

            if (!std::isfinite(edge) || edge < min_edge) {
                continue;
            }

            bool fillable = false;
            double actual_entry_px = entry_px;
            if (market_order_entry) {
                fillable = true;
                actual_entry_px = open(t, s);
                if (!std::isfinite(actual_entry_px)) {
                    actual_entry_px = close(t, s);
                }
            } else {
                if (is_long) {
                    fillable = low(t, s) <= entry_px * (1.0 - bar_margin);
                } else {
                    fillable = high(t, s) >= entry_px * (1.0 + bar_margin);
                }
            }
            if (!fillable || !finite_positive(actual_entry_px)) {
                continue;
            }

            Candidate cand;
            cand.sym = static_cast<int32_t>(s);
            cand.direction = is_long ? 1 : -1;
            cand.edge = edge;
            cand.entry_price = actual_entry_px;
            cand.exit_price = exit_px;
            cand.signal_amount = signal_amount;
            cand.intensity_fraction = intensity_fraction;
            cand.required_move_frac = required_move_to_fill(open(t, s), actual_entry_px, is_long, bar_margin);
            candidates.push_back(cand);
        }

        if (first_trigger_mode) {
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const Candidate& a, const Candidate& b) {
                    if (a.required_move_frac == b.required_move_frac) {
                        return a.edge > b.edge;
                    }
                    return a.required_move_frac < b.required_move_frac;
                });
        } else {
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const Candidate& a, const Candidate& b) { return a.edge > b.edge; });
        }

        const int32_t fill_limit =
            std::min<int32_t>(open_slots, static_cast<int32_t>(candidates.size()));
        for (int32_t idx = 0; idx < fill_limit; ++idx) {
            const Candidate& cand = candidates[static_cast<size_t>(idx)];
            const int32_t s = cand.sym;
            const double fee_rate = fee(s);
            const double sym_leverage = is_crypto(s) ? 1.0 : max_leverage;
            const double per_position_alloc =
                (equity_for_entries * sym_leverage) / static_cast<double>(max_positions);
            const double alloc = per_position_alloc * cand.intensity_fraction;
            double qty = alloc / (cand.entry_price * (1.0 + fee_rate));
            if (!std::isfinite(qty)) {
                continue;
            }
            if (int_qty) {
                qty = std::floor(qty);
            }
            if (qty <= 0.0) {
                continue;
            }

            const double cost = qty * cand.entry_price * (1.0 + fee_rate);
            cash -= cost;

            PositionState pos;
            pos.active = true;
            pos.qty = qty;
            pos.entry_price = cand.entry_price;
            pos.sell_target = cand.exit_price;
            pos.market_hours_held = 0;
            pos.direction = cand.direction;
            positions[static_cast<size_t>(s)] = pos;
            position_order.push_back(s);

            const int8_t side = (cand.direction < 0) ? SIDE_SHORT_SELL : SIDE_BUY;
                push_trade(
                    static_cast<int32_t>(t),
                    s,
                    side,
                    cand.entry_price,
                qty,
                cash,
                qty,
                    REASON_ENTRY);
        }

        equity_values[static_cast<size_t>(t)] = cash + mark_to_market();
    }

    const double final_equity = cash + mark_to_market();

    py::array_t<double> equity_out(static_cast<ssize_t>(equity_values.size()));
    auto eq_mut = equity_out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < static_cast<ssize_t>(equity_values.size()); ++i) {
        eq_mut(i) = equity_values[static_cast<size_t>(i)];
    }

    const ssize_t k = static_cast<ssize_t>(trades.size());
    py::array_t<int32_t> trade_t_idx(k);
    py::array_t<int32_t> trade_sym_idx(k);
    py::array_t<int8_t> trade_side(k);
    py::array_t<double> trade_price(k);
    py::array_t<double> trade_qty(k);
    py::array_t<double> trade_cash_after(k);
    py::array_t<double> trade_inventory_after(k);
    py::array_t<int8_t> trade_reason(k);

    auto t_idx_mut = trade_t_idx.mutable_unchecked<1>();
    auto sym_idx_mut = trade_sym_idx.mutable_unchecked<1>();
    auto side_mut = trade_side.mutable_unchecked<1>();
    auto price_mut = trade_price.mutable_unchecked<1>();
    auto qty_mut = trade_qty.mutable_unchecked<1>();
    auto cash_mut = trade_cash_after.mutable_unchecked<1>();
    auto inv_mut = trade_inventory_after.mutable_unchecked<1>();
    auto reason_mut = trade_reason.mutable_unchecked<1>();
    for (ssize_t i = 0; i < k; ++i) {
        const TradeEvent& ev = trades[static_cast<size_t>(i)];
        t_idx_mut(i) = ev.t_idx;
        sym_idx_mut(i) = ev.sym_idx;
        side_mut(i) = ev.side;
        price_mut(i) = ev.price;
        qty_mut(i) = ev.qty;
        cash_mut(i) = ev.cash_after;
        inv_mut(i) = ev.inventory_after;
        reason_mut(i) = ev.reason;
    }

    py::dict out;
    out["equity_values"] = std::move(equity_out);
    out["final_equity"] = final_equity;
    out["trade_t_idx"] = std::move(trade_t_idx);
    out["trade_sym_idx"] = std::move(trade_sym_idx);
    out["trade_side"] = std::move(trade_side);
    out["trade_price"] = std::move(trade_price);
    out["trade_qty"] = std::move(trade_qty);
    out["trade_cash_after"] = std::move(trade_cash_after);
    out["trade_inventory_after"] = std::move(trade_inventory_after);
    out["trade_reason"] = std::move(trade_reason);
    out["timestamps_ns"] = std::move(timestamps_ns_arr);
    return out;
}

PYBIND11_MODULE(portfolio_sim_native_ext, m) {
    m.doc() = "Native dense backend for unified portfolio simulation";
    m.def("simulate_portfolio_dense", &simulate_portfolio_dense);
}
