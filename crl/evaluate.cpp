#include "evaluate.h"
#include "checkpoint.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>

static float median(std::vector<float>& v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    int n = v.size();
    if (n % 2 == 0) return (v[n/2 - 1] + v[n/2]) / 2.0f;
    return v[n/2];
}

EvalSummary evaluate_holdout(
    const std::string& checkpoint_path,
    const std::string& data_path,
    const EvalConfig& config,
    const RunningObsNorm* obs_norm,
    int hidden_size,
    torch::Device device
) {
    // load data to get metadata
    MarketData* md = market_data_load(data_path.c_str());
    if (!md) {
        fprintf(stderr, "evaluate_holdout: cannot load %s\n", data_path.c_str());
        return {};
    }

    int S = md->num_symbols;
    int F = md->features_per_sym;
    int obs_size = S * F + 5 + S;
    int alloc_bins = config.action_allocation_bins;
    int level_bins = config.action_level_bins;
    int num_actions = 1 + 2 * S * alloc_bins * level_bins;
    int num_timesteps = md->num_timesteps;
    market_data_free(md);

    // load policy
    TradingPolicy policy(obs_size, num_actions, hidden_size);
    try {
        torch::load(policy, checkpoint_path);
    } catch (...) {
        fprintf(stderr, "evaluate_holdout: cannot load checkpoint %s\n", checkpoint_path.c_str());
        return {};
    }
    policy->to(device);
    policy->eval();

    // compute window offsets (deterministic)
    int max_offset = num_timesteps - config.window_length - 1;
    if (max_offset < 0) max_offset = 0;
    std::vector<int> offsets(config.n_windows);
    for (int i = 0; i < config.n_windows; i++) {
        offsets[i] = (int)((long long)i * max_offset / std::max(config.n_windows - 1, 1));
    }

    VecEnvConfig env_cfg = {};
    env_cfg.max_steps = config.window_length;
    env_cfg.fee_rate = config.fee_rate;
    env_cfg.max_leverage = config.max_leverage;
    env_cfg.periods_per_year = config.periods_per_year;
    env_cfg.action_allocation_bins = alloc_bins;
    env_cfg.action_level_bins = level_bins;
    env_cfg.action_max_offset_bps = config.action_max_offset_bps;
    env_cfg.reward_scale = 10.0f;
    env_cfg.reward_clip = 5.0f;
    env_cfg.fill_slippage_bps = config.fill_slippage_bps;
    env_cfg.fill_probability = 1.0f;

    // run each window sequentially with 1 env
    EvalSummary summary;
    summary.n_windows = config.n_windows;

    for (int w = 0; w < config.n_windows; w++) {
        VecEnv* ve = vec_env_create(data_path.c_str(), 1, &env_cfg);
        if (!ve) continue;

        vec_env_set_offset(ve, 0, offsets[w]);
        vec_env_reset(ve, 42 + w);
        vec_env_set_offset(ve, 0, offsets[w]);
        c_reset(&ve->envs[0]);

        torch::NoGradGuard no_grad;
        bool done = false;
        while (!done) {
            auto obs_cpu = torch::from_blob(ve->obs_buf, {1, obs_size}, torch::kFloat32);
            if (obs_norm && obs_norm->size_ > 0) {
                auto norm_buf = std::vector<float>(obs_size);
                obs_norm->normalize(ve->obs_buf, norm_buf.data(), 1);
                obs_cpu = torch::from_blob(norm_buf.data(), {1, obs_size}, torch::kFloat32).clone();
            }

            auto obs_gpu = obs_cpu.to(device);
            auto [logits, value] = policy->forward(obs_gpu);

            int action;
            if (config.deterministic) {
                action = logits.argmax(-1).item<int>();
            } else {
                auto probs = torch::softmax(logits, -1);
                action = torch::multinomial(probs, 1).item<int>();
            }

            ve->act_buf[0] = action;
            vec_env_step(ve);
            done = ve->done_buf[0] != 0;
        }

        Log log;
        vec_env_get_log(ve, 0, &log);
        EvalResult res;
        res.total_return = (log.n > 0) ? log.total_return / log.n : 0;
        res.sortino = (log.n > 0) ? log.sortino / log.n : 0;
        res.max_drawdown = (log.n > 0) ? log.max_drawdown / log.n : 0;
        res.win_rate = (log.n > 0) ? log.win_rate / log.n : 0;
        res.num_trades = (log.n > 0) ? log.num_trades / log.n : 0;
        res.avg_hold_hours = (log.n > 0) ? log.avg_hold_hours / log.n : 0;
        summary.windows.push_back(res);

        vec_env_free(ve);
    }

    // compute aggregates
    std::vector<float> returns, sortinos, dds, wrs;
    for (auto& w : summary.windows) {
        returns.push_back(w.total_return);
        sortinos.push_back(w.sortino);
        dds.push_back(w.max_drawdown);
        wrs.push_back(w.win_rate);
    }
    summary.mean_return = std::accumulate(returns.begin(), returns.end(), 0.0f) / returns.size();
    summary.median_return = median(returns);
    summary.mean_sortino = std::accumulate(sortinos.begin(), sortinos.end(), 0.0f) / sortinos.size();
    summary.median_sortino = median(sortinos);
    summary.mean_max_drawdown = std::accumulate(dds.begin(), dds.end(), 0.0f) / dds.size();
    summary.mean_win_rate = std::accumulate(wrs.begin(), wrs.end(), 0.0f) / wrs.size();

    fprintf(stderr, "eval: %d windows, mean_ret=%.4f med_ret=%.4f mean_sort=%.2f med_sort=%.2f dd=%.3f\n",
            config.n_windows, summary.mean_return, summary.median_return,
            summary.mean_sortino, summary.median_sortino, summary.mean_max_drawdown);

    return summary;
}
