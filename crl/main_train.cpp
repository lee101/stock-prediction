#include "ppo.h"
#include "evaluate.h"
#include <cstdio>
#include <cstring>
#include <string>

static void print_usage() {
    fprintf(stderr,
        "Usage: crl_train [options]\n"
        "  --data-path PATH        Training data (.bin)\n"
        "  --val-data PATH         Validation data (.bin) [optional]\n"
        "  --checkpoint-dir DIR    Checkpoint output directory\n"
        "  --hidden SIZE           Hidden layer size (default 1024)\n"
        "  --lr RATE               Learning rate (default 3e-4)\n"
        "  --weight-decay WD       Weight decay (default 0)\n"
        "  --num-envs N            Parallel environments (default 128)\n"
        "  --rollout-len T         Rollout length (default 256)\n"
        "  --total-timesteps N     Total training timesteps (default 10M)\n"
        "  --time-budget SEC       Max training time in seconds (0=unlimited)\n"
        "  --seed N                Random seed (default 42)\n"
        "  --obs-norm              Enable observation normalization\n"
        "  --anneal-ent            Enable entropy annealing\n"
        "  --ent-coef F            Entropy coefficient (default 0.05)\n"
        "  --ent-coef-end F        Entropy end value (default 0.02)\n"
        "  --lr-schedule STR       LR schedule: none/cosine (default cosine)\n"
        "  --fee-rate F            Trading fee (default 0.001)\n"
        "  --fill-slippage-bps F   Fill slippage in bps (default 0)\n"
        "  --trade-penalty F       Per-trade penalty (default 0)\n"
        "  --cash-penalty F        Cash penalty (default 0.01)\n"
        "  --max-steps N           Episode length hours (default 720)\n"
        "  --max-leverage F        Max leverage (default 1.0)\n"
        "  --gpu                   Use GPU (default CPU)\n"
    );
}

int main(int argc, char** argv) {
    PPOConfig ppo_cfg;
    VecEnvConfig env_cfg = {};
    std::string data_path, val_path, ckpt_dir = "checkpoints/default";
    int time_budget = 0;
    bool use_gpu = false;

    // defaults
    env_cfg.max_steps = CRL_DEFAULT_MAX_STEPS;
    env_cfg.fee_rate = CRL_DEFAULT_FEE_RATE;
    env_cfg.max_leverage = 1.0f;
    env_cfg.periods_per_year = CRL_DEFAULT_PERIODS_PER_YEAR;
    env_cfg.action_allocation_bins = CRL_DEFAULT_ALLOC_BINS;
    env_cfg.action_level_bins = CRL_DEFAULT_LEVEL_BINS;
    env_cfg.action_max_offset_bps = CRL_DEFAULT_MAX_OFFSET_BPS;
    env_cfg.reward_scale = CRL_DEFAULT_REWARD_SCALE;
    env_cfg.reward_clip = CRL_DEFAULT_REWARD_CLIP;
    env_cfg.cash_penalty = CRL_DEFAULT_CASH_PENALTY;
    env_cfg.fill_probability = 1.0f;
    env_cfg.smooth_downside_temperature = 0.02f;

    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };

        if (arg("--data-path")) data_path = next();
        else if (arg("--val-data")) val_path = next();
        else if (arg("--checkpoint-dir")) ckpt_dir = next();
        else if (arg("--hidden")) ppo_cfg.hidden_size = atoi(next());
        else if (arg("--lr")) ppo_cfg.lr = atof(next());
        else if (arg("--weight-decay")) ppo_cfg.weight_decay = atof(next());
        else if (arg("--num-envs")) ppo_cfg.num_envs = atoi(next());
        else if (arg("--rollout-len")) ppo_cfg.rollout_len = atoi(next());
        else if (arg("--total-timesteps")) ppo_cfg.total_timesteps = atoi(next());
        else if (arg("--time-budget")) time_budget = atoi(next());
        else if (arg("--seed")) ppo_cfg.seed = atoi(next());
        else if (arg("--obs-norm")) ppo_cfg.obs_norm = true;
        else if (arg("--anneal-ent")) ppo_cfg.anneal_ent = true;
        else if (arg("--ent-coef")) ppo_cfg.ent_coef = atof(next());
        else if (arg("--ent-coef-end")) ppo_cfg.ent_coef_end = atof(next());
        else if (arg("--lr-schedule")) ppo_cfg.lr_schedule = next();
        else if (arg("--fee-rate")) env_cfg.fee_rate = atof(next());
        else if (arg("--fill-slippage-bps")) env_cfg.fill_slippage_bps = atof(next());
        else if (arg("--trade-penalty")) env_cfg.trade_penalty = atof(next());
        else if (arg("--cash-penalty")) env_cfg.cash_penalty = atof(next());
        else if (arg("--max-steps")) env_cfg.max_steps = atoi(next());
        else if (arg("--max-leverage")) env_cfg.max_leverage = atof(next());
        else if (arg("--downside-penalty")) env_cfg.downside_penalty = atof(next());
        else if (arg("--smooth-downside-penalty")) env_cfg.smooth_downside_penalty = atof(next());
        else if (arg("--smoothness-penalty")) env_cfg.smoothness_penalty = atof(next());
        else if (arg("--drawdown-penalty")) env_cfg.drawdown_penalty = atof(next());
        else if (arg("--gpu")) use_gpu = true;
        else if (arg("--help") || arg("-h")) { print_usage(); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); print_usage(); return 1; }
    }

    if (data_path.empty()) {
        fprintf(stderr, "Error: --data-path required\n");
        print_usage();
        return 1;
    }

    torch::Device device = use_gpu && torch::cuda::is_available()
        ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    fprintf(stderr, "Device: %s\n", device.is_cuda() ? "CUDA" : "CPU");

    PPOTrainer trainer(data_path, ppo_cfg, env_cfg, device);
    TrainResult result = trainer.train(ckpt_dir, time_budget);

    fprintf(stderr, "\nTraining complete. best_return=%.4f checkpoint=%s\n",
            result.val_return, result.checkpoint_path.c_str());

    // run holdout eval if val data provided
    if (!val_path.empty() && !result.checkpoint_path.empty()) {
        EvalConfig eval_cfg;
        eval_cfg.n_windows = 20;
        eval_cfg.window_length = env_cfg.max_steps;
        eval_cfg.fee_rate = env_cfg.fee_rate;
        eval_cfg.max_leverage = env_cfg.max_leverage;
        eval_cfg.periods_per_year = env_cfg.periods_per_year;
        eval_cfg.fill_slippage_bps = 5.0f;

        auto summary = evaluate_holdout(result.checkpoint_path, val_path, eval_cfg,
                                        ppo_cfg.obs_norm ? &trainer : nullptr,
                                        ppo_cfg.hidden_size, device);
        fprintf(stderr, "Holdout: mean_ret=%.4f mean_sort=%.2f mean_dd=%.3f\n",
                summary.mean_return, summary.mean_sortino, summary.mean_max_drawdown);
    }

    return 0;
}
