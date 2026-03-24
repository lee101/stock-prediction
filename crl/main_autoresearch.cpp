#include "autoresearch.h"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    std::string train_data, val_data, experiments_file = "experiments.json";
    std::string checkpoint_root = "checkpoints";
    std::string leaderboard = "leaderboard.csv";
    int time_budget = 300;
    int max_trials = 15;
    bool once = false;
    bool use_gpu = false;
    float baseline_return = 1.914f;
    float baseline_sortino = 23.94f;

    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };

        if (arg("--train-data")) train_data = next();
        else if (arg("--val-data")) val_data = next();
        else if (arg("--experiments")) experiments_file = next();
        else if (arg("--checkpoint-root")) checkpoint_root = next();
        else if (arg("--leaderboard")) leaderboard = next();
        else if (arg("--time-budget")) time_budget = atoi(next());
        else if (arg("--max-trials")) max_trials = atoi(next());
        else if (arg("--once")) once = true;
        else if (arg("--gpu")) use_gpu = true;
        else if (arg("--baseline-return")) baseline_return = atof(next());
        else if (arg("--baseline-sortino")) baseline_sortino = atof(next());
    }

    if (train_data.empty()) {
        fprintf(stderr, "Usage: crl_autoresearch --train-data PATH --val-data PATH [options]\n"
                "  --experiments FILE     JSON config (default experiments.json)\n"
                "  --checkpoint-root DIR  Checkpoint directory (default checkpoints)\n"
                "  --leaderboard FILE     CSV output (default leaderboard.csv)\n"
                "  --time-budget SEC      Seconds per trial (default 300)\n"
                "  --max-trials N         Max trials per round (default 15)\n"
                "  --once                 Run one round then exit\n"
                "  --gpu                  Use CUDA\n"
                "  --baseline-return F    Baseline return to beat (default 1.914)\n"
                "  --baseline-sortino F   Baseline sortino to beat (default 23.94)\n");
        return 1;
    }

    torch::Device device = use_gpu && torch::cuda::is_available()
        ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    Autoresearch ar(train_data, val_data, time_budget, max_trials,
                    checkpoint_root, leaderboard, device);
    ar.set_baseline(baseline_return, baseline_sortino);
    ar.load_experiments(experiments_file);
    ar.run_forever(once);

    return 0;
}
