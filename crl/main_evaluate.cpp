#include "evaluate.h"
#include "checkpoint.h"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    std::string checkpoint_path, data_path;
    int n_windows = 20;
    int window_length = 720;
    int hidden = 1024;
    float fill_slippage = 5.0f;
    float fee_rate = 0.001f;
    bool use_gpu = false;
    bool deterministic = true;

    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };

        if (arg("--checkpoint")) checkpoint_path = next();
        else if (arg("--data-path")) data_path = next();
        else if (arg("--n-windows")) n_windows = atoi(next());
        else if (arg("--window-length")) window_length = atoi(next());
        else if (arg("--hidden")) hidden = atoi(next());
        else if (arg("--fill-slippage-bps")) fill_slippage = atof(next());
        else if (arg("--fee-rate")) fee_rate = atof(next());
        else if (arg("--gpu")) use_gpu = true;
        else if (arg("--stochastic")) deterministic = false;
    }

    if (checkpoint_path.empty() || data_path.empty()) {
        fprintf(stderr, "Usage: crl_evaluate --checkpoint PATH --data-path PATH [options]\n"
                "  --n-windows N          Number of eval windows (default 20)\n"
                "  --window-length N      Hours per window (default 720)\n"
                "  --hidden N             Hidden size (default 1024)\n"
                "  --fill-slippage-bps F  Fill slippage bps (default 5)\n"
                "  --fee-rate F           Fee rate (default 0.001)\n"
                "  --gpu                  Use CUDA\n"
                "  --stochastic           Sample actions instead of argmax\n");
        return 1;
    }

    torch::Device device = use_gpu && torch::cuda::is_available()
        ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    EvalConfig cfg;
    cfg.n_windows = n_windows;
    cfg.window_length = window_length;
    cfg.deterministic = deterministic;
    cfg.fill_slippage_bps = fill_slippage;
    cfg.fee_rate = fee_rate;

    auto summary = evaluate_holdout(checkpoint_path, data_path, cfg, nullptr, hidden, device);

    printf("=== Holdout Evaluation ===\n");
    printf("Windows: %d\n", summary.n_windows);
    printf("Mean Return: %.4f\n", summary.mean_return);
    printf("Median Return: %.4f\n", summary.median_return);
    printf("Mean Sortino: %.2f\n", summary.mean_sortino);
    printf("Median Sortino: %.2f\n", summary.median_sortino);
    printf("Mean MaxDD: %.3f\n", summary.mean_max_drawdown);
    printf("Mean WinRate: %.2f\n", summary.mean_win_rate);
    printf("\nPer-window:\n");
    for (int i = 0; i < (int)summary.windows.size(); i++) {
        auto& w = summary.windows[i];
        printf("  [%2d] ret=%.4f sort=%.2f dd=%.3f wr=%.2f trades=%.0f\n",
               i, w.total_return, w.sortino, w.max_drawdown, w.win_rate, w.num_trades);
    }

    return 0;
}
