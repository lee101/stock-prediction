extern "C" {
#include "policy_infer.h"
}

#include <torch/script.h>
#include <cstring>
#include <cstdio>

int policy_load(Policy *policy, const char *model_path) {
    memset(policy, 0, sizeof(*policy));
    try {
        auto *mod = new torch::jit::Module(torch::jit::load(model_path));
        mod->eval();
        policy->model = static_cast<void *>(mod);
        policy->loaded = 1;
        return 0;
    } catch (const c10::Error &e) {
        fprintf(stderr, "policy_load(%s): %s\n", model_path, e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "policy_load(%s): unknown error\n", model_path);
        return -1;
    }
}

void policy_unload(Policy *policy) {
    if (policy->model) {
        auto *mod = static_cast<torch::jit::Module *>(policy->model);
        delete mod;
        policy->model = NULL;
    }
    policy->loaded = 0;
}

int policy_forward(
    Policy *policy,
    const double *obs,
    int obs_len,
    PolicyAction *out_actions,
    int n_symbols
) {
    if (!policy->loaded || !policy->model) {
        fprintf(stderr, "policy_forward: model not loaded\n");
        return -1;
    }

    auto *mod = static_cast<torch::jit::Module *>(policy->model);

    std::vector<float> fobs(obs_len);
    for (int i = 0; i < obs_len; i++)
        fobs[i] = static_cast<float>(obs[i]);

    try {
        auto input = torch::from_blob(fobs.data(), {1, obs_len}, torch::kFloat32);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        auto output = mod->forward(inputs).toTensor();
        output = output.contiguous().to(torch::kFloat64);
        int n_out = static_cast<int>(output.numel());

        const int fields_per_sym = 4;
        const double *data = output.data_ptr<double>();
        for (int i = 0; i < n_symbols; i++) {
            if (i * fields_per_sym + 3 < n_out) {
                out_actions[i].buy_price   = data[i * fields_per_sym + 0];
                out_actions[i].sell_price  = data[i * fields_per_sym + 1];
                out_actions[i].buy_amount  = data[i * fields_per_sym + 2];
                out_actions[i].sell_amount = data[i * fields_per_sym + 3];
            } else {
                out_actions[i].buy_price = 0.0;
                out_actions[i].sell_price = 0.0;
                out_actions[i].buy_amount = 0.0;
                out_actions[i].sell_amount = 0.0;
            }
        }
        return 0;
    } catch (const c10::Error &e) {
        fprintf(stderr, "policy_forward: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "policy_forward: unknown error\n");
        return -1;
    }
}

int policy_export_torchscript(const char *checkpoint_path, const char *output_path) {
    (void)checkpoint_path;
    (void)output_path;
    fprintf(stderr, "policy_export_torchscript: use Python torch.jit.trace to export\n");
    return -1;
}
