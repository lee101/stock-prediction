#include "policy_infer.h"
#include <stdio.h>
#include <string.h>

/*
 * Stub implementation. Real version links against libtorch C API:
 *   #include <torch/csrc/api/include/torch/torch.h>
 * or the C-only API:
 *   torch_jit_load(), at::Tensor, etc.
 *
 * Build with:
 *   -I$(TORCH_DIR)/include -L$(TORCH_DIR)/lib -ltorch -lc10
 */

int policy_load(Policy *policy, const char *model_path) {
    memset(policy, 0, sizeof(*policy));
    fprintf(stderr, "policy_load(%s): stub -- needs libtorch\n", model_path);
    policy->loaded = 0;
    return -1;
}

void policy_unload(Policy *policy) {
    if (policy->model) {
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
    (void)obs; (void)obs_len;
    if (!policy->loaded) {
        fprintf(stderr, "policy_forward: model not loaded\n");
        return -1;
    }
    for (int i = 0; i < n_symbols; i++) {
        out_actions[i].buy_price = 0.0;
        out_actions[i].sell_price = 0.0;
        out_actions[i].buy_amount = 0.0;
        out_actions[i].sell_amount = 0.0;
    }
    return 0;
}

int policy_export_torchscript(const char *checkpoint_path, const char *output_path) {
    (void)checkpoint_path; (void)output_path;
    fprintf(stderr, "policy_export_torchscript: stub -- needs Python torch.jit.trace\n");
    return -1;
}
