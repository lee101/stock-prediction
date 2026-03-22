#ifndef POLICY_INFER_H
#define POLICY_INFER_H

#define POLICY_MAX_OBS 512
#define POLICY_MAX_ACTIONS 16

typedef struct {
    void *model;
    int obs_dim;
    int action_dim;
    int loaded;
} Policy;

typedef struct {
    double buy_price;
    double sell_price;
    double buy_amount;
    double sell_amount;
} PolicyAction;

int policy_load(Policy *policy, const char *model_path);
void policy_unload(Policy *policy);

int policy_forward(
    Policy *policy,
    const double *obs,
    int obs_len,
    PolicyAction *out_actions,
    int n_symbols
);

int policy_export_torchscript(const char *checkpoint_path, const char *output_path);

#endif
