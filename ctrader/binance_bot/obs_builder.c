#include "obs_builder.h"

#include <stdio.h>
#include <string.h>

void ctrdpol_init_portfolio(CtrdpolPortfolioState *pf, double initial_cash, int max_steps) {
    memset(pf, 0, sizeof(*pf));
    pf->cash = initial_cash;
    pf->equity = initial_cash;
    pf->current_position = -1;
    pf->max_steps = max_steps;
}

int ctrdpol_build_obs(
    const MktdData *data,
    int t,
    const CtrdpolPortfolioState *pf,
    float *out_obs,
    int out_obs_len
) {
    int S = data->num_symbols;
    int F = data->features_per_sym;
    int want = S * F + 5 + S;
    if (out_obs_len != want) {
        fprintf(stderr, "obs_builder: len mismatch got %d want %d (S=%d F=%d)\n",
                out_obs_len, want, S, F);
        return -1;
    }
    if (t < 0 || t >= data->num_timesteps) {
        fprintf(stderr, "obs_builder: t=%d out of range [0,%d)\n", t, data->num_timesteps);
        return -1;
    }

    /* Copy per-symbol features at timestep t into [0, S*F). */
    const float *feat_row = data->features + (size_t)t * S * F;
    memcpy(out_obs, feat_row, (size_t)S * F * sizeof(float));

    /* Portfolio state at [S*F + 0..4]. */
    int base = S * F;
    double pos_val = 0.0;
    if (pf->current_position >= 0) {
        int sym_idx = pf->current_position % S;
        int is_short = (pf->current_position >= S);
        double cur_price = (double)data->closes[(size_t)t * S + sym_idx];
        if (is_short) pos_val = -pf->position_qty * cur_price;
        else          pos_val =  pf->position_qty * cur_price;
    }

    out_obs[base + 0] = (float)(pf->cash / 10000.0);
    out_obs[base + 1] = (float)(pos_val / 10000.0);
    out_obs[base + 2] = 0.0f; /* unrealized pnl (simplified, matches Python) */
    out_obs[base + 3] = pf->max_steps > 0 ? (float)((double)pf->hold_hours / pf->max_steps) : 0.0f;
    out_obs[base + 4] = pf->max_steps > 0 ? (float)((double)pf->step / pf->max_steps) : 0.0f;

    /* One-hot position at [S*F + 5 .. S*F + 5 + S). */
    for (int s = 0; s < S; s++) out_obs[base + 5 + s] = 0.0f;
    if (pf->current_position >= 0) {
        int sym_idx = pf->current_position % S;
        int is_short = (pf->current_position >= S);
        out_obs[base + 5 + sym_idx] = is_short ? -1.0f : 1.0f;
    }

    return 0;
}
