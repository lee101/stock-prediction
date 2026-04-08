#ifndef CTRADER_BINANCE_BOT_OBS_BUILDER_H
#define CTRADER_BINANCE_BOT_OBS_BUILDER_H

#include <stdint.h>

#include "../mktd_reader.h"

/*
 * Build the 209-dim observation vector that the stocks12 v5_rsi policies
 * (and friends) were trained against.  This mirrors exactly
 * `pufferlib_market/inference.py:254 build_observation` so the C pipeline
 * feeds the policy identical input to training.
 *
 * obs_layout (S = num_symbols, F = features_per_sym, typically 12 and 16):
 *   [0            .. S*F)         per-symbol features, row-major [sym0..symS][f0..fF]
 *   [S*F          .. S*F+5)       portfolio state: cash/10k, pos_val/10k, 0, hold_frac, step_frac
 *   [S*F+5        .. S*F+5+S)     one-hot position (+1 long, -1 short on that symbol, else 0)
 *
 *   Total dim = S*F + 5 + S
 *
 * Caller owns `out_obs`, which must have length S*F + 5 + S float32 slots.
 */

typedef struct {
    double cash;
    double equity;

    /* -1 if flat; 0..S-1 for long on that sym; S..2S-1 for short. Matches the
     * "position index" convention used in inference.py. */
    int current_position;
    double position_qty;
    double entry_price;

    int hold_hours;      /* bars spent holding current position */
    int step;            /* bar index within the current episode */
    int max_steps;       /* episode length (used to normalise hold/step) */
} CtrdpolPortfolioState;

void ctrdpol_init_portfolio(CtrdpolPortfolioState *pf, double initial_cash, int max_steps);

/* Build obs from an MKTD file slice at time `t` (features[t, sym, :]) plus
 * current portfolio state plus current prices (closes[t, sym]).
 *
 *   out_obs must be [num_symbols * features_per_sym + 5 + num_symbols]
 *   returns 0 on success, -1 on dimension mismatch.                     */
int ctrdpol_build_obs(
    const MktdData *data,
    int t,
    const CtrdpolPortfolioState *pf,
    float *out_obs,
    int out_obs_len
);

#endif
