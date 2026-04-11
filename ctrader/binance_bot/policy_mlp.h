#ifndef CTRADER_BINANCE_BOT_POLICY_MLP_H
#define CTRADER_BINANCE_BOT_POLICY_MLP_H

#include <stddef.h>
#include <stdint.h>

/*
 * Pure-C loader + forward pass for pufferlib_market stocks12-style MLP
 * policies (TradingPolicy with optional encoder LayerNorm).  Zero third-party
 * dependencies (no libtorch, no BLAS).  Weights are produced by
 * scripts/export_policy_to_ctrader.py; see that file for the binary format
 * contract.
 *
 * Designed so the live Binance bot (ctrader/binance_bot/main_live.c,
 * planned) and the training loop load *the same* weights and therefore
 * produce identical actions for a given observation.  A parity test
 * (tests/test_policy_mlp_parity.c / parity_fixture) enforces this.
 */

#define CTRDPOL_MAGIC      "CTRDPOL1"
#define CTRDPOL_VERSION    1u
#define CTRDPOL_HEADER_SIZE 64u

typedef enum {
    CTRDPOL_ACT_RELU    = 0,
    CTRDPOL_ACT_RELU_SQ = 1,  /* not yet implemented in forward pass */
} CtrdpolActivation;

typedef struct {
    uint32_t obs_dim;
    uint32_t hidden;
    uint32_t n_encoder_layers;
    uint32_t use_encoder_norm;    /* 0/1 */
    uint32_t actor_hidden;
    uint32_t num_actions;
    uint32_t num_symbols;
    uint32_t activation;          /* CtrdpolActivation */
    uint32_t disable_shorts;      /* 0/1 */
} CtrdpolHeader;

typedef struct {
    CtrdpolHeader hdr;

    /* Encoder: n_encoder_layers Linear layers, each followed by ReLU.
     * Layer 0 is [hidden × obs_dim]; subsequent layers are [hidden × hidden].
     * Biases are [hidden]. */
    float **encoder_weights;   /* [n_encoder_layers][hidden * in_dim_l] */
    float **encoder_biases;    /* [n_encoder_layers][hidden] */

    float *encoder_norm_weight;  /* [hidden] if use_encoder_norm, else NULL */
    float *encoder_norm_bias;    /* [hidden] */

    /* Actor head: Linear(hidden, actor_hidden) -> ReLU -> Linear(actor_hidden, num_actions) */
    float *actor0_weight;      /* [actor_hidden × hidden] */
    float *actor0_bias;        /* [actor_hidden] */
    float *actor2_weight;      /* [num_actions × actor_hidden] */
    float *actor2_bias;        /* [num_actions] */

    /* Scratch buffers owned by the policy, sized once at load.  NOT
     * thread-safe: one CtrdpolPolicy instance per thread. */
    float *scratch_enc_in;     /* [obs_dim]    initialised from caller obs */
    float *scratch_enc_a;      /* [hidden] */
    float *scratch_enc_b;      /* [hidden] */
    float *scratch_actor;      /* [actor_hidden] */
    float *scratch_logits;     /* [num_actions] */
} CtrdpolPolicy;

/* Load a .ctrdpol file from disk into `pol`.  Returns 0 on success, nonzero
 * on error (prints to stderr).  Caller must call ctrdpol_free() afterwards. */
int  ctrdpol_load(CtrdpolPolicy *pol, const char *path);

/* Free all weights + scratch owned by `pol`. */
void ctrdpol_free(CtrdpolPolicy *pol);

/* Forward pass.  `obs_f32` must have length `pol->hdr.obs_dim` and match the
 * feature layout the Python `pufferlib_market` trainer uses (209 dims for
 * stocks12 v5_rsi).  `out_logits` receives `pol->hdr.num_actions` floats.
 * Returns 0 on success.
 *
 * Use ctrdpol_argmax() below to pick the deterministic action. */
int  ctrdpol_forward(const CtrdpolPolicy *pol, const float *obs_f32, float *out_logits);

/* Convenience: argmax of a logits vector.  Returns -1 if n == 0. */
int  ctrdpol_argmax(const float *logits, int n);

#endif
