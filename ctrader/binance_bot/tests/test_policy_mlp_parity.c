/*
 * Verify the pure-C policy_mlp.c forward pass matches the Python reference
 * exactly on a fixed RNG observation.  Fixture produced by
 * scripts/gen_policy_mlp_parity_fixture.py.
 *
 * Build:
 *   TMPDIR=$PWD/.tmp gcc -O2 -Wall -std=c11 \
 *       -I../. -o .tmp/test_policy_mlp_parity \
 *       tests/test_policy_mlp_parity.c policy_mlp.c -lm
 * Run (from ctrader/binance_bot):
 *   ./.tmp/test_policy_mlp_parity \
 *       ../models/stocks12_v5_rsi_s42.ctrdpol \
 *       tests/policy_mlp_parity_fixture.bin
 */
#include "../policy_mlp.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_all(FILE *f, void *buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <policy.ctrdpol> <fixture.bin>\n", argv[0]);
        return 2;
    }

    CtrdpolPolicy pol;
    if (ctrdpol_load(&pol, argv[1]) != 0) {
        fprintf(stderr, "FAIL: cannot load policy\n");
        return 1;
    }
    fprintf(stderr, "loaded %s  obs_dim=%u hidden=%u actor_hidden=%u num_actions=%u\n",
            argv[1], pol.hdr.obs_dim, pol.hdr.hidden, pol.hdr.actor_hidden, pol.hdr.num_actions);

    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "FAIL: cannot open fixture %s\n", argv[2]);
        ctrdpol_free(&pol);
        return 1;
    }

    uint32_t obs_dim = 0, num_actions = 0;
    if (read_all(f, &obs_dim, 4) != 0 || read_all(f, &num_actions, 4) != 0) {
        fprintf(stderr, "FAIL: short header read\n");
        fclose(f); ctrdpol_free(&pol); return 1;
    }
    if (obs_dim != pol.hdr.obs_dim || num_actions != pol.hdr.num_actions) {
        fprintf(stderr, "FAIL: fixture shape mismatch: fixture=%u/%u policy=%u/%u\n",
                obs_dim, num_actions, pol.hdr.obs_dim, pol.hdr.num_actions);
        fclose(f); ctrdpol_free(&pol); return 1;
    }

    float *obs      = (float *)malloc(obs_dim * sizeof(float));
    float *expected = (float *)malloc(num_actions * sizeof(float));
    float *got      = (float *)malloc(num_actions * sizeof(float));
    if (!obs || !expected || !got) { fprintf(stderr, "oom\n"); return 1; }

    if (read_all(f, obs, obs_dim * sizeof(float)) != 0 ||
        read_all(f, expected, num_actions * sizeof(float)) != 0) {
        fprintf(stderr, "FAIL: short body read\n");
        return 1;
    }
    int32_t expected_argmax = 0;
    if (read_all(f, &expected_argmax, 4) != 0) {
        fprintf(stderr, "FAIL: short argmax read\n");
        return 1;
    }
    fclose(f);

    if (ctrdpol_forward(&pol, obs, got) != 0) {
        fprintf(stderr, "FAIL: forward returned nonzero\n");
        return 1;
    }

    double max_abs_diff = 0.0;
    int argmax_diff_bin = -1;
    for (uint32_t i = 0; i < num_actions; i++) {
        double d = fabs((double)got[i] - (double)expected[i]);
        if (d > max_abs_diff) { max_abs_diff = d; argmax_diff_bin = (int)i; }
    }
    int got_argmax = ctrdpol_argmax(got, (int)num_actions);

    const double TOL = 1e-4;
    int failed = 0;
    fprintf(stderr, "\n=== policy_mlp parity ===\n");
    fprintf(stderr, "  max_abs_diff  = %.3e (tol %.0e) at logit %d\n",
            max_abs_diff, TOL, argmax_diff_bin);
    fprintf(stderr, "  expected_argmax = %d   got_argmax = %d\n", expected_argmax, got_argmax);
    fprintf(stderr, "  expected[%d] = %+.6f   got[%d] = %+.6f\n",
            expected_argmax, expected[expected_argmax],
            expected_argmax, got[expected_argmax]);

    if (max_abs_diff > TOL) {
        fprintf(stderr, "FAIL: logits exceed tolerance\n");
        failed = 1;
    }
    if (got_argmax != expected_argmax) {
        fprintf(stderr, "FAIL: argmax mismatch\n");
        failed = 1;
    }

    free(obs); free(expected); free(got);
    ctrdpol_free(&pol);

    if (!failed) {
        fprintf(stderr, "OK\n");
        return 0;
    }
    return 1;
}
