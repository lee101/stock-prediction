/*
 * Verify the C obs_builder produces the exact same 209-dim vector as the
 * Python inference.build_observation helper when the portfolio is flat.
 * Fixture: scripts/gen_obs_parity_fixture.py.
 */
#include "../obs_builder.h"
#include "../../mktd_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_all(FILE *f, void *b, size_t n) {
    return fread(b, 1, n, f) == n ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <data.mktd> <fixture.bin>\n", argv[0]);
        return 2;
    }

    MktdData data;
    if (mktd_load(argv[1], &data) != 0) {
        fprintf(stderr, "FAIL: mktd_load failed on %s\n", argv[1]);
        return 1;
    }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "FAIL: open fixture\n"); return 1; }

    uint32_t obs_dim, S, F, t;
    if (read_all(f, &obs_dim, 4) || read_all(f, &S, 4) ||
        read_all(f, &F, 4) || read_all(f, &t, 4)) {
        fprintf(stderr, "FAIL: header\n"); return 1;
    }
    if ((int)S != data.num_symbols || (int)F != data.features_per_sym) {
        fprintf(stderr, "FAIL: shape mismatch (S=%u F=%u) vs mktd (S=%d F=%d)\n",
                S, F, data.num_symbols, data.features_per_sym);
        return 1;
    }
    float *expected = (float *)malloc(obs_dim * sizeof(float));
    if (read_all(f, expected, obs_dim * sizeof(float))) {
        fprintf(stderr, "FAIL: expected obs read\n"); return 1;
    }
    fclose(f);

    CtrdpolPortfolioState pf;
    ctrdpol_init_portfolio(&pf, 10000.0, 90);
    pf.step = (int)t;     /* matches fixture: step=t, hold_hours=0 */

    float *got = (float *)malloc(obs_dim * sizeof(float));
    if (ctrdpol_build_obs(&data, (int)t, &pf, got, (int)obs_dim) != 0) {
        fprintf(stderr, "FAIL: build_obs returned nonzero\n"); return 1;
    }

    double max_abs = 0.0;
    int bad = -1;
    for (uint32_t i = 0; i < obs_dim; i++) {
        double d = fabs((double)got[i] - (double)expected[i]);
        if (d > max_abs) { max_abs = d; bad = (int)i; }
    }
    fprintf(stderr, "=== obs_builder parity (t=%u) ===\n", t);
    fprintf(stderr, "  max_abs_diff = %.3e  at idx %d  (tol 1e-6)\n", max_abs, bad);
    if (max_abs > 1e-6) {
        fprintf(stderr, "  got[%d]=%+.6f expected[%d]=%+.6f\n", bad, got[bad], bad, expected[bad]);
        fprintf(stderr, "FAIL\n");
        mktd_free(&data);
        return 1;
    }
    fprintf(stderr, "OK\n");
    mktd_free(&data);
    free(got); free(expected);
    return 0;
}
