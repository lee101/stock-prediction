#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../mktd_reader.h"
#include "../market_sim.h"

static int g_pass = 0, g_fail = 0;

#define ASSERT_EQ_INT(a, b, msg) do { \
    int _a = (a), _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL %s: %d != %d\n", msg, _a, _b); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    double _a = (a), _b = (b), _t = (tol); \
    if (fabs(_a - _b) > _t) { \
        fprintf(stderr, "FAIL %s: %.10f != %.10f (tol=%.10f)\n", msg, _a, _b, _t); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s\n", msg); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

static int write_test_mktd(const char *path, int S, int T, int F) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "cannot create %s\n", path);
        return -1;
    }

    /* header: 64 bytes */
    unsigned int version = 2;
    unsigned int price_features = 5;
    char padding[40];
    memset(padding, 0, sizeof(padding));

    fwrite("MKTD", 1, 4, fp);
    fwrite(&version, 4, 1, fp);
    unsigned int us = (unsigned int)S;
    unsigned int ut = (unsigned int)T;
    unsigned int uf = (unsigned int)F;
    fwrite(&us, 4, 1, fp);
    fwrite(&ut, 4, 1, fp);
    fwrite(&uf, 4, 1, fp);
    fwrite(&price_features, 4, 1, fp);
    fwrite(padding, 1, 40, fp);

    /* symbol names */
    for (int i = 0; i < S; i++) {
        char name[16];
        memset(name, 0, 16);
        snprintf(name, 16, "SYM%d", i);
        fwrite(name, 1, 16, fp);
    }

    /* features: float[T][S][F] */
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < S; s++) {
            for (int f = 0; f < F; f++) {
                float val = (float)(t * 100 + s * 10 + f) * 0.01f;
                fwrite(&val, sizeof(float), 1, fp);
            }
        }
    }

    /* prices: float[T][S][5] (O,H,L,C,V) */
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < S; s++) {
            float base = 100.0f + (float)s * 50.0f + (float)t * 1.0f;
            float o = base;
            float h = base + 5.0f;
            float l = base - 3.0f;
            float c = base + 1.0f;
            float v = 1000.0f + (float)t;
            fwrite(&o, sizeof(float), 1, fp);
            fwrite(&h, sizeof(float), 1, fp);
            fwrite(&l, sizeof(float), 1, fp);
            fwrite(&c, sizeof(float), 1, fp);
            fwrite(&v, sizeof(float), 1, fp);
        }
    }

    /* tradable mask: uint8[T][S] -- all tradable */
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < S; s++) {
            unsigned char m = 1;
            fwrite(&m, 1, 1, fp);
        }
    }

    fclose(fp);
    return 0;
}

static void build_test_path(char *out, size_t out_size, const char *filename) {
    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !tmpdir[0]) {
        tmpdir = "/tmp";
    }
    snprintf(out, out_size, "%s/%s", tmpdir, filename);
}

static void test_load_basic(void) {
    char path[512];
    build_test_path(path, sizeof(path), "test_mktd_basic.bin");
    ASSERT_EQ_INT(write_test_mktd(path, 3, 10, 16), 0, "load_basic: fixture created");

    MktdData data;
    int rc = mktd_load(path, &data);
    ASSERT_EQ_INT(rc, 0, "load_basic: return code");
    if (rc != 0) return;
    ASSERT_EQ_INT(data.num_symbols, 3, "load_basic: num_symbols");
    ASSERT_EQ_INT(data.num_timesteps, 10, "load_basic: num_timesteps");
    ASSERT_EQ_INT(data.features_per_sym, 16, "load_basic: features_per_sym");

    ASSERT_TRUE(strncmp(data.symbol_names[0], "SYM0", 4) == 0, "load_basic: sym0 name");
    ASSERT_TRUE(strncmp(data.symbol_names[1], "SYM1", 4) == 0, "load_basic: sym1 name");
    ASSERT_TRUE(strncmp(data.symbol_names[2], "SYM2", 4) == 0, "load_basic: sym2 name");

    /* check first close: base=100+0*50+0*1=100, close=101 */
    ASSERT_NEAR(data.closes[0], 101.0f, 0.01, "load_basic: closes[0]");
    /* sym1 t=0: base=150, close=151 */
    ASSERT_NEAR(data.closes[1], 151.0f, 0.01, "load_basic: closes[1]");
    /* sym0 t=1: base=101, close=102 */
    ASSERT_NEAR(data.closes[3], 102.0f, 0.01, "load_basic: closes[t=1,s=0]");

    ASSERT_NEAR(data.opens[0], 100.0f, 0.01, "load_basic: opens[0]");
    ASSERT_NEAR(data.highs[0], 105.0f, 0.01, "load_basic: highs[0]");
    ASSERT_NEAR(data.lows[0], 97.0f, 0.01, "load_basic: lows[0]");
    ASSERT_NEAR(data.volumes[0], 1000.0f, 0.01, "load_basic: volumes[0]");

    ASSERT_TRUE(data.tradable != NULL, "load_basic: tradable not null");
    ASSERT_EQ_INT(data.tradable[0], 1, "load_basic: tradable[0]");

    /* features: t=0,s=0,f=0 => 0.0*0.01=0.0 */
    ASSERT_NEAR(data.features[0], 0.0f, 0.001, "load_basic: features[0]");
    /* t=1,s=0,f=0 => (1*100+0+0)*0.01 = 1.0 */
    ASSERT_NEAR(data.features[1 * 3 * 16], 1.0f, 0.001, "load_basic: features[t=1]");

    mktd_free(&data);
}

static void test_load_bad_magic(void) {
    char path[512];
    build_test_path(path, sizeof(path), "test_mktd_badmagic.bin");
    FILE *fp = fopen(path, "wb");
    char buf[64];
    memset(buf, 0, 64);
    memcpy(buf, "XYZW", 4);
    fwrite(buf, 1, 64, fp);
    fclose(fp);

    MktdData data;
    int rc = mktd_load(path, &data);
    ASSERT_EQ_INT(rc, -1, "bad_magic: should fail");
}

static void test_load_missing_file(void) {
    char path[512];
    build_test_path(path, sizeof(path), "nonexistent_mktd_file.bin");
    MktdData data;
    int rc = mktd_load(path, &data);
    ASSERT_EQ_INT(rc, -1, "missing_file: should fail");
}

static void test_load_truncated(void) {
    char path[512];
    build_test_path(path, sizeof(path), "test_mktd_trunc.bin");
    FILE *fp = fopen(path, "wb");
    /* write valid header but no data after it */
    unsigned int version = 2;
    unsigned int num_sym = 2, num_ts = 100, feat = 16, price_feat = 5;
    char padding[40];
    memset(padding, 0, 40);
    fwrite("MKTD", 1, 4, fp);
    fwrite(&version, 4, 1, fp);
    fwrite(&num_sym, 4, 1, fp);
    fwrite(&num_ts, 4, 1, fp);
    fwrite(&feat, 4, 1, fp);
    fwrite(&price_feat, 4, 1, fp);
    fwrite(padding, 1, 40, fp);
    fclose(fp);

    MktdData data;
    int rc = mktd_load(path, &data);
    ASSERT_EQ_INT(rc, -1, "truncated: should fail");
}

static void test_free_zeroed(void) {
    MktdData data;
    memset(&data, 0, sizeof(data));
    mktd_free(&data);
    g_pass++;
}

static void test_single_symbol(void) {
    char path[512];
    build_test_path(path, sizeof(path), "test_mktd_1sym.bin");
    ASSERT_EQ_INT(write_test_mktd(path, 1, 5, 16), 0, "1sym: fixture created");

    MktdData data;
    int rc = mktd_load(path, &data);
    ASSERT_EQ_INT(rc, 0, "1sym: load ok");
    if (rc != 0) return;
    ASSERT_EQ_INT(data.num_symbols, 1, "1sym: num_symbols");
    ASSERT_EQ_INT(data.num_timesteps, 5, "1sym: num_timesteps");

    /* sym0 t=4: base=100+0+4=104, close=105 */
    ASSERT_NEAR(data.closes[4], 105.0f, 0.01, "1sym: last close");

    mktd_free(&data);
}

static void test_equity_curve_from_mktd(void) {
    char path[512];
    build_test_path(path, sizeof(path), "test_mktd_eq.bin");
    ASSERT_EQ_INT(write_test_mktd(path, 2, 20, 16), 0, "eq: fixture created");

    MktdData data;
    int rc = mktd_load(path, &data);
    ASSERT_EQ_INT(rc, 0, "eq: load ok");
    if (rc != 0) return;

    int T = data.num_timesteps;
    int S = data.num_symbols;
    double *eq = (double *)malloc((size_t)(T + 1) * sizeof(double));
    double cash = 10000.0;
    eq[0] = cash;

    double alloc = cash / S;
    double *shares = (double *)calloc((size_t)S, sizeof(double));
    double rem = cash;
    for (int s = 0; s < S; s++) {
        float c0 = data.closes[0 * S + s];
        if (c0 > 0.0f) {
            shares[s] = alloc / (double)c0;
            rem -= shares[s] * (double)c0;
        }
    }
    for (int t = 0; t < T; t++) {
        double e = rem;
        for (int s = 0; s < S; s++) e += shares[s] * (double)data.closes[t * S + s];
        eq[t + 1] = e;
    }

    double sortino = compute_sortino(eq, T + 1);
    double max_dd = compute_max_drawdown(eq, T + 1);

    /* prices monotonically increase in test data, sortino should be positive */
    ASSERT_TRUE(sortino > 0.0, "eq: positive sortino for rising prices");
    ASSERT_NEAR(max_dd, 0.0, 0.001, "eq: no drawdown for rising prices");

    free(eq);
    free(shares);
    mktd_free(&data);
}

static void test_20_features(void) {
    char path[512];
    build_test_path(path, sizeof(path), "test_mktd_f20.bin");
    ASSERT_EQ_INT(write_test_mktd(path, 2, 5, 20), 0, "f20: fixture created");

    MktdData data;
    int rc = mktd_load(path, &data);
    ASSERT_EQ_INT(rc, 0, "f20: load ok");
    if (rc != 0) return;
    ASSERT_EQ_INT(data.features_per_sym, 20, "f20: features=20");
    ASSERT_EQ_INT(data.num_symbols, 2, "f20: num_symbols");
    ASSERT_EQ_INT(data.num_timesteps, 5, "f20: num_timesteps");

    mktd_free(&data);
}

static void test_load_null_args(void) {
    MktdData data;
    ASSERT_EQ_INT(mktd_load(NULL, &data), -1, "null_args: null path");
    ASSERT_EQ_INT(mktd_load("/dev/null", NULL), -1, "null_args: null data");
}

int main(void) {
    fprintf(stderr, "=== mktd_reader tests ===\n");

    test_load_basic();
    test_load_bad_magic();
    test_load_missing_file();
    test_load_truncated();
    test_load_null_args();
    test_free_zeroed();
    test_single_symbol();
    test_equity_curve_from_mktd();
    test_20_features();

    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
