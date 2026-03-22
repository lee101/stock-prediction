extern "C" {
#include "../policy_infer.h"
}

#include <torch/script.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>

static int g_pass = 0, g_fail = 0;

#define ASSERT_EQ_INT(a, b, msg) do { \
    int _a = (a), _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL %s: %d != %d\n", msg, _a, _b); \
        g_fail++; \
    } else { g_pass++; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    double _a = (a), _b = (b), _t = (tol); \
    if (fabs(_a - _b) > _t) { \
        fprintf(stderr, "FAIL %s: %.10f != %.10f (tol=%.10f)\n", msg, _a, _b, _t); \
        g_fail++; \
    } else { g_pass++; } \
} while(0)

static std::string create_test_model(int in_dim, int out_dim) {
    torch::jit::Module mod("TestModel");

    std::ostringstream src;
    src << "def forward(self, x: Tensor) -> Tensor:\n"
        << "    return torch.mm(x, torch.ones(" << in_dim << ", " << out_dim << ")) + 0.5\n";
    mod.define(src.str());

    std::string path = "/tmp/test_policy_model.pt";
    mod.save(path);
    return path;
}

static void test_load_unload() {
    std::string path = create_test_model(8, 4);

    Policy p;
    int rc = policy_load(&p, path.c_str());
    ASSERT_EQ_INT(rc, 0, "load: returns 0");
    ASSERT_EQ_INT(p.loaded, 1, "load: loaded flag");

    policy_unload(&p);
    ASSERT_EQ_INT(p.loaded, 0, "unload: loaded flag");
    if (p.model != NULL) {
        fprintf(stderr, "FAIL unload: model not null\n");
        g_fail++;
    } else {
        g_pass++;
    }
}

static void test_load_bad_path() {
    Policy p;
    int rc = policy_load(&p, "/tmp/nonexistent_model_xyz.pt");
    ASSERT_EQ_INT(rc, -1, "bad_path: returns -1");
    ASSERT_EQ_INT(p.loaded, 0, "bad_path: not loaded");
}

static void test_forward_not_loaded() {
    Policy p;
    memset(&p, 0, sizeof(p));
    double obs[4] = {1.0, 2.0, 3.0, 4.0};
    PolicyAction actions[1];
    int rc = policy_forward(&p, obs, 4, actions, 1);
    ASSERT_EQ_INT(rc, -1, "not_loaded: returns -1");
}

static void test_forward_basic() {
    int in_dim = 8, out_dim = 4;
    std::string path = create_test_model(in_dim, out_dim);

    Policy p;
    int rc = policy_load(&p, path.c_str());
    ASSERT_EQ_INT(rc, 0, "forward_basic: load ok");

    double obs[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    PolicyAction actions[1];
    memset(actions, 0, sizeof(actions));

    rc = policy_forward(&p, obs, in_dim, actions, 1);
    ASSERT_EQ_INT(rc, 0, "forward_basic: returns 0");

    // model is x @ ones(8,4) + 0.5, sum of obs = 36, so each output = 36.5
    ASSERT_NEAR(actions[0].buy_price, 36.5, 0.01, "forward_basic: buy_price");
    ASSERT_NEAR(actions[0].sell_price, 36.5, 0.01, "forward_basic: sell_price");
    ASSERT_NEAR(actions[0].buy_amount, 36.5, 0.01, "forward_basic: buy_amount");
    ASSERT_NEAR(actions[0].sell_amount, 36.5, 0.01, "forward_basic: sell_amount");

    policy_unload(&p);
}

static void test_forward_multi_symbol() {
    int in_dim = 6, n_symbols = 3, out_dim = n_symbols * 4;
    std::string path = create_test_model(in_dim, out_dim);

    Policy p;
    int rc = policy_load(&p, path.c_str());
    ASSERT_EQ_INT(rc, 0, "multi_sym: load ok");

    double obs[6] = {1, 2, 3, 4, 5, 6};
    PolicyAction actions[3];
    memset(actions, 0, sizeof(actions));

    rc = policy_forward(&p, obs, in_dim, actions, n_symbols);
    ASSERT_EQ_INT(rc, 0, "multi_sym: returns 0");

    // sum = 21, each output = 21.5
    for (int i = 0; i < n_symbols; i++) {
        ASSERT_NEAR(actions[i].buy_price, 21.5, 0.01, "multi_sym: output value");
    }

    policy_unload(&p);
}

static void test_forward_deterministic() {
    int in_dim = 4, out_dim = 4;
    std::string path = create_test_model(in_dim, out_dim);

    Policy p;
    policy_load(&p, path.c_str());

    double obs[4] = {1.0, 2.0, 3.0, 4.0};
    PolicyAction a1, a2;

    policy_forward(&p, obs, in_dim, &a1, 1);
    policy_forward(&p, obs, in_dim, &a2, 1);

    ASSERT_NEAR(a1.buy_price, a2.buy_price, 1e-6, "deterministic: buy_price");
    ASSERT_NEAR(a1.sell_price, a2.sell_price, 1e-6, "deterministic: sell_price");
    ASSERT_NEAR(a1.buy_amount, a2.buy_amount, 1e-6, "deterministic: buy_amount");
    ASSERT_NEAR(a1.sell_amount, a2.sell_amount, 1e-6, "deterministic: sell_amount");

    policy_unload(&p);
}

static void test_load_reload() {
    std::string path = create_test_model(4, 4);

    Policy p;
    policy_load(&p, path.c_str());
    ASSERT_EQ_INT(p.loaded, 1, "reload: first load");

    policy_unload(&p);
    ASSERT_EQ_INT(p.loaded, 0, "reload: after unload");

    int rc = policy_load(&p, path.c_str());
    ASSERT_EQ_INT(rc, 0, "reload: second load");
    ASSERT_EQ_INT(p.loaded, 1, "reload: loaded after second load");

    policy_unload(&p);
}

static void test_output_fewer_than_requested() {
    int in_dim = 4, out_dim = 4;
    std::string path = create_test_model(in_dim, out_dim);

    Policy p;
    policy_load(&p, path.c_str());

    double obs[4] = {1.0, 2.0, 3.0, 4.0};
    PolicyAction actions[3];
    memset(actions, 0xff, sizeof(actions));

    int rc = policy_forward(&p, obs, in_dim, actions, 3);
    ASSERT_EQ_INT(rc, 0, "fewer_out: returns 0");

    ASSERT_NEAR(actions[1].buy_price, 0.0, 1e-10, "fewer_out: sym1 zeroed");
    ASSERT_NEAR(actions[2].buy_price, 0.0, 1e-10, "fewer_out: sym2 zeroed");

    policy_unload(&p);
}

static void test_export_returns_error() {
    int rc = policy_export_torchscript("foo", "bar");
    ASSERT_EQ_INT(rc, -1, "export: returns -1");
}

int main() {
    fprintf(stderr, "=== policy_infer tests (libtorch) ===\n");

    test_load_unload();
    test_load_bad_path();
    test_forward_not_loaded();
    test_forward_basic();
    test_forward_multi_symbol();
    test_forward_deterministic();
    test_load_reload();
    test_output_fewer_than_requested();
    test_export_returns_error();

    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
