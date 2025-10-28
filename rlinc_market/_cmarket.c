// Minimal ultra-fast C market environment with a CPython type
// Exposes: class MarketEnv(...): reset() -> obs(np.float32[obs_dim]),
// step(action: np.float32[n_assets]) -> (obs, reward, terminated, truncated)

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    PyObject_HEAD
    int n_assets;
    int window;
    int episode_len;
    int step;
    int steps_per_day;
    float intraday_leverage_max;
    float overnight_leverage_max;
    float finance_rate_daily;
    float trading_cost_bps; // unified trading fee (bps)
    float risk_aversion;
    float return_sigma;
    uint64_t rng;

    // buffers
    float* ret_hist;      // shape [window, n_assets]
    float* weights;       // [n_assets]
    float* last_weights;  // [n_assets]
    float* last_returns;  // [n_assets]
} MarketEnv;

static inline uint64_t xorshift64(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static inline float randu(MarketEnv* env) {
    // Uniform in (0,1)
    const uint64_t m = ((uint64_t)1 << 53) - 1;
    uint64_t r = xorshift64(&env->rng) & m;
    return (float)((r + 1.0) / (double)(m + 2.0));
}

static float randn(MarketEnv* env) {
    // Box-Muller
    float u1 = randu(env);
    float u2 = randu(env);
    float r = sqrtf(-2.0f * logf(u1 + 1e-12f));
    float z = r * cosf(2.0f * (float)M_PI * u2);
    return z;
}

static int clamp_l1(float* w, int n, float l1_max) {
    // If sum(|w|) > l1_max, scale down uniformly.
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += fabs((double)w[i]);
    if (s <= (double)l1_max || s == 0.0) return 0;
    float scale = (float)(l1_max / s);
    for (int i = 0; i < n; ++i) w[i] *= scale;
    return 1;
}

static void shift_ret_hist(MarketEnv* env) {
    // shift rows up by 1: [1:] -> [0:-1]
    const int n = env->n_assets;
    const int w = env->window;
    if (w <= 1) return;
    memmove(env->ret_hist, env->ret_hist + n, (size_t)(n * (w - 1)) * sizeof(float));
}

static PyObject* build_obs(MarketEnv* env) {
    // obs = flatten(ret_hist [w,n]) concat weights [n]
    const int n = env->n_assets;
    const int w = env->window;
    npy_intp obs_dim = (npy_intp)(n * (w + 1));
    PyObject* arr = PyArray_SimpleNew(1, &obs_dim, NPY_FLOAT32);
    if (!arr) return NULL;
    float* out = (float*)PyArray_DATA((PyArrayObject*)arr);
    memcpy(out, env->ret_hist, (size_t)(n * w) * sizeof(float));
    memcpy(out + n * w, env->weights, (size_t)n * sizeof(float));
    return arr;
}

static int MarketEnv_traverse(MarketEnv* self, visitproc visit, void* arg) {
    return 0;
}

static int MarketEnv_clear(MarketEnv* self) {
    return 0;
}

static void MarketEnv_dealloc(MarketEnv* self) {
    PyObject_GC_UnTrack(self);
    MarketEnv_clear(self);
    if (self->ret_hist) PyMem_Free(self->ret_hist);
    if (self->weights) PyMem_Free(self->weights);
    if (self->last_weights) PyMem_Free(self->last_weights);
    if (self->last_returns) PyMem_Free(self->last_returns);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* MarketEnv_new(PyTypeObject* type, PyObject* args, PyObject* kw) {
    MarketEnv* self = (MarketEnv*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->n_assets = 0;
    self->window = 0;
    self->episode_len = 0;
    self->step = 0;
    self->steps_per_day = 1;
    self->intraday_leverage_max = 4.0f;
    self->overnight_leverage_max = 2.0f;
    self->finance_rate_daily = 0.0675f / 252.0f;
    self->trading_cost_bps = 0.0f;
    self->risk_aversion = 0.0f;
    self->return_sigma = 0.01f;
    self->rng = 88172645463393265ULL; // default non-zero seed
    self->ret_hist = NULL;
    self->weights = NULL;
    self->last_weights = NULL;
    self->last_returns = NULL;
    return (PyObject*)self;
}

static int MarketEnv_init(MarketEnv* self, PyObject* args, PyObject* kw) {
    static char* kws[] = {
        "n_assets", "window", "episode_len", "seed",
        "leverage_limit", "trading_cost_bps", "risk_aversion",
        "steps_per_day", "finance_rate_annual", "trading_days_per_year",
        "intraday_leverage_max", "overnight_leverage_max", "trading_fee_bps",
        "return_sigma",
        NULL
    };
    int n_assets, window, episode_len;
    unsigned long long seed = 0ULL;
    float leverage_limit = 1.0f, trading_cost_bps = 0.0f, risk_aversion = 0.0f;
    int steps_per_day = 1;
    float finance_rate_annual = 0.0675f;
    int trading_days_per_year = 252;
    float intraday_leverage_max = 4.0f;
    float overnight_leverage_max = 2.0f;
    float trading_fee_bps = -1.0f; // alias for trading_cost_bps
    float return_sigma = 0.01f;

    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "iii|Kfffififfff", kws,
            &n_assets, &window, &episode_len, &seed,
            &leverage_limit, &trading_cost_bps, &risk_aversion,
            &steps_per_day, &finance_rate_annual, &trading_days_per_year,
            &intraday_leverage_max, &overnight_leverage_max, &trading_fee_bps,
            &return_sigma))
        return -1;

    if (n_assets <= 0 || window <= 0 || episode_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_assets, window, episode_len must be > 0");
        return -1;
    }

    self->n_assets = n_assets;
    self->window = window;
    self->episode_len = episode_len;
    self->steps_per_day = steps_per_day > 0 ? steps_per_day : 1;
    self->intraday_leverage_max = intraday_leverage_max > 0 ? intraday_leverage_max : 4.0f;
    self->overnight_leverage_max = overnight_leverage_max > 0 ? overnight_leverage_max : 2.0f;
    self->finance_rate_daily = finance_rate_annual / (float)(trading_days_per_year > 0 ? trading_days_per_year : 252);
    self->trading_cost_bps = (trading_fee_bps >= 0.0f) ? trading_fee_bps : trading_cost_bps;
    self->risk_aversion = risk_aversion;
    self->return_sigma = return_sigma;
    self->rng = seed ? seed : 88172645463393265ULL;

    size_t hist_sz = (size_t)n_assets * (size_t)window;
    self->ret_hist = (float*)PyMem_Calloc(hist_sz, sizeof(float));
    self->weights = (float*)PyMem_Calloc((size_t)n_assets, sizeof(float));
    self->last_weights = (float*)PyMem_Calloc((size_t)n_assets, sizeof(float));
    self->last_returns = (float*)PyMem_Calloc((size_t)n_assets, sizeof(float));
    if (!self->ret_hist || !self->weights || !self->last_weights || !self->last_returns) {
        PyErr_NoMemory();
        return -1;
    }
    self->step = 0;

    // Prefill return history with tiny noise
    for (int t = 0; t < window; ++t) {
        for (int i = 0; i < n_assets; ++i) {
            self->ret_hist[t * n_assets + i] = 0.0001f * randn(self);
        }
    }
    memset(self->weights, 0, (size_t)n_assets * sizeof(float));
    memset(self->last_weights, 0, (size_t)n_assets * sizeof(float));
    memset(self->last_returns, 0, (size_t)n_assets * sizeof(float));
    return 0;
}

static PyObject* MarketEnv_reset(MarketEnv* self, PyObject* Py_UNUSED(ignored)) {
    // Reset counters and refill history with fresh noise
    self->step = 0;
    for (int t = 0; t < self->window; ++t) {
        for (int i = 0; i < self->n_assets; ++i) {
            self->ret_hist[t * self->n_assets + i] = 0.0001f * randn(self);
        }
    }
    memset(self->weights, 0, (size_t)self->n_assets * sizeof(float));
    memset(self->last_weights, 0, (size_t)self->n_assets * sizeof(float));
    memset(self->last_returns, 0, (size_t)self->n_assets * sizeof(float));
    return build_obs(self);
}

static PyObject* MarketEnv_step(MarketEnv* self, PyObject* args) {
    PyObject* act_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &act_obj)) return NULL;

    PyArrayObject* act_arr = (PyArrayObject*)PyArray_FROM_OTF(act_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!act_arr) return NULL;
    if (PyArray_NDIM(act_arr) != 1 || PyArray_DIM(act_arr, 0) != self->n_assets) {
        Py_DECREF(act_arr);
        PyErr_SetString(PyExc_ValueError, "action must be shape [n_assets] float32");
        return NULL;
    }
    float* a = (float*)PyArray_DATA(act_arr);

    // Determine day boundaries
    int is_open = (self->steps_per_day > 0) ? ((self->step % self->steps_per_day) == 0) : 1;
    int is_close = (self->steps_per_day > 0) ? (((self->step + 1) % self->steps_per_day) == 0) : 1;

    // Financing at market open on overnight leverage above 1x
    double finance = 0.0;
    if (is_open) {
        double l1_prev = 0.0;
        for (int i = 0; i < self->n_assets; ++i) l1_prev += fabs((double)self->last_weights[i]);
        if (l1_prev > 1.0) finance = (double)self->finance_rate_daily * (l1_prev - 1.0);
    }

    // Copy action -> candidate weights, clamp to intraday max leverage
    memcpy(self->weights, a, (size_t)self->n_assets * sizeof(float));
    clamp_l1(self->weights, self->n_assets, self->intraday_leverage_max);

    // Generate next returns ~ Normal(0, sigma)
    for (int i = 0; i < self->n_assets; ++i) {
        float r = self->return_sigma * randn(self);
        self->last_returns[i] = r;
    }

    // Base trading cost for moving from last_weights -> intraday weights
    double l1_turnover = 0.0;
    double var = 0.0;
    for (int i = 0; i < self->n_assets; ++i) {
        l1_turnover += fabs((double)(self->weights[i] - self->last_weights[i]));
        var += (double)(self->weights[i] * self->weights[i]);
    }
    double trade_cost = (self->trading_cost_bps * 1e-4) * l1_turnover;

    // Intraday return using intraday weights
    double dot = 0.0;
    for (int i = 0; i < self->n_assets; ++i) {
        dot += (double)self->weights[i] * (double)self->last_returns[i];
    }

    // Close auto-deleverage to overnight max with fee on that rebalance
    double close_cost = 0.0;
    if (is_close) {
        double l1_now = 0.0;
        for (int i = 0; i < self->n_assets; ++i) l1_now += fabs((double)self->weights[i]);
        if (l1_now > (double)self->overnight_leverage_max && l1_now > 0.0) {
            float s = (float)(self->overnight_leverage_max / l1_now);
            // L1 turnover due to scaling: l1_now - overnight_max
            close_cost = (self->trading_cost_bps * 1e-4) * (l1_now - (double)self->overnight_leverage_max);
            for (int i = 0; i < self->n_assets; ++i) self->weights[i] *= s;
        }
    }

    // Risk penalty
    double risk = (double)self->risk_aversion * var * 0.5;

    float reward = (float)(dot - trade_cost - close_cost - finance - risk);

    // Update last_weights to final weights (after potential close scaling)
    memcpy(self->last_weights, self->weights, (size_t)self->n_assets * sizeof(float));

    // Update return history (shift and append new row)
    shift_ret_hist(self);
    memcpy(self->ret_hist + (self->window - 1) * self->n_assets,
           self->last_returns, (size_t)self->n_assets * sizeof(float));

    self->step += 1;
    int terminated = (self->step >= self->episode_len) ? 1 : 0;
    int truncated = 0;

    PyObject* obs = build_obs(self);
    if (!obs) { Py_DECREF(act_arr); return NULL; }
    PyObject* py_reward = PyFloat_FromDouble((double)reward);
    PyObject* py_term = PyBool_FromLong(terminated);
    PyObject* py_trunc = PyBool_FromLong(truncated);

    PyObject* tup = PyTuple_New(4);
    // (obs, reward, terminated, truncated)
    PyTuple_SET_ITEM(tup, 0, obs);
    PyTuple_SET_ITEM(tup, 1, py_reward);
    PyTuple_SET_ITEM(tup, 2, py_term);
    PyTuple_SET_ITEM(tup, 3, py_trunc);

    Py_DECREF(act_arr);
    return tup;
}

static PyObject* MarketEnv_state(MarketEnv* self, PyObject* Py_UNUSED(ignored)) {
    PyObject* d = PyDict_New();
    if (!d) return NULL;
    npy_intp n = (npy_intp)self->n_assets;
    PyObject* w = PyArray_SimpleNew(1, &n, NPY_FLOAT32);
    if (!w) { Py_DECREF(d); return NULL; }
    memcpy(PyArray_DATA((PyArrayObject*)w), self->weights, (size_t)self->n_assets * sizeof(float));
    double l1 = 0.0; for (int i = 0; i < self->n_assets; ++i) l1 += fabs((double)self->weights[i]);
    PyObject* step = PyLong_FromLong(self->step);
    PyObject* l1v = PyFloat_FromDouble(l1);
    if (!step || !l1v) { Py_XDECREF(step); Py_XDECREF(l1v); Py_DECREF(w); Py_DECREF(d); return NULL; }
    PyDict_SetItemString(d, "weights", w);
    PyDict_SetItemString(d, "step", step);
    PyDict_SetItemString(d, "l1", l1v);
    Py_DECREF(w); Py_DECREF(step); Py_DECREF(l1v);
    return d;
}

static PyMethodDef MarketEnv_methods[] = {
    {"reset", (PyCFunction)MarketEnv_reset, METH_NOARGS, "Reset the environment and return observation."},
    {"step", (PyCFunction)MarketEnv_step, METH_VARARGS, "Step with action vector -> (obs, reward, terminated, truncated)."},
    {"state", (PyCFunction)MarketEnv_state, METH_NOARGS, "Return {'weights': np.ndarray, 'step': int, 'l1': float}"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject MarketEnvType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "rlinc_cmarket.MarketEnv",
    .tp_basicsize = sizeof(MarketEnv),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)MarketEnv_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)MarketEnv_traverse,
    .tp_clear = (inquiry)MarketEnv_clear,
    .tp_doc = "C fast market environment",
    .tp_methods = MarketEnv_methods,
    .tp_new = MarketEnv_new,
    .tp_init = (initproc)MarketEnv_init,
};

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "rlinc_cmarket",
    .m_doc = "C-backed market simulator for RL",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit_rlinc_cmarket(void) {
    import_array();
    if (PyType_Ready(&MarketEnvType) < 0) return NULL;
    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&MarketEnvType);
    if (PyModule_AddObject(m, "MarketEnv", (PyObject*)&MarketEnvType) < 0) {
        Py_DECREF(&MarketEnvType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
