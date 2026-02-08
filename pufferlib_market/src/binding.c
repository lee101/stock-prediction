/*
 * binding.c  –  PufferLib Python binding for the C trading environment.
 *
 * MarketData is loaded once per process via shared() and reused by all envs.
 */

#include "../include/trading_env.h"

/* Forward declarations required before env_binding.h */
static MarketData* g_shared_data = NULL;

/* Tell env_binding.h we provide our own my_shared */
#define MY_SHARED

#define Env TradingEnv
#include "../../external/pufferlib-3.0.0/pufferlib/ocean/env_binding.h"

/* ---------- shared market data (one per worker process) ---------- */

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (kwargs == NULL) {
        PyErr_SetString(PyExc_TypeError, "shared() requires data_path kwarg");
        return NULL;
    }
    PyObject* path_obj = PyDict_GetItemString(kwargs, "data_path");
    if (!path_obj || !PyUnicode_Check(path_obj)) {
        PyErr_SetString(PyExc_TypeError, "shared() requires data_path=str");
        return NULL;
    }
    const char* path = PyUnicode_AsUTF8(path_obj);

    if (g_shared_data) {
        Py_RETURN_NONE;
    }

    g_shared_data = market_data_load(path);
    if (!g_shared_data) {
        PyErr_Format(PyExc_RuntimeError, "Failed to load market data from %s", path);
        return NULL;
    }
    Py_RETURN_NONE;
}

/* ---------- per-env init ---------- */

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (!g_shared_data) {
        PyErr_SetString(PyExc_RuntimeError,
            "MarketData not loaded. Call binding.shared(data_path=...) first.");
        return -1;
    }

    env->data = g_shared_data;

    PyObject* val;

    val = kwargs ? PyDict_GetItemString(kwargs, "max_steps") : NULL;
    env->max_steps = val ? (int)PyLong_AsLong(val) : 720;

    val = kwargs ? PyDict_GetItemString(kwargs, "fee_rate") : NULL;
    env->fee_rate = val ? (float)PyFloat_AsDouble(val) : 0.001f;

    val = kwargs ? PyDict_GetItemString(kwargs, "max_leverage") : NULL;
    env->max_leverage = val ? (float)PyFloat_AsDouble(val) : 1.0f;

    int S = g_shared_data->num_symbols;
    env->obs_size = S * FEATURES_PER_SYM + 5 + S;
    env->num_actions = 1 + 2 * S;

    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "total_return", log->total_return);
    assign_to_dict(dict, "sortino",      log->sortino);
    assign_to_dict(dict, "max_drawdown", log->max_drawdown);
    assign_to_dict(dict, "num_trades",   log->num_trades);
    assign_to_dict(dict, "win_rate",     log->win_rate);
    assign_to_dict(dict, "avg_hold_hours", log->avg_hold_hours);
    return 0;
}

static void my_free(Env* env) {
    (void)env;
}
