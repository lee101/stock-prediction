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
#include "../../PufferLib/pufferlib/ocean/env_binding.h"

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

    val = kwargs ? PyDict_GetItemString(kwargs, "short_borrow_apr") : NULL;
    env->short_borrow_apr = val ? (float)PyFloat_AsDouble(val) : 0.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "periods_per_year") : NULL;
    env->periods_per_year = val ? (float)PyFloat_AsDouble(val) : 8760.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "action_allocation_bins") : NULL;
    env->action_allocation_bins = val ? (int)PyLong_AsLong(val) : 1;
    if (env->action_allocation_bins < 1) env->action_allocation_bins = 1;

    val = kwargs ? PyDict_GetItemString(kwargs, "action_level_bins") : NULL;
    env->action_level_bins = val ? (int)PyLong_AsLong(val) : 1;
    if (env->action_level_bins < 1) env->action_level_bins = 1;

    val = kwargs ? PyDict_GetItemString(kwargs, "action_max_offset_bps") : NULL;
    env->action_max_offset_bps = val ? (float)PyFloat_AsDouble(val) : 0.0f;
    if (env->action_max_offset_bps < 0.0f) env->action_max_offset_bps = 0.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "reward_scale") : NULL;
    env->reward_scale = val ? (float)PyFloat_AsDouble(val) : 10.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "reward_clip") : NULL;
    env->reward_clip = val ? (float)PyFloat_AsDouble(val) : 5.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "cash_penalty") : NULL;
    env->cash_penalty = val ? (float)PyFloat_AsDouble(val) : 0.01f;

    val = kwargs ? PyDict_GetItemString(kwargs, "drawdown_penalty") : NULL;
    env->drawdown_penalty = val ? (float)PyFloat_AsDouble(val) : 0.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "downside_penalty") : NULL;
    env->downside_penalty = val ? (float)PyFloat_AsDouble(val) : 0.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "smooth_downside_penalty") : NULL;
    env->smooth_downside_penalty = val ? (float)PyFloat_AsDouble(val) : 0.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "smooth_downside_temperature") : NULL;
    env->smooth_downside_temperature = val ? (float)PyFloat_AsDouble(val) : 0.02f;

    val = kwargs ? PyDict_GetItemString(kwargs, "trade_penalty") : NULL;
    env->trade_penalty = val ? (float)PyFloat_AsDouble(val) : 0.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "smoothness_penalty") : NULL;
    env->smoothness_penalty = val ? (float)PyFloat_AsDouble(val) : 0.0f;

    int S = g_shared_data->num_symbols;
    int side_block = S * env->action_allocation_bins * env->action_level_bins;
    env->obs_size = S * FEATURES_PER_SYM + 5 + S;
    env->num_actions = 1 + 2 * side_block;

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
