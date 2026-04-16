/*
 * binding.c  –  PufferLib Python binding for the C trading environment.
 *
 * MarketData is loaded once per process via shared() and reused by all envs.
 */

#include "../include/trading_env.h"

/* Forward declarations required before env_binding.h */
static MarketData* g_shared_data = NULL;

/* Tell env_binding.h we provide our own my_shared, my_put, my_get, and custom methods */
#define MY_SHARED
#define MY_PUT
#define MY_GET
#define Env TradingEnv

/* Include Python.h early so we can forward-declare custom methods before
   env_binding.h expands the method table that references them. */
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* my_vec_set_offsets(PyObject* self, PyObject* args);
static PyObject* my_vec_env_at(PyObject* self, PyObject* args);

#define MY_METHODS \
    {"vec_set_offsets", (PyCFunction)my_vec_set_offsets, METH_VARARGS, "Set forced_offset per env in a VecEnv"}, \
    {"vec_env_at", (PyCFunction)my_vec_env_at, METH_VARARGS, "Get env handle at index in a VecEnv"}

#include "env_binding.h"

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

    val = kwargs ? PyDict_GetItemString(kwargs, "fill_slippage_bps") : NULL;
    env->fill_slippage_bps = val ? (float)PyFloat_AsDouble(val) : 0.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "fill_probability") : NULL;
    env->fill_probability = val ? (float)PyFloat_AsDouble(val) : 1.0f;
    if (env->fill_probability < 0.0f) env->fill_probability = 0.0f;
    if (env->fill_probability > 1.0f) env->fill_probability = 1.0f;

    val = kwargs ? PyDict_GetItemString(kwargs, "decision_lag") : NULL;
    /* Default 2 mirrors production. Pass decision_lag=1 explicitly for legacy
     * lookahead-tolerant research paths; warn loudly when that happens so a
     * silent default never re-enters production. */
    env->decision_lag = val ? (int)PyLong_AsLong(val) : 2;
    if (env->decision_lag < 1) env->decision_lag = 1;
    if (env->decision_lag < 2) {
        PyErr_WarnEx(PyExc_UserWarning,
                     "trading_env: decision_lag<2 has lookahead bias; production must use >=2",
                     1);
    }

    val = kwargs ? PyDict_GetItemString(kwargs, "forced_offset") : NULL;
    if (val) {
        int offset = (int)PyLong_AsLong(val);
        env->forced_offset = (offset < -1) ? -1 : offset;
    } else {
        env->forced_offset = -1;
    }

    val = kwargs ? PyDict_GetItemString(kwargs, "max_hold_hours") : NULL;
    env->max_hold_hours = val ? (int)PyLong_AsLong(val) : 0;

    val = kwargs ? PyDict_GetItemString(kwargs, "enable_drawdown_profit_early_exit") : NULL;
    env->enable_drawdown_profit_early_exit = val ? PyObject_IsTrue(val) : 0;
    if (env->enable_drawdown_profit_early_exit < 0) {
        return -1;
    }

    val = kwargs ? PyDict_GetItemString(kwargs, "drawdown_profit_early_exit_verbose") : NULL;
    env->drawdown_profit_early_exit_verbose = val ? PyObject_IsTrue(val) : 0;
    if (env->drawdown_profit_early_exit_verbose < 0) {
        return -1;
    }

    val = kwargs ? PyDict_GetItemString(kwargs, "drawdown_profit_early_exit_min_steps") : NULL;
    env->drawdown_profit_early_exit_min_steps = val ? (int)PyLong_AsLong(val) : 20;
    if (env->drawdown_profit_early_exit_min_steps < 0) {
        env->drawdown_profit_early_exit_min_steps = 0;
    }

    val = kwargs ? PyDict_GetItemString(kwargs, "drawdown_profit_early_exit_progress_fraction") : NULL;
    env->drawdown_profit_early_exit_progress_fraction = val ? (float)PyFloat_AsDouble(val) : 0.5f;
    if (env->drawdown_profit_early_exit_progress_fraction < 0.0f) {
        env->drawdown_profit_early_exit_progress_fraction = 0.0f;
    } else if (env->drawdown_profit_early_exit_progress_fraction > 1.0f) {
        env->drawdown_profit_early_exit_progress_fraction = 1.0f;
    }

    int S = g_shared_data->num_symbols;
    int side_block = S * env->action_allocation_bins * env->action_level_bins;
    env->obs_size = S * g_shared_data->features_per_sym + 5 + S;
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

static PyObject* my_get(PyObject* dict, Env* env) {
    /* Return per-env log data (same fields as my_log) */
    assign_to_dict(dict, "total_return", env->log.total_return);
    assign_to_dict(dict, "sortino",      env->log.sortino);
    assign_to_dict(dict, "max_drawdown", env->log.max_drawdown);
    assign_to_dict(dict, "num_trades",   env->log.num_trades);
    assign_to_dict(dict, "win_rate",     env->log.win_rate);
    assign_to_dict(dict, "avg_hold_hours", env->log.avg_hold_hours);
    assign_to_dict(dict, "n",           env->log.n);
    return dict;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    if (!kwargs) return 0;
    PyObject* val;

    val = PyDict_GetItemString(kwargs, "forced_offset");
    if (val) {
        int offset = (int)PyLong_AsLong(val);
        env->forced_offset = (offset < -1) ? -1 : offset;
    }

    return 0;
}

static void my_free(Env* env) {
    (void)env;
}

/* ---------- custom methods for fast eval ---------- */

static PyObject* my_vec_set_offsets(PyObject* self, PyObject* args) {
    /* vec_set_offsets(vec_handle, offsets_array) */
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_set_offsets requires 2 arguments");
        return NULL;
    }
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* arr_obj = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(arr_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "offsets must be a numpy array");
        return NULL;
    }
    PyArrayObject* offsets = (PyArrayObject*)arr_obj;
    if (PyArray_NDIM(offsets) != 1 || PyArray_SIZE(offsets) != vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "offsets length must match num_envs");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        long val = 0;
        if (PyArray_TYPE(offsets) == NPY_INT32) {
            val = ((int32_t*)PyArray_DATA(offsets))[i];
        } else if (PyArray_TYPE(offsets) == NPY_INT64) {
            val = ((int64_t*)PyArray_DATA(offsets))[i];
        } else {
            /* generic fallback */
            PyObject* item = PyArray_GETITEM(offsets, (char*)PyArray_DATA(offsets) + i * PyArray_STRIDE(offsets, 0));
            val = PyLong_AsLong(item);
            Py_DECREF(item);
        }
        vec->envs[i]->forced_offset = (val < -1) ? -1 : (int)val;
    }
    Py_RETURN_NONE;
}

static PyObject* my_vec_env_at(PyObject* self, PyObject* args) {
    /* vec_env_at(vec_handle, index) -> env_handle */
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_env_at requires 2 arguments");
        return NULL;
    }
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* idx_obj = PyTuple_GetItem(args, 1);
    int idx = (int)PyLong_AsLong(idx_obj);
    if (idx < 0 || idx >= vec->num_envs) {
        PyErr_SetString(PyExc_IndexError, "env index out of range");
        return NULL;
    }
    return PyLong_FromVoidPtr(vec->envs[idx]);
}
