#include "csim/trading_sim.h"
#define Env TradingSim
#define MY_SHARED
#include "env_binding.h"

static float* g_close = NULL;
static float* g_high = NULL;
static float* g_low = NULL;
static float* g_features = NULL;

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* close_obj = PyDict_GetItemString(kwargs, "close");
    PyObject* high_obj = PyDict_GetItemString(kwargs, "high");
    PyObject* low_obj = PyDict_GetItemString(kwargs, "low");
    PyObject* features_obj = PyDict_GetItemString(kwargs, "features");
    if (!close_obj || !high_obj || !low_obj || !features_obj) {
        PyErr_SetString(PyExc_TypeError, "Missing market data arrays");
        return NULL;
    }
    g_close = (float*)PyArray_DATA((PyArrayObject*)close_obj);
    g_high = (float*)PyArray_DATA((PyArrayObject*)high_obj);
    g_low = (float*)PyArray_DATA((PyArrayObject*)low_obj);
    g_features = (float*)PyArray_DATA((PyArrayObject*)features_obj);
    Py_RETURN_NONE;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->n_symbols = (int)unpack(kwargs, "n_symbols");
    env->n_bars = (int)unpack(kwargs, "n_bars");
    env->n_features = (int)unpack(kwargs, "n_features");
    env->initial_cash = (float)unpack(kwargs, "initial_cash");
    env->fee_rate = (float)unpack(kwargs, "fee_rate");
    env->max_hold_bars = (int)unpack(kwargs, "max_hold_bars");
    env->episode_length = (int)unpack(kwargs, "episode_length");
    env->data_start = (int)unpack(kwargs, "data_start");
    env->data_end = (int)unpack(kwargs, "data_end");
    env->obs_dim = env->n_symbols * env->n_features + env->n_symbols + 3;
    env->close = g_close;
    env->high = g_high;
    env->low = g_low;
    env->features = g_features;
    init(env);
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_pnl_pct", log->episode_pnl_pct);
    assign_to_dict(dict, "max_drawdown", log->max_drawdown);
    assign_to_dict(dict, "n_trades", log->n_trades);
    assign_to_dict(dict, "win_rate", log->win_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
