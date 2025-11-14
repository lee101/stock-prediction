# Profile Analysis: trade_stock_e2e_paper.prof

## Summary

Total functions profiled: 27874

Total execution time: 2047.472 seconds

## Top 50 Time-Consuming Functions (Hot Paths)

| Rank | Function | Location | Cumulative | Own Time | Calls | % Total |
|------|----------|----------|------------|----------|-------|---------|
| 1 | `predict` | kronos.py:549 | 170.308s | 0.014s | 114 | 8.3% |
| 2 | `generate` | kronos.py:538 | 162.843s | 0.006s | 114 | 8.0% |
| 3 | `auto_regressive_inference` | kronos.py:455 | 153.404s | 0.007s | 114 | 7.5% |
| 4 | `decode_s1` | kronos.py:344 | 109.202s | 0.138s | 798 | 5.3% |
| 5 | `forward` | module.py:493 | 109.112s | 1.707s | 10,257 | 5.3% |
| 6 | `__exit__` | grad_mode.py:287 | 95.160s | 0.092s | 483 | 4.6% |
| 7 | `_wrapped_call_impl` | module.py:1771 | 77.098s | 0.398s | 174,483 | 3.8% |
| 8 | `_call_impl` | module.py:1779 | 77.070s | 0.654s | 174,489 | 3.8% |
| 9 | `forward` | module.py:348 | 56.002s | 0.409s | 10,257 | 2.7% |
| 10 | `<built-in method torch._C._nn.linear>` | ~:0 | 37.083s | 36.806s | 77,850 | 1.8% |
| 11 | `forward` | linear.py:130 | 36.977s | 0.118s | 77,829 | 1.8% |
| 12 | `<method 'cpu' of 'torch._C.TensorBase...` | ~:0 | 33.730s | 33.730s | 5,081 | 1.6% |
| 13 | `forward` | module.py:277 | 27.626s | 3.204s | 10,256 | 1.3% |
| 14 | `scaled_dot_product_attention` | module.py:307 | 27.316s | 21.437s | 11,053 | 1.3% |
| 15 | `forward` | module.py:263 | 25.447s | 2.210s | 22,107 | 1.2% |
| 16 | `<module>` | __init__.py:1 | 21.719s | 0.020s | 618 | 1.1% |
| 17 | `forward` | module.py:295 | 18.910s | 6.825s | 11,054 | 0.9% |
| 18 | `_norm` | module.py:260 | 15.717s | 11.276s | 22,107 | 0.8% |
| 19 | `write` | record_writer.py:36 | 15.520s | 0.057s | 8,601 | 0.8% |
| 20 | `_compute_toto_forecast` | backtest_test3_inline.py:1768 | 13.713s | 0.184s | 453 | 0.7% |
| 21 | `<method 'item' of 'torch._C.TensorBas...` | ~:0 | 12.049s | 12.049s | 9,297 | 0.6% |
| 22 | `write` | gfile.py:751 | 11.851s | 0.053s | 8,607 | 0.6% |
| 23 | `evaluate_maxdiff_strategy` | backtest_test3_inline.py:459 | 11.187s | 0.031s | 113 | 0.5% |
| 24 | `decode_s2` | kronos.py:376 | 9.123s | 0.008s | 797 | 0.4% |
| 25 | `forward` | module.py:472 | 8.555s | 0.075s | 797 | 0.4% |
| 26 | `evaluate_pctdiff_strategy` | backtest_test3_inline.py:862 | 7.371s | 0.027s | 113 | 0.4% |
| 27 | `forward` | module.py:390 | 7.254s | 0.030s | 797 | 0.4% |
| 28 | `evaluate_maxdiff_always_on_strategy` | backtest_test3_inline.py:670 | 7.220s | 0.020s | 133 | 0.4% |
| 29 | `_update_cos_sin_cache` | module.py:287 | 6.898s | 0.363s | 11,054 | 0.3% |
| 30 | `_ensure_predictor` | kronos_wrapper.py:586 | 6.222s | 0.000s | 114 | 0.3% |
| 31 | `wrapper` | disk_cache.py:14 | 5.988s | 0.084s | 3,171 | 0.3% |
| 32 | `_multi_stage_grid_search` | optimization_utils_fast.py:449 | 5.840s | 0.066s | 472 | 0.3% |
| 33 | `<built-in method torch.cat>` | ~:0 | 5.793s | 5.779s | 37,017 | 0.3% |
| 34 | `make_prim` | inductor_prims.py:20 | 5.532s | 0.000s | 10 | 0.3% |
| 35 | `_rotate_half` | module.py:302 | 5.187s | 1.540s | 22,108 | 0.3% |
| 36 | `optimize_maxdiff_always_on` | maxdiff_optimizer.py:137 | 5.116s | 0.024s | 133 | 0.2% |
| 37 | `<method 'to' of 'torch._C.TensorBase'...` | ~:0 | 5.012s | 3.995s | 29,735 | 0.2% |
| 38 | `_inner_fn` | _validators.py:98 | 4.954s | 0.001s | 61 | 0.2% |
| 39 | `from_pretrained` | hub_mixin.py:462 | 4.954s | 0.000s | 5 | 0.2% |
| 40 | `run_bounded_optimizer` | optimization_utils.py:471 | 4.902s | 0.002s | 107 | 0.2% |
| 41 | `direct` | _direct_py.py:36 | 4.900s | 0.004s | 107 | 0.2% |
| 42 | `<built-in method scipy.optimize._dire...` | ~:0 | 4.880s | 0.125s | 107 | 0.2% |
| 43 | `_func_wrap` | _direct_py.py:247 | 4.755s | 0.124s | 74,711 | 0.2% |
| 44 | `<method 'type_as' of 'torch._C.Tensor...` | ~:0 | 4.744s | 4.744s | 33,161 | 0.2% |
| 45 | `decode` | kronos.py:227 | 4.724s | 0.003s | 113 | 0.2% |
| 46 | `optimize_maxdiff_entry_exit` | maxdiff_optimizer.py:50 | 4.718s | 0.023s | 113 | 0.2% |
| 47 | `<module>` | trade_stock_e2e.py:1 | 4.648s | 0.000s | 1 | 0.2% |
| 48 | `main` | trade_stock_e2e.py:4169 | 4.648s | 0.000s | 1 | 0.2% |
| 49 | `<built-in method time.sleep>` | ~:0 | 4.648s | 4.648s | 1 | 0.2% |
| 50 | `_objective` | backtest_test3_inline.py:1023 | 4.482s | 0.324s | 74,711 | 0.2% |

## Performance Hotspots (>1.0% of total time)

### `predict` (kronos.py:549)
████████ **8.32%**
- **Cumulative time**: 170.308s
- **Own time**: 0.014s
- **Calls**: 114
- **Time per call**: 1.493933s

### `generate` (kronos.py:538)
███████ **7.95%**
- **Cumulative time**: 162.843s
- **Own time**: 0.006s
- **Calls**: 114
- **Time per call**: 1.428445s

### `auto_regressive_inference` (kronos.py:455)
███████ **7.49%**
- **Cumulative time**: 153.404s
- **Own time**: 0.007s
- **Calls**: 114
- **Time per call**: 1.345647s

### `decode_s1` (kronos.py:344)
█████ **5.33%**
- **Cumulative time**: 109.202s
- **Own time**: 0.138s
- **Calls**: 798
- **Time per call**: 0.136844s

### `forward` (module.py:493)
█████ **5.33%**
- **Cumulative time**: 109.112s
- **Own time**: 1.707s
- **Calls**: 10,257
- **Time per call**: 0.010638s

### `__exit__` (grad_mode.py:287)
████ **4.65%**
- **Cumulative time**: 95.160s
- **Own time**: 0.092s
- **Calls**: 483
- **Time per call**: 0.197018s

### `_wrapped_call_impl` (module.py:1771)
███ **3.77%**
- **Cumulative time**: 77.098s
- **Own time**: 0.398s
- **Calls**: 174,483
- **Time per call**: 0.000442s

### `_call_impl` (module.py:1779)
███ **3.76%**
- **Cumulative time**: 77.070s
- **Own time**: 0.654s
- **Calls**: 174,489
- **Time per call**: 0.000442s

### `forward` (module.py:348)
██ **2.74%**
- **Cumulative time**: 56.002s
- **Own time**: 0.409s
- **Calls**: 10,257
- **Time per call**: 0.005460s

### `<built-in method torch._C._nn.linear>` (~:0)
█ **1.81%**
- **Cumulative time**: 37.083s
- **Own time**: 36.806s
- **Calls**: 77,850
- **Time per call**: 0.000476s

### `forward` (linear.py:130)
█ **1.81%**
- **Cumulative time**: 36.977s
- **Own time**: 0.118s
- **Calls**: 77,829
- **Time per call**: 0.000475s

### `<method 'cpu' of 'torch._C.TensorBase' objects>` (~:0)
█ **1.65%**
- **Cumulative time**: 33.730s
- **Own time**: 33.730s
- **Calls**: 5,081
- **Time per call**: 0.006638s

### `forward` (module.py:277)
█ **1.35%**
- **Cumulative time**: 27.626s
- **Own time**: 3.204s
- **Calls**: 10,256
- **Time per call**: 0.002694s

### `scaled_dot_product_attention` (module.py:307)
█ **1.33%**
- **Cumulative time**: 27.316s
- **Own time**: 21.437s
- **Calls**: 11,053
- **Time per call**: 0.002471s

### `forward` (module.py:263)
█ **1.24%**
- **Cumulative time**: 25.447s
- **Own time**: 2.210s
- **Calls**: 22,107
- **Time per call**: 0.001151s

### `<module>` (__init__.py:1)
█ **1.06%**
- **Cumulative time**: 21.719s
- **Own time**: 0.020s
- **Calls**: 618
- **Time per call**: 0.035145s


## Optimization Recommendations

Based on the profile data, consider optimizing:

1. **predict** (kronos.py:549) - 8.3% of total time
2. **generate** (kronos.py:538) - 8.0% of total time
3. **auto_regressive_inference** (kronos.py:455) - 7.5% of total time
4. **decode_s1** (kronos.py:344) - 5.3% of total time
5. **forward** (module.py:493) - 5.3% of total time