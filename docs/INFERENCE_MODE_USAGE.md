# Inference Mode Usage

All model wrappers use `torch.inference_mode()` instead of `torch.no_grad()` for better performance.

## Benefits of inference_mode

`torch.inference_mode()` is faster and more memory-efficient than `torch.no_grad()`:
- Disables view tracking (no version counter updates)
- Prevents accidental gradient computation
- Lower overhead than autograd disable
- Available in PyTorch 1.9+

## Implementation

### Kronos Wrapper
`external/kronos/model/kronos.py:58-62`
```python
def _inference_context():
    context_ctor = getattr(torch, "inference_mode", None)
    if callable(context_ctor):
        return context_ctor()
    return torch.no_grad()  # fallback for old torch
```

Used in `auto_regressive_inference()` at line 443.

### Toto Wrapper
`src/models/toto_wrapper.py:179-185`
```python
def _inference_context() -> ContextManager[None]:
    """Return the best available inference context manager (inference_mode or no_grad)."""
    torch_module = _require_torch()
    context_ctor = getattr(torch_module, "inference_mode", None)
    if callable(context_ctor):
        return cast(ContextManager[None], context_ctor())
    return cast(ContextManager[None], torch_module.no_grad())
```

Used in `_forecast_with_retries()` at line 232.

### Chronos2 Pipeline
`chronos-forecasting/src/chronos/chronos2/pipeline.py`

Updated to use `@torch.inference_mode()` decorator:
- Line 378: `@torch.inference_mode()` on `predict()` method
- Line 652: `with torch.inference_mode():` around model call

### Chronos2 Wrapper
`src/models/chronos2_wrapper.py`

Calls `pipeline.predict_df()` which is covered by the decorator above.

## Pattern

All wrappers follow this pattern:

1. **Helper function**: `_inference_context()` that prefers `inference_mode` over `no_grad`
2. **Automatic fallback**: Falls back to `no_grad()` on old PyTorch versions
3. **Context manager**: Uses `with _inference_context():` around model calls
4. **Decorator option**: Can also use `@torch.inference_mode()` on methods

## Verification

To verify inference_mode is being used:

```python
import torch
print(f"torch.is_inference_mode_enabled(): {torch.is_inference_mode_enabled()}")

# During model prediction:
with torch.inference_mode():
    assert torch.is_inference_mode_enabled()
    # model forward pass here
```

## Performance Impact

Approximate speedup from using `inference_mode` vs `no_grad`:
- Memory: 5-10% reduction in peak usage
- Speed: 2-5% faster inference (varies by model)
- Most benefit comes from large models with many intermediate tensors

Combined with:
- `torch.compile()`: 1.5-2x speedup
- `bfloat16`: 1.3-1.5x speedup
- Total: 2-4x faster inference

## Migration Notes

If adding new model wrappers:

1. Import torch optionally:
```python
try:
    import torch
except ImportError:
    torch = None
```

2. Use the inference context helper:
```python
def _inference_context():
    if torch is None:
        from contextlib import nullcontext
        return nullcontext()
    context_ctor = getattr(torch, "inference_mode", None)
    if callable(context_ctor):
        return context_ctor()
    return torch.no_grad()
```

3. Wrap prediction methods:
```python
def predict(self, ...):
    with _inference_context():
        # prediction code
        ...
```

## Related Files

- `src/models/toto_wrapper.py` - Toto model wrapper
- `src/models/kronos_wrapper.py` - Kronos model wrapper
- `src/models/chronos2_wrapper.py` - Chronos2 model wrapper
- `chronos-forecasting/src/chronos/chronos2/pipeline.py` - Chronos2 pipeline
- `external/kronos/model/kronos.py` - Kronos core implementation
