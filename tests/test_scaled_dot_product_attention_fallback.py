import importlib

import torch


train_hf = importlib.import_module("hftraining.train_hf")


def _force_fallback():
    """Temporarily force the fallback path by replacing the native kernel."""

    original = train_hf._NATIVE_SCALED_DOT_PRODUCT_ATTENTION

    def _raise(*args, **kwargs):  # noqa: D401 - short helper
        raise RuntimeError("scaled dot product attention not implemented on CPU")

    train_hf._NATIVE_SCALED_DOT_PRODUCT_ATTENTION = _raise
    return original


def _restore_native(original):
    train_hf._NATIVE_SCALED_DOT_PRODUCT_ATTENTION = original


def test_scaled_dot_product_attention_fallback_bool_mask_matches_reference():
    torch.manual_seed(123)
    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 4, 8)
    v = torch.randn(2, 1, 4, 8)
    attn_mask = torch.rand(2, 1, 4, 4) > 0.5

    rng_state = torch.random.get_rng_state()
    expected = train_hf._scaled_dot_product_attention_reference(
        q, k, v, attn_mask=attn_mask, dropout_p=0.1, is_causal=True
    )

    original = _force_fallback()
    try:
        torch.random.set_rng_state(rng_state)
        result = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.1, is_causal=True
        )
    finally:
        _restore_native(original)

    torch.testing.assert_close(result, expected, equal_nan=True)


def test_scaled_dot_product_attention_fallback_respects_no_grad_dropout():
    torch.manual_seed(321)
    q = torch.randn(1, 2, 3, 5)
    k = torch.randn(1, 2, 3, 5)
    v = torch.randn(1, 2, 3, 5)
    attn_mask = torch.randn(1, 2, 3, 3)

    with torch.no_grad():
        rng_state = torch.random.get_rng_state()
        expected = train_hf._scaled_dot_product_attention_reference(
            q, k, v, attn_mask=attn_mask, dropout_p=0.2, is_causal=False
        )

    original = _force_fallback()
    try:
        with torch.no_grad():
            torch.random.set_rng_state(rng_state)
            result = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.2, is_causal=False
            )
    finally:
        _restore_native(original)

    torch.testing.assert_close(result, expected, equal_nan=True)
