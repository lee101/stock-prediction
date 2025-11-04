"""
CONSERVATIVE OPTIMIZATION: async GPU transfers only (no batching changes)

This only adds non_blocking=True for GPU transfers, keeping the same
walk-forward prediction logic. This ensures no semantic changes to forecasts.

Expected speedup: 1.2-1.5x (modest but safe)
"""

def _compute_toto_forecast(
    symbol: str,
    target_key: str,
    price_frame: pd.DataFrame,
    current_last_price: float,
    toto_params: dict,
):
    """
    Generate Toto forecasts for a prepared price frame.
    Returns (predictions_tensor, band_tensor, predicted_absolute_last).

    OPTIMIZATION: Async GPU transfers (non_blocking=True) for better utilization.
    """
    global TOTO_DEVICE_OVERRIDE
    predictions_list: List[float] = []
    band_list: List[float] = []
    max_horizon = 7

    if price_frame.empty:
        return torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32), float(current_last_price)

    # Toto expects a context vector of historical targets; walk forward to build forecasts.
    for pred_idx in reversed(range(1, max_horizon + 1)):
        if len(price_frame) <= pred_idx:
            continue
        current_context = price_frame[:-pred_idx]
        if current_context.empty:
            continue

        # OPTIMIZATION: Async GPU transfer
        context = torch.tensor(current_context["y"].values, dtype=torch.float32)
        if torch.cuda.is_available() and context.device.type == 'cpu':
            context = context.to('cuda', non_blocking=True)

        requested_num_samples = int(toto_params["num_samples"])
        requested_batch = int(toto_params["samples_per_batch"])

        attempts = 0
        cpu_fallback_used = False
        while True:
            requested_num_samples, requested_batch = _normalise_sampling_params(
                requested_num_samples,
                requested_batch,
            )
            toto_params["num_samples"] = requested_num_samples
            toto_params["samples_per_batch"] = requested_batch
            _toto_params_cache[symbol] = toto_params.copy()
            try:
                forecast = cached_predict(
                    context,
                    1,
                    num_samples=requested_num_samples,
                    samples_per_batch=requested_batch,
                    symbol=symbol,
                )
                break
            except RuntimeError as exc:
                if not _is_cuda_oom_error(exc) or attempts >= TOTO_BACKTEST_MAX_RETRIES:
                    if not _is_cuda_oom_error(exc):
                        raise
                    if cpu_fallback_used:
                        raise
                    logger.warning(
                        "Toto forecast OOM for %s %s after %d GPU retries; falling back to CPU inference.",
                        symbol,
                        target_key,
                        attempts,
                    )
                    cpu_fallback_used = True
                    TOTO_DEVICE_OVERRIDE = "cpu"
                    _drop_toto_pipeline()
                    attempts = 0
                    requested_num_samples = max(TOTO_MIN_NUM_SAMPLES, requested_num_samples // 2)
                    requested_batch = max(TOTO_MIN_SAMPLES_PER_BATCH, requested_batch // 2)
                    continue
                attempts += 1
                requested_num_samples = max(
                    TOTO_MIN_NUM_SAMPLES,
                    requested_num_samples // 2,
                )
                requested_batch = max(
                    TOTO_MIN_SAMPLES_PER_BATCH,
                    min(requested_batch // 2, requested_num_samples),
                )
                logger.warning(
                    "Toto forecast OOM for %s %s; retrying with num_samples=%d, samples_per_batch=%d (attempt %d/%d).",
                    symbol,
                    target_key,
                    requested_num_samples,
                    requested_batch,
                    attempts,
                    TOTO_BACKTEST_MAX_RETRIES,
                )
                continue

        updated_params = _apply_toto_runtime_feedback(symbol, toto_params, requested_num_samples, requested_batch)
        if updated_params is not None:
            toto_params = updated_params
        tensor = forecast[0]
        numpy_method = getattr(tensor, "numpy", None)
        if callable(numpy_method):
            try:
                array_data = numpy_method()
            except Exception:
                array_data = None
        else:
            array_data = None

        if array_data is None:
            detach_method = getattr(tensor, "detach", None)
            if callable(detach_method):
                try:
                    array_data = detach_method().cpu().numpy()
                except Exception:
                    array_data = None

        if array_data is None:
            array_data = tensor

        distribution = np.asarray(array_data, dtype=np.float32).reshape(-1)
        if distribution.size == 0:
            distribution = np.zeros(1, dtype=np.float32)

        lower_q = np.percentile(distribution, 40)
        upper_q = np.percentile(distribution, 60)
        band_width = float(max(upper_q - lower_q, 0.0))
        band_list.append(band_width)

        aggregated = aggregate_with_spec(distribution, toto_params["aggregate"])
        predictions_list.append(float(np.atleast_1d(aggregated)[0]))

    if not predictions_list:
        predictions_list = [0.0]
    if not band_list:
        band_list = [0.0]

    predictions = torch.tensor(predictions_list, dtype=torch.float32)
    bands = torch.tensor(band_list, dtype=torch.float32)
    predicted_absolute_last = float(current_last_price * (1.0 + predictions[-1].item()))
    return predictions, bands, predicted_absolute_last
