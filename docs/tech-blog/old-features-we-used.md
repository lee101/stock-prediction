# Old Features We Used: From KNN Baselines to Neural Trading + Chronos2

Early versions of the trading stack leaned on a classic baseline: k-nearest neighbors (KNN). It was fast to prototype, easy to debug, and gave us a surprisingly strong sanity check when we were still validating data pipelines and feature construction. Once we had enough history and a broader symbol universe, we moved to neural learned trading with Chronos2 driving the forecasting layer. This post documents what the old KNN workflow looked like, why it worked, and what replaced it.

## The KNN baseline we started with

KNN is a non-parametric algorithm: it makes predictions by comparing a new example to the most similar examples in a stored history and then aggregating their outcomes. For forecasting, that aggregation is typically an average (for regression) or a vote (for classification). The method is simple, interpretable, and often strong as a baseline because it does not assume a parametric model form. It also emphasizes feature scaling and distance metrics, both of which helped us clarify what our model should pay attention to. Sources: Wikipedia and scikit-learn provide the canonical KNN formulation and practical behavior for regression use cases.

Our KNN variant was a time-series similarity lookup:

1. Build a feature vector for each historical window. Features were derived from the recent OHLC path, including percent returns, range expansion/contraction, and simple volatility proxies.
2. Normalize features so that distances reflected shape, not magnitude.
3. Compute distances between the current window and all prior windows.
4. Select the k closest neighbors.
5. Use the neighbor set to form a forecast distribution (next close, next high/low range, and expected return).
6. Convert the forecast to simple trade rules (e.g., buy if expected return above a threshold and the range skew was favorable).

We used KNN because it was easy to reason about:
- When predictions were wrong, we could inspect the actual neighbor windows.
- It forced us to get feature engineering and data normalization correct early.
- It made it obvious when we were matching on the wrong regime (e.g., volatility spikes).

## Where KNN started to break down

The same properties that make KNN simple can become liabilities:

- **Curse of dimensionality:** As we added more features and longer context windows, distance measures became less meaningful, and neighbor quality degraded.
- **Regime shifts:** KNN does not learn a representation; it copies what is nearby. When regimes shift, KNN can anchor to outdated patterns.
- **Latency and scale:** Exact neighbor searches are O(N) per query. Even with indexing tricks, it was expensive to run for larger universes or frequent updates.
- **Weak uncertainty modeling:** KNN provides a distribution via neighbors, but it is not a calibrated probabilistic model.

The result was a system that was great for quick checks but too brittle for production-grade forecasting and decision making.

## The replacement: neural learned trading + Chronos2

We replaced the KNN baseline with a two-part system:

1. **Neural forecasting with Chronos2**
2. **Neural learned trading policies** (position sizing and selection)

Chronos2 is a foundation model for time series forecasting that builds on the Chronos and Chronos-Bolt work. It can produce multi-step forecasts and quantile outputs, which are crucial for expressing uncertainty and for downstream risk-aware decision rules. The Chronos2 pipeline is designed to handle diverse time series and can be deployed as an inference service or run locally for batch generation. Sources: the Chronos2 arXiv paper and the Amazon Science / AWS documentation.

What changed with Chronos2:
- We moved from neighbor matching to a learned representation of time series dynamics.
- We gained quantile forecasts (not just point estimates), making risk controls more principled.
- We reduced manual feature engineering and let the model learn the relevant structure.

### Neural learned trading
On top of the forecast layer, we introduced neural policies that learn how to allocate and size trades from forecast signals. This let the system learn non-linear relationships between forecast confidence, expected return, volatility, and position sizing. The output is a set of actionable decisions: what to buy, how much, and at what risk budget.

## What we kept from the KNN era

The KNN baseline still influenced the system in a few ways:
- **Sanity checks:** We keep a lightweight similarity baseline for validation when retraining or adding new symbols.
- **Feature audits:** The KNN feature pipeline doubles as a diagnostic layer to detect data drift.
- **Interpretability:** The “neighbor window” lens remains useful for human review.

## Summary

KNN was the right tool for our first iterations: it was quick, transparent, and forced us to clean up our data and features. But as the system scaled, the limitations became too painful. Chronos2 and neural learned trading now provide the forecasting accuracy, uncertainty modeling, and decision quality we need for production.

## References

- https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
- https://scikit-learn.org/stable/modules/neighbors.html
- https://arxiv.org/abs/2403.07815
- https://www.amazon.science/blog/chronos-2-a-foundation-model-for-time-series-forecasting
- https://github.com/amazon-science/chronos-forecasting
