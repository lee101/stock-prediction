# Differentiable Market + Toto Embedding

This package mirrors the core differentiable market trainer while augmenting
each asset/timestep with a frozen Toto embedding. The Toto backbone is loaded
once, materialises embeddings for the requested context window, and the RL
policy remains the only trainable component.

Use `diff-market-toto-train` to launch experiments. Helpful flags:

- `--toto-context-length`: sliding window length used to build Toto inputs
- `--disable-real-toto`: skip loading the official Toto weights and fall back
  to the lightweight transformer if the dependency stack is unavailable
- `--toto-cache-dir`: path for materialised embeddings; set `--disable-toto-cache`
  to force on-the-fly regeneration

See `differentiable_market_totoembedding/train.py` for the full CLI.
