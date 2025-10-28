# cppsimulator

High-performance market simulator implemented in C++17 with LibTorch tensors. The simulator keeps all state on device (CPU or CUDA) and exposes a vectorised `step()` API suitable for reinforcement-learning workflows.

## Layout

- `include/` public headers (`market_sim.hpp`, `forecast.hpp`, `types.hpp`)
- `src/` simulator and forecast bridge implementations
- `apps/run_sim.cpp` synthetic demo that exercises the simulator
- `models/` placeholder directory for TorchScript exports (e.g. Chronos/Kronos/Toto)
- `data/` optional placeholder for pre-baked OHLC tensors or CSV inputs

## Building

1. Download LibTorch (CPU or CUDA) from <https://pytorch.org/get-started/locally/> and extract it.
2. Configure with `Torch_DIR` pointing to the extracted distribution, e.g.:

   ```bash
   cmake -S cppsimulator -B cppsimulator/build -DTorch_DIR=/opt/libtorch/share/cmake/Torch
   cmake --build cppsimulator/build -j
   ```

   Set `TORCH_CUDA_ARCH_LIST` (e.g. `8.9` for RTX 5090) before building if you are targeting CUDA.

3. Run the synthetic demo:

   ```bash
   ./cppsimulator/build/run_sim
   ```

The simulator constructor accepts preloaded OHLC tensors; for production you should pre-bake your market data and TorchScript models so that the hot loop stays entirely within C++/LibTorch.
